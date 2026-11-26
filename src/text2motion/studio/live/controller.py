from __future__ import annotations

import logging
import queue
import threading
import time
import traceback

import numpy as np

from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.live.actions import (
    log_action,
    log_result,
)
from text2motion.studio.live.body_fitter import BodyFitter
from text2motion.studio.live.chunk_buffers import prepare_joints, publish_chunk
from text2motion.studio.live.model_loader import ModelLoader
from text2motion.studio.live.playback_pacer import PlaybackPacer
from text2motion.studio.live.request_setup import (
    request_chunks,
    request_params,
    request_settings,
    reset_fit_buffers,
    reset_stream_progress,
)

logger = logging.getLogger(__name__)


############################################################
#       StudioGenerationController
############################################################
class StudioGenerationController:
    def __init__(self, host: ViewerHost, state, avatar: SmplxAvatarController) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar
        self.fitter = BodyFitter(state, avatar)
        self.loader = ModelLoader(state)
        self.pacer = PlaybackPacer(host, state)

    def submit_prompt(self) -> None:
        prompt = self.state.prompt_text.strip()
        if prompt and not self.state.generating:
            self.state.live.last_dispatched = prompt
            self.start_generation(prompt, live=False)

    def start_generation(self, prompt: str, live: bool) -> None:
        self.state.generating = True
        reset_stream_progress(self.state)
        steps, cfg_scale, fit_body = request_settings(self.state, prompt, live)
        log_action(
            "generate", f"{prompt!r} steps={steps or 'forever'} cfg={cfg_scale} skin={fit_body}"
        )
        params = request_params(self.state, steps, cfg_scale, fit_body)
        self.state.stream_buffers.current_fit_body = fit_body
        reset_fit_buffers(self.state)
        threading.Thread(target=self.stream_generation, args=(prompt, params), daemon=True).start()

    def tick_live(self) -> None:
        if not self.state.live.enabled:
            return
        signature = (self.state.temperature, self.state.top_p, self.state.steps)
        if signature != self.state.live.signature:
            self.state.live.signature = signature
            if self.state.prompt_text.strip():
                self.state.live.mark_dirty(self.state.prompt_text, time.perf_counter())
                self.state.live.last_dispatched = ""
        prompt = self.state.prompt_text.strip()
        ready = self.state.live.debounce_elapsed(
            time.perf_counter(), self.state.config.live.debounce_seconds
        )
        if prompt and ready and prompt != self.state.live.last_dispatched:
            self.state.live.prev_prompt = ""
            self.state.live.last_dispatched = prompt
            if self.state.generating:
                self.state.client.steer(prompt)
                self.state.status = f'live: steering to "{prompt}"'
            else:
                self.start_generation(prompt, live=True)
        if self.state.live.regen_pending and not self.state.generating:
            self.state.live.regen_pending = False
            if prompt:
                self.start_generation(prompt, live=True)

    def stop(self) -> None:
        self.state.stream_buffers.cancel = True
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.new_chunks.clear()
        self.state.status = "stopping..."

    def finish(self, n_frames: int, params: dict, fit_died: bool = False) -> None:
        if not self.state.generating:
            return
        duration = n_frames / self.host.playback_fps
        verb = "stopped" if self.state.stream_buffers.cancel else "done"
        log_result("generation", True, f"{verb}, {n_frames} frames (~{duration:.1f}s)")
        stats = getattr(self.state.client, "last_stats", None)
        if stats:
            log_action(
                "inference",
                f"{stats['realtime_factor']:.1f}x realtime, {stats['ms_per_step']:.1f} ms/step, "
                f"encode {stats['text_encode_ms']:.0f}ms generate {stats['generate_ms']:.0f}ms "
                f"detokenize {stats['detokenize_ms']:.0f}ms, peak gpu {stats['peak_gpu_gb']:.2f}GB",
            )
        suffix = "" if params["fit_body"] else ", skeleton"
        if fit_died:
            suffix = ", skeleton -- body fit failed, see the log"
        self.state.status = f"{verb} -- {n_frames} frames (~{duration:.1f}s{suffix})"
        if not params["fit_body"]:
            self.state.stream_buffers.fit_progress = None
        if not self.host.run_animations:
            self.host.toggle_animation(True)
        self.state.generating = False
        if params["fit_body"] and self.state.smplx_fit.after_gen and self.state.smplx_fit.orient:
            self.state.smplx_fit.after_gen = False
            self.avatar.mesh.schedule_remesh()

    def stream_generation(self, prompt: str, params: dict) -> None:
        n_frames = 0
        total_frames = params["steps"] * params["downsample"] * params["upsample"]
        fit_queue, fit_thread = self.fitter.start_fit_thread(params["fit_body"])
        try:
            logger.info(
                f"[GEN] request prompt={prompt!r} steps={params['steps']} "
                f"cfg={params['cfg_scale']} temp={params['temperature']} "
                f"fit_body={params['fit_body']} -> motion service"
            )
            for chunk in request_chunks(self.state, prompt, params):
                n_frames += self.accept_chunk(chunk, n_frames, total_frames, fit_queue, fit_thread)
        except Exception as exc:
            traceback.print_exc()
            log_result("generation", False, f"{type(exc).__name__}: {exc}")
            self.state.status = f"error after {n_frames} frames: {type(exc).__name__}: {exc}"
            self.fitter.stop_fitter(fit_queue, fit_thread, True)
            self.state.generating = False
            return

        logger.info(f"[GEN] worker DONE total_frames={n_frames} fit_body={params['fit_body']}")
        fit_died = fit_thread is not None and not fit_thread.is_alive()
        self.fitter.stop_fitter(fit_queue, fit_thread, self.state.stream_buffers.cancel)
        self.finish(n_frames, params, fit_died)

    def accept_chunk(
        self,
        chunk,
        n_frames: int,
        total_frames: int,
        fit_queue: queue.Queue | None,
        fit_thread: threading.Thread | None,
    ) -> int:
        self.pacer.wait_for_playback(n_frames)
        if self.state.stream_buffers.cancel:
            return 0
        joints = prepare_joints(self.state, chunk)
        emitted = n_frames + len(joints)
        logger.info(
            f"[GEN] chunk joints={joints.shape} n_frames={emitted} "
            f"finite={np.isfinite(joints).all()}"
        )
        publish_chunk(self.state, joints, emitted, total_frames)
        if fit_queue is not None and fit_thread is not None:
            self.fitter.feed_fitter(fit_queue, fit_thread, chunk)
            if fit_thread.is_alive():
                self.state.status = f"streaming... {emitted} frames"
        else:
            self.state.status = (
                f"streaming... {emitted} frames (skeleton only -- tick 'avatar skin' for the body)"
            )
        return len(joints)
