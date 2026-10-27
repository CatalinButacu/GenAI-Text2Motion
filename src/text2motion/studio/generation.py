from __future__ import annotations

import logging
import queue
import threading
import time
import traceback

import numpy as np

from text2motion.motion.representation import JOINTS
from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.scene import StudioModelEntry, live_step_budget

logger = logging.getLogger(__name__)


def log_action(name: str, detail: str = "") -> None:
    logger.info(f"[UI] {name}{f' {detail}' if detail else ''}")


def log_toggle(name: str, value: bool) -> None:
    logger.info(f"[UI] {name} -> {'ON' if value else 'OFF'}")


def log_result(name: str, ok: bool, detail: str = "") -> None:
    logger.info(f"[UI] {name} -> {'OK' if ok else 'FAILED'}{f' ({detail})' if detail else ''}")


def upsample_joints(previous: np.ndarray | None, joints: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return joints
    start = joints[:1] if previous is None else previous[None]
    anchors = np.concatenate([start, joints], axis=0)
    weights = (np.arange(1, factor + 1, dtype=np.float32) / factor)[None, :, None, None]
    begin = anchors[:-1, None]
    end = anchors[1:, None]
    blended = begin + (end - begin) * weights
    return blended.reshape(len(joints) * factor, *joints.shape[1:]).astype(joints.dtype)


def discard_pending_items(pending: queue.Queue) -> None:
    try:
        while True:
            pending.get_nowait()
    except queue.Empty:
        pass


############################################################
#       StudioGenerationController
############################################################


class StudioGenerationController:
    def __init__(self, host: ViewerHost, state, avatar: SmplxAvatarController) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar

    def submit_prompt(self) -> None:
        prompt = self.state.prompt_text.strip()
        if prompt and not self.state.generating:
            self.state.live.last_dispatched = prompt
            self.__start_generation(prompt, live=False)

    def __start_generation(self, prompt: str, live: bool) -> None:
        self.state.generating = True
        self.state.stream_buffers.cancel = False
        self.state.stream_buffers.gen_progress = 0.0
        self.state.stream_buffers.frames_emitted = 0
        self.state.stream_buffers.fit_progress = None
        if live:
            steps = live_step_budget(self.state.steps, self.state.config.live.steps)
            cfg_scale = 1.0
            fit_body = False
            self.state.status = f'live: "{prompt}"'
        else:
            steps = self.state.steps
            cfg_scale = self.state.cfg_scale
            fit_body = self.state.fit_body and bool(self.state.model_dir)
            self.state.status = (
                f'generating forever: "{prompt}" -- press Stop to end'
                if steps == 0
                else f'generating: "{prompt}"'
            )
            self.state.history.append(prompt)
        log_action(
            "generate", f"{prompt!r} steps={steps or 'forever'} cfg={cfg_scale} skin={fit_body}"
        )
        params = {
            "temperature": self.state.temperature,
            "top_p": self.state.top_p,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "fit_body": fit_body,
            "downsample": self.state.downsample,
            "upsample": self.state.upsample,
        }
        self.state.stream_buffers.current_fit_body = fit_body
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.take += 1
            self.state.stream_buffers.interpolation_anchor = None
            self.state.stream_buffers.fit_anchor = None
            self.state.stream_buffers.new_chunks.clear()
            self.state.stream_buffers.pending_body_chunks = []
            self.state.stream_buffers.pending_remesh = None
            self.state.stream_buffers.started = False
            self.state.stream_buffers.accum_joints = None
            self.state.stream_buffers.source_joints = None
            self.state.smplx_fit.orient = []
            self.state.smplx_fit.pose = []
            self.state.smplx_fit.transl = []
            self.state.smplx_fit.betas = None
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
                self.state.stream_buffers.cancel = True
                self.state.live.regen_pending = True
            else:
                self.__start_generation(prompt, live=True)
        if self.state.live.regen_pending and not self.state.generating:
            self.state.live.regen_pending = False
            if prompt:
                self.__start_generation(prompt, live=True)

    def stop(self) -> None:
        self.state.stream_buffers.cancel = True
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.new_chunks.clear()
        self.state.status = "stopping..."

    def __finish(self, n_frames: int, params: dict, fit_died: bool = False) -> None:
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
            self.avatar.schedule_remesh()

    def fit_motion_chunks(self, fit_queue: queue.Queue, take: int) -> None:
        buffers = self.state.stream_buffers
        buffers.fit_take = take
        buffers.fitting = True
        n_frames = 0
        try:
            while True:
                chunk = fit_queue.get()
                if chunk is None:
                    break
                if buffers.fit_is_stale():
                    continue
                n_frames += len(chunk.joints) * self.state.upsample
                self.avatar.fit_single_chunk(chunk.joints, n_frames, self.state.upsample)
        except Exception as exc:
            traceback.print_exc()
            log_result("body fit", False, f"{type(exc).__name__}: {exc}")
            self.state.status = f"body fit failed: {type(exc).__name__}: {exc}"
        else:
            log_result("body fit", True, f"{n_frames} frames")
            if buffers.fit_take == take and not self.state.generating and not buffers.cancel:
                self.state.status = f"done -- body fitted on {n_frames} frames"
        finally:
            if buffers.fit_take == take:
                buffers.fitting = False
                buffers.fit_progress = None

    def stream_generation(self, prompt: str, params: dict) -> None:
        n_frames = 0
        total_frames = params["steps"] * params["downsample"] * params["upsample"]
        fit_queue: queue.Queue | None = None
        fit_thread: threading.Thread | None = None
        if params["fit_body"]:
            fit_queue = queue.Queue(maxsize=self.state.fit_lookahead_chunks)
            take = self.state.stream_buffers.take
            fit_thread = threading.Thread(
                target=self.fit_motion_chunks, args=(fit_queue, take), daemon=True
            )
            fit_thread.start()
        try:
            logger.info(
                f"[GEN] request prompt={prompt!r} steps={params['steps']} "
                f"cfg={params['cfg_scale']} temp={params['temperature']} "
                f"fit_body={params['fit_body']} -> motion service"
            )
            chunks = self.state.client.generate(
                prompt,
                steps=params["steps"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                cfg_scale=params["cfg_scale"],
                should_cancel=lambda: self.state.stream_buffers.cancel,
            )
            for chunk in chunks:
                n_frames += self.__accept_chunk(
                    chunk, n_frames, total_frames, fit_queue, fit_thread
                )
        except Exception as exc:
            traceback.print_exc()
            log_result("generation", False, f"{type(exc).__name__}: {exc}")
            self.state.status = f"error after {n_frames} frames: {type(exc).__name__}: {exc}"
            if fit_queue is not None and fit_thread is not None:
                discard_pending_items(fit_queue)
                self.__close_fitter(fit_queue, fit_thread)
            self.state.generating = False
            return

        logger.info(f"[GEN] worker DONE total_frames={n_frames} fit_body={params['fit_body']}")
        fit_died = fit_thread is not None and not fit_thread.is_alive()
        if fit_queue is not None and fit_thread is not None:
            if self.state.stream_buffers.cancel:
                discard_pending_items(fit_queue)
            self.__close_fitter(fit_queue, fit_thread)
        self.__finish(n_frames, params, fit_died)

    def __feed_fitter(self, fit_queue: queue.Queue, fit_thread: threading.Thread, chunk) -> None:
        while not self.state.stream_buffers.cancel and fit_thread.is_alive():
            try:
                fit_queue.put(chunk, timeout=0.2)
                return
            except queue.Full:
                continue

    def __close_fitter(self, fit_queue: queue.Queue, fit_thread: threading.Thread) -> None:
        while fit_thread.is_alive():
            try:
                fit_queue.put(None, timeout=0.2)
                return
            except queue.Full:
                continue

    def frames_ahead_of_playhead(self, accepted_frames: int) -> int:
        buffers = self.state.stream_buffers
        played = buffers.trimmed_frames + self.host.scene.current_frame_id
        return accepted_frames - played

    def wait_for_playback(self, accepted_frames: int) -> None:
        while (
            not self.state.stream_buffers.cancel
            and self.frames_ahead_of_playhead(accepted_frames)
            > self.state.stream_lead_frames * self.state.upsample
        ):
            time.sleep(0.01)

    def __accept_chunk(
        self,
        chunk,
        n_frames: int,
        total_frames: int,
        fit_queue: queue.Queue | None,
        fit_thread: threading.Thread | None,
    ) -> int:
        self.wait_for_playback(n_frames)
        if self.state.stream_buffers.cancel:
            return 0
        buffers = self.state.stream_buffers
        joints = upsample_joints(buffers.interpolation_anchor, chunk.joints, self.state.upsample)
        buffers.interpolation_anchor = chunk.joints[-1]
        buffers.add_source_joints(chunk.joints)
        assert joints.ndim == 3 and joints.shape[1:] == (JOINTS, 3), f"bad joints {joints.shape}"
        emitted = n_frames + len(joints)
        logger.info(
            f"[GEN] chunk joints={joints.shape} n_frames={emitted} "
            f"finite={np.isfinite(joints).all()}"
        )
        self.state.stream_buffers.frames_emitted = emitted
        self.state.stream_buffers.gen_progress = (
            0.0 if total_frames == 0 else min(1.0, emitted / total_frames)
        )
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.new_chunks.append(joints)
        if fit_queue is not None and fit_thread is not None:
            self.__feed_fitter(fit_queue, fit_thread, chunk)
            if fit_thread.is_alive():
                self.state.status = f"streaming... {emitted} frames"
        else:
            self.state.status = (
                f"streaming... {emitted} frames (skeleton only -- tick 'avatar skin' for the body)"
            )
        return len(joints)

    def start_model_load(self) -> None:
        entry = self.state.models[self.state.model_idx]
        self.state.loading_model = True
        self.state.load_status = f"loading {entry.label}..."
        threading.Thread(target=self.__load_selected_model, args=(entry,), daemon=True).start()

    def __load_selected_model(self, entry: StudioModelEntry) -> None:
        try:
            hello = self.state.client.load(
                entry.config, entry.ckpt, entry.tokenizer_ckpt, entry.backbone
            )
            self.state.backbone = entry.backbone
            self.state.apply_model_limits(hello)
            if self.state.max_steps:
                capped = min(self.state.steps or self.state.max_steps, self.state.max_steps)
                if capped != self.state.steps:
                    self.state.status = (
                        f"{entry.label} cannot run indefinitely -- clamped to {capped} steps"
                    )
                self.state.steps = capped
            self.state.load_status = f"active: {entry.label}"
            log_result("load model", True, entry.key)
        except Exception as exc:
            traceback.print_exc()
            log_result("load model", False, f"{type(exc).__name__}: {exc}")
            self.state.load_status = f"load failed: {type(exc).__name__}: {exc}"
        finally:
            self.state.loading_model = False
