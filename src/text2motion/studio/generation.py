from __future__ import annotations

import queue
import threading
import time
import traceback

import numpy as np

from text2motion.motion.representation import JOINTS
from text2motion.studio.actions import log_action, log_result
from text2motion.studio.avatar import SmplxAvatarController
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.scene import StudioModelEntry


def discard_pending_items(pending: queue.Queue) -> None:
    try:
        while True:
            pending.get_nowait()
    except queue.Empty:
        pass


class StudioGenerationController:
    def __init__(self, host: ViewerHost, state, avatar: SmplxAvatarController) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar

    def submit_prompt(self) -> None:
        prompt = self.state.prompt_text.strip()
        if prompt and not self.state.generating:
            self.state.live.last_dispatched = prompt
            self._start_generation(prompt, live=False)

    def _start_generation(self, prompt: str, live: bool) -> None:
        self.state.generating = True
        self.state.stream_buffers.cancel = False
        self.state.stream_buffers.gen_progress = 0.0
        self.state.stream_buffers.frames_emitted = 0
        self.state.stream_buffers.fit_progress = None
        if live:
            steps = min(self.state.steps, self.state.config.live.steps)
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
        log_action("generate", f"{prompt!r} steps={steps or 'forever'} cfg={cfg_scale} skin={fit_body}")
        params = {
            "temperature": self.state.temperature,
            "top_p": self.state.top_p,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "fit_body": fit_body,
            "downsample": self.state.downsample,
        }
        self.state.stream_buffers.current_fit_body = fit_body
        self.state.stream_buffers.drop_skeleton = False
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.new_chunks.clear()
            self.state.stream_buffers.pending_body_chunks = []
            self.state.stream_buffers.pending_remesh = None
            self.state.stream_buffers.started = False
            self.state.stream_buffers.accum_joints = None
            self.state.smplx_fit.orient = []
            self.state.smplx_fit.pose = []
            self.state.smplx_fit.transl = []
            self.state.smplx_fit.betas = None
        threading.Thread(target=self._stream_generation, args=(prompt, params), daemon=True).start()

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
                self._start_generation(prompt, live=True)
        if self.state.live.regen_pending and not self.state.generating:
            self.state.live.regen_pending = False
            if prompt:
                self._start_generation(prompt, live=True)

    def stop(self) -> None:
        self.state.stream_buffers.cancel = True
        self.state.status = "stopping..."

    def _finish(self, n_frames: int, params: dict) -> None:
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
        self.state.status = f"{verb} -- {n_frames} frames (~{duration:.1f}s{suffix})"
        self.state.stream_buffers.fit_progress = None
        if params["fit_body"] and not self.state.stream_buffers.cancel:
            self.state.stream_buffers.drop_skeleton = True
        if not self.host.run_animations:
            self.host.toggle_animation(True)
        self.state.generating = False
        if (
            params["fit_body"]
            and self.state.smplx_fit.after_gen
            and self.state.smplx_fit.orient
        ):
            self.state.smplx_fit.after_gen = False
            self.avatar._schedule_remesh()

    def _fit_motion_chunks(self, fit_queue: queue.Queue) -> None:
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        n_frames = 0
        try:
            while True:
                chunk = fit_queue.get()
                if chunk is None:
                    break
                if self.state.stream_buffers.cancel:
                    continue
                n_frames += len(chunk.joints)
                warm_start = self.avatar._fit_chunk(chunk.joints, warm_start, n_frames)
        except Exception as exc:
            traceback.print_exc()
            log_result("body fit", False, f"{type(exc).__name__}: {exc}")
            self.state.status = f"body fit failed: {type(exc).__name__}: {exc}"
        else:
            log_result("body fit", True, f"{n_frames} frames")

    def _stream_generation(self, prompt: str, params: dict) -> None:
        n_frames = 0
        total_frames = params["steps"] * params["downsample"]
        fit_queue: queue.Queue | None = None
        if params["fit_body"]:
            fit_queue = queue.Queue(maxsize=self.state.fit_lookahead_chunks)
            threading.Thread(target=self._fit_motion_chunks, args=(fit_queue,), daemon=True).start()
        try:
            print(
                f"[GEN] request prompt={prompt!r} steps={params['steps']} "
                f"cfg={params['cfg_scale']} temp={params['temperature']} "
                f"fit_body={params['fit_body']} -> motion service",
                flush=True,
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
                n_frames += self._accept_chunk(chunk, n_frames, total_frames, fit_queue)
        except Exception as exc:
            traceback.print_exc()
            log_result("generation", False, f"{type(exc).__name__}: {exc}")
            self.state.status = f"error after {n_frames} frames: {type(exc).__name__}: {exc}"
            if fit_queue is not None:
                discard_pending_items(fit_queue)
                fit_queue.put(None)
            self.state.generating = False
            return

        print(f"[GEN] worker DONE total_frames={n_frames} fit_body={params['fit_body']}", flush=True)
        if fit_queue is not None:
            discard_pending_items(fit_queue)
            fit_queue.put(None)
        self._finish(n_frames, params)

    def _accept_chunk(self, chunk, n_frames: int, total_frames: int, fit_queue) -> int:
        joints = chunk.joints
        assert joints.ndim == 3 and joints.shape[1:] == (JOINTS, 3), f"bad joints {joints.shape}"
        emitted = n_frames + len(joints)
        print(
            f"[GEN] chunk joints={joints.shape} n_frames={emitted} "
            f"finite={np.isfinite(joints).all()}",
            flush=True,
        )
        self.state.stream_buffers.frames_emitted = emitted
        self.state.stream_buffers.gen_progress = (
            0.0 if total_frames == 0 else min(1.0, emitted / total_frames)
        )
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.new_chunks.append(joints)
        if fit_queue is not None:
            while not self.state.stream_buffers.cancel:
                try:
                    fit_queue.put(chunk, timeout=0.2)
                    break
                except queue.Full:
                    continue
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
        threading.Thread(target=self._load_selected_model, args=(entry,), daemon=True).start()

    def _load_selected_model(self, entry: StudioModelEntry) -> None:
        try:
            hello = self.state.client.load(
                entry.config, entry.ckpt, entry.tokenizer_ckpt, entry.backbone
            )
            self.state.backbone = entry.backbone
            self.state.downsample = int(hello["downsample"])
            self.state.max_steps = int(hello["max_steps"])
            if self.state.max_steps:
                capped = min(self.state.steps or self.state.max_steps, self.state.max_steps)
                if capped != self.state.steps:
                    self.state.status = f"{entry.label} cannot run indefinitely -- clamped to {capped} steps"
                self.state.steps = capped
            self.state.load_status = f"active: {entry.label}"
            log_result("load model", True, entry.key)
        except Exception as exc:
            traceback.print_exc()
            log_result("load model", False, f"{type(exc).__name__}: {exc}")
            self.state.load_status = f"load failed: {type(exc).__name__}: {exc}"
        finally:
            self.state.loading_model = False
