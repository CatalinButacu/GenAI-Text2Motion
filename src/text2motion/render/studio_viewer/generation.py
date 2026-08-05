from __future__ import annotations

import threading
import time
import traceback

import numpy as np

from text2motion.render.studio_viewer.constants import LIVE_DEBOUNCE_S, LIVE_STEPS
from text2motion.render.studio_viewer.registry import ModelEntry


class GenerationControl:
    def __init__(self, host, state, avatar) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar

    def submit_prompt(self) -> None:
        prompt = self.state.prompt_text.strip()
        if not prompt or self.state.generating:
            return
        self.state.live.last_dispatched = prompt
        self._start_generation(prompt, live=False)

    def _start_generation(self, prompt: str, live: bool) -> None:
        self.state.generating = True
        self.state.buf.cancel = False
        self.state.buf.gen_progress = 0.0
        self.state.buf.fit_progress = None
        if live:
            steps = min(self.state.steps, LIVE_STEPS)
            cfg_scale = 1.0
            fit_body = False
            self.state.status = f'live: "{prompt}"'
        else:
            steps = self.state.steps
            cfg_scale = self.state.cfg_scale
            fit_body = self.state.fit_body and bool(self.state.model_dir)
            self.state.status = f'generating: "{prompt}"'
            self.state.history.append(prompt)
        params = {
            "temperature": self.state.temperature,
            "top_p": self.state.top_p,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "fit_body": fit_body,
            "downsample": self.state.downsample,
        }
        self.state.buf.current_fit_body = fit_body
        self.state.buf.drop_skeleton = False
        with self.state.buf.lock:
            self.state.buf.new_chunks.clear()
            self.state.buf.pending_body_chunks = []
            self.state.buf.pending_remesh = None
            self.state.buf.started = False
            self.state.buf.accum_joints = None
            self.state.fit.orient = []
            self.state.fit.pose = []
            self.state.fit.transl = []
            self.state.fit.betas = None
        threading.Thread(target=self._worker, args=(prompt, params), daemon=True).start()

    def _tick_live(self) -> None:
        if not self.state.live.enabled:
            return
        sig = (self.state.temperature, self.state.top_p, self.state.steps)
        if sig != self.state.live.signature:
            self.state.live.signature = sig
            if self.state.prompt_text.strip():
                self.state.live.mark_dirty(self.state.prompt_text, time.perf_counter())
                self.state.live.last_dispatched = ""
        prompt = self.state.prompt_text.strip()
        ready = self.state.live.debounce_elapsed(time.perf_counter(), LIVE_DEBOUNCE_S)
        if prompt and ready and prompt != self.state.live.last_dispatched:
            self.state.live.prev_prompt = ""
            self.state.live.last_dispatched = prompt
            if self.state.generating:
                self.state.buf.cancel = True
                self.state.live.regen_pending = True
            else:
                self._start_generation(prompt, live=True)
        if self.state.live.regen_pending and not self.state.generating:
            self.state.live.regen_pending = False
            if self.state.prompt_text.strip():
                self._start_generation(self.state.prompt_text.strip(), live=True)

    def _worker(self, prompt: str, params: dict) -> None:
        n_frames = 0
        total_frames = params["steps"] * params["downsample"]
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        try:
            print(
                f"[GEN] request prompt={prompt!r} steps={params['steps']} "
                f"cfg={params['cfg_scale']} temp={params['temperature']} "
                f"fit_body={params['fit_body']} -> motion service",
                flush=True,
            )
            chunk_iter = self.state.client.generate(
                prompt,
                steps=params["steps"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                cfg_scale=params["cfg_scale"],
                should_cancel=lambda: self.state.buf.cancel,
            )
            for joints in chunk_iter:
                assert joints.ndim == 3 and joints.shape[1:] == (22, 3), (
                    f"bad joints {joints.shape}"
                )
                n_frames += len(joints)
                print(
                    f"[GEN] chunk joints={joints.shape} n_frames={n_frames} "
                    f"finite={np.isfinite(joints).all()}",
                    flush=True,
                )
                self.state.buf.gen_progress = min(1.0, n_frames / total_frames)
                with self.state.buf.lock:
                    self.state.buf.new_chunks.append(joints)
                if params["fit_body"]:
                    self.state.status = f"streaming... {n_frames} frames"
                    warm_start = self.avatar._fit_chunk(joints, warm_start, n_frames)
                else:
                    self.state.status = (
                        f"streaming... {n_frames} frames (skeleton only -- tick"
                        " 'avatar skin' for the body)"
                    )
        except Exception as exc:
            traceback.print_exc()
            self.state.status = f"error after {n_frames} frames: {type(exc).__name__}: {exc}"
            self.state.generating = False
            return

        print(
            f"[GEN] worker DONE total_frames={n_frames} fit_body={params['fit_body']}", flush=True
        )
        duration = n_frames / self.host.playback_fps
        verb = "stopped" if self.state.buf.cancel else "done"
        suffix = "" if params["fit_body"] else ", skeleton"
        self.state.status = f"{verb} -- {n_frames} frames (~{duration:.1f}s{suffix})"
        self.state.buf.fit_progress = None
        if params["fit_body"] and not self.state.buf.cancel:
            self.state.buf.drop_skeleton = True
        if not self.host.run_animations:
            self.host.toggle_animation(True)
        self.state.generating = False
        if params["fit_body"] and self.state.fit.after_gen and self.state.fit.orient:
            self.state.fit.after_gen = False
            self.avatar._schedule_remesh()

    def _start_model_load(self) -> None:
        entry = self.state.models[self.state.model_idx]
        self.state.loading_model = True
        self.state.load_status = f"loading {entry.label}..."
        threading.Thread(target=self._load_worker, args=(entry,), daemon=True).start()

    def _load_worker(self, entry: ModelEntry) -> None:
        try:
            hello = self.state.client.load(
                entry.config, entry.ckpt, entry.tokenizer_ckpt, entry.backbone
            )
            self.state.backbone = entry.backbone
            self.state.downsample = int(hello["downsample"])
            self.state.max_steps = int(hello["max_steps"])
            self.state.steps = min(self.state.steps, self.state.max_steps)
            self.state.load_status = f"active: {entry.label}"
        except Exception as exc:
            traceback.print_exc()
            self.state.load_status = f"load failed: {type(exc).__name__}: {exc}"
        finally:
            self.state.loading_model = False
