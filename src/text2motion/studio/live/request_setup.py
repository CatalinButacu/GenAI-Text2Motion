from __future__ import annotations

from text2motion.studio.session.model_registry import live_step_budget


def reset_stream_progress(state) -> None:
    state.stream_buffers.cancel = False
    state.stream_buffers.gen_progress = 0.0
    state.stream_buffers.frames_emitted = 0
    state.stream_buffers.fit_progress = None


def request_settings(state, prompt: str, live: bool) -> tuple[int, float, bool]:
    if live:
        steps = live_step_budget(state.steps, state.config.live.steps)
        cfg_scale = 1.0
        fit_body = False
        state.status = f'live: "{prompt}"'
        return steps, cfg_scale, fit_body
    steps = state.steps
    cfg_scale = state.cfg_scale
    fit_body = state.fit_body and bool(state.model_dir)
    state.status = (
        f'generating forever: "{prompt}" -- press Stop to end'
        if steps == 0
        else f'generating: "{prompt}"'
    )
    state.history.append(prompt)
    return steps, cfg_scale, fit_body


def request_params(state, steps: int, cfg_scale: float, fit_body: bool) -> dict:
    return {
        "temperature": state.temperature,
        "top_p": state.top_p,
        "cfg_scale": cfg_scale,
        "steps": steps,
        "fit_body": fit_body,
        "downsample": state.downsample,
        "upsample": state.upsample,
    }


def reset_fit_buffers(state) -> None:
    with state.stream_buffers.lock:
        state.stream_buffers.take += 1
        state.stream_buffers.interpolation_anchor = None
        state.stream_buffers.fit_anchor = None
        state.stream_buffers.new_chunks.clear()
        state.stream_buffers.pending_body_chunks = []
        state.stream_buffers.pending_remesh = None
        state.stream_buffers.started = False
        state.stream_buffers.accum_joints = None
        state.stream_buffers.source_joints = None
        state.smplx_fit.orient = []
        state.smplx_fit.pose = []
        state.smplx_fit.transl = []
        state.smplx_fit.betas = None


def request_chunks(state, prompt: str, params: dict):
    return state.client.generate(
        prompt,
        steps=params["steps"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        cfg_scale=params["cfg_scale"],
        should_cancel=lambda: state.stream_buffers.cancel,
    )
