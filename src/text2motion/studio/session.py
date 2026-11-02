from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from text2motion.studio.config import StudioConfig
from text2motion.studio.contracts import (
    MotionInferenceClientPort,
    StudioLaunchArguments,
)
from text2motion.studio.scene import StudioSceneNodes

logger = logging.getLogger(__name__)


############################################################
#       StudioModelEntry
############################################################
@dataclass(frozen=True)
class StudioModelEntry:
    name: str
    version: str
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}"


def live_step_budget(steps: int, cap: int) -> int:
    if cap == 0:
        return steps
    if steps == 0:
        return cap
    return min(steps, cap)


def default_step_budget(trained_steps: int, max_steps: int) -> int:
    return min(trained_steps, max_steps) if max_steps else trained_steps


def resolve_step_budget(args: StudioLaunchArguments, downsample: int, max_steps: int) -> int:
    fps = max(1, int(args.fps))
    requested = int(args.steps)
    if float(args.seconds) > 0.0:
        requested = max(1, math.ceil(float(args.seconds) * fps / downsample))
    if requested == 0:
        if max_steps == 0:
            return 0
        raise SystemExit(
            f"indefinite generation needs a recurrent backbone, but this model caps at "
            f"{max_steps} steps (~{max_steps * downsample / fps:.1f}s). "
            f"Pass --seconds N (or --steps N) to set a finite budget, or run a mamba model."
        )
    if max_steps and requested > max_steps:
        raise SystemExit(
            f"requested {requested} steps but this model supports at most {max_steps} "
            f"(~{max_steps * downsample / fps:.1f}s)"
        )
    return requested


def load_model_registry(
    launch: StudioModelEntry | None, registry: Path
) -> tuple[list[StudioModelEntry], int]:
    entries: list[StudioModelEntry] = []
    if registry.is_file():
        raw = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
        defaults = raw.get("defaults", {})
        for entry in raw.get("models", []):
            values = defaults | entry
            entries.append(
                StudioModelEntry(
                    name=values["name"],
                    version=str(values["version"]),
                    config=values["config"],
                    ckpt=values["ckpt"],
                    tokenizer_ckpt=values["tokenizer_ckpt"],
                    backbone=values.get("backbone", values["name"]),
                )
            )
    else:
        logger.info(f"[warn] model registry not found: {registry} -> launch args only")
    if launch is None:
        return entries, 0
    launch_ckpt = Path(launch.ckpt).resolve()
    for i, e in enumerate(entries):
        if Path(e.ckpt).resolve() == launch_ckpt:
            return entries, i
    entries.insert(0, launch)
    return entries, 0


############################################################
#       SmplxFitState
############################################################
class SmplxFitState:
    def __init__(self) -> None:
        self.model_cache: dict[tuple[int, str], object] = {}
        self.orient: list[np.ndarray] = []
        self.pose: list[np.ndarray] = []
        self.transl: list[np.ndarray] = []
        self.betas: np.ndarray | None = None
        self.version = 0
        self.done_version = 0
        self.running = False
        self.after_gen = False
        self.note = ""

    def reset_track(self) -> None:
        self.orient.clear()
        self.pose.clear()
        self.transl.clear()
        self.betas = None

    def drop_first_frames(self, count: int) -> None:
        if count <= 0 or not self.orient:
            return
        self.orient = [np.concatenate(self.orient)[count:]]
        self.pose = [np.concatenate(self.pose)[count:]]
        self.transl = [np.concatenate(self.transl)[count:]]


############################################################
#       StudioStreamBuffers
############################################################
class StudioStreamBuffers:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.new_chunks: list[np.ndarray] = []
        self.pending_body_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self.pending_remesh: tuple[np.ndarray, np.ndarray, bool] | None = None
        self.accum_joints: np.ndarray | None = None
        self.source_joints: np.ndarray | None = None
        self.body_verts: np.ndarray | None = None
        self.current_fit_body = False
        self.take = 0
        self.trimmed_frames = 0
        self.interpolation_anchor: np.ndarray | None = None
        self.fit_anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.fit_take = -1
        self.fitting = False
        self.started = False
        self.need_reset_view = False
        self.cancel = False
        self.gen_progress = 0.0
        self.frames_emitted = 0
        self.last_body_lag = -1
        self.fit_progress: float | None = None

    def drain(self) -> tuple[list, list, tuple | None]:
        with self.lock:
            chunks, self.new_chunks = self.new_chunks, []
            body, self.pending_body_chunks = self.pending_body_chunks, []
            remesh, self.pending_remesh = self.pending_remesh, None
        return chunks, body, remesh

    def push_joints(self, joints: np.ndarray) -> None:
        with self.lock:
            self.new_chunks.append(joints)

    def push_body(self, verts: np.ndarray, faces: np.ndarray) -> None:
        with self.lock:
            self.pending_body_chunks.append((verts, faces))

    def push_remesh(self, verts: np.ndarray, faces: np.ndarray, is_rest: bool) -> None:
        with self.lock:
            self.pending_remesh = (verts, faces, is_rest)

    def append_fitted_chunk(self, fit, verts, faces, res) -> bool:
        with self.lock:
            if self.fit_is_stale():
                return False
            self.pending_body_chunks.append((verts, faces))
            fit.orient.append(res.global_orient)
            fit.pose.append(res.body_pose)
            fit.transl.append(res.transl)
            if fit.betas is None:
                fit.betas = res.betas
            return True

    def reset_fit_track(self, fit) -> None:
        with self.lock:
            self.pending_body_chunks = []
            self.fit_anchor = None
            fit.reset_track()

    def add_source_joints(self, joints: np.ndarray) -> None:
        with self.lock:
            previous = self.source_joints
            self.source_joints = joints if previous is None else np.concatenate([previous, joints])

    def keep_last_source_frames(self, count: int) -> None:
        with self.lock:
            if self.source_joints is not None:
                self.source_joints = self.source_joints[-count:]

    def reset_for_new_take(self) -> None:
        self.accum_joints = None
        self.source_joints = None
        self.body_verts = None
        self.started = False
        self.cancel = False
        self.gen_progress = 0.0
        self.fit_progress = None

    @property
    def body_frames(self) -> int:
        return 0 if self.body_verts is None else len(self.body_verts)

    def fit_is_stale(self) -> bool:
        return self.cancel or self.fit_take != self.take


############################################################
#       LiveGenerationDebouncer
############################################################
class LiveGenerationDebouncer:
    def __init__(self) -> None:
        self.enabled = False
        self.prev_prompt = ""
        self.dirty_at = 0.0
        self.last_dispatched = ""
        self.regen_pending = False
        self.signature: tuple | None = None

    def mark_dirty(self, text: str, now: float) -> None:
        self.prev_prompt = text
        self.dirty_at = now

    def debounce_elapsed(self, now: float, debounce_s: float) -> bool:
        return self.prev_prompt != "" and (now - self.dirty_at) >= debounce_s


############################################################
#       StudioSession
############################################################
class StudioSession:
    def __init__(
        self,
        client: MotionInferenceClientPort,
        model_dir: str,
        device: str,
        args: StudioLaunchArguments,
        scene,
        config: StudioConfig,
    ) -> None:
        self.client = client
        self.config = config
        self.device = device
        self.model_dir = model_dir
        self.apply_model_limits(client.hello)

        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.cfg_scale = float(args.cfg_scale)
        self.steps = resolve_step_budget(args, self.downsample, self.max_steps)
        self.window_frames = int(config.display.window_frames)
        self.fit_lookahead_chunks = int(config.display.fit_lookahead_chunks)
        self.stream_lead_frames = int(config.display.stream_lead_frames)
        self.upsample = int(config.display.upsample)
        if self.upsample < 1:
            raise ValueError(f"display.upsample must be >= 1, got {self.upsample}")
        self.window_steps = max(8, self.window_frames // self.downsample)

        self.gender_idx = 0
        self.fit_body = bool(model_dir)
        self.show_skin = True
        self.skin_color = config.appearance.skin_color
        self.user_betas = np.zeros(config.fit.num_betas, dtype=np.float32)

        self.backbone = args.backbone
        launch_entry = StudioModelEntry(
            name=Path(str(args.ckpt)).stem,
            version="local",
            config=str(args.config),
            ckpt=str(args.ckpt),
            tokenizer_ckpt=str(args.tokenizer_ckpt),
            backbone=str(args.backbone),
        )
        self.launch_entry = launch_entry
        self.models, self.model_idx = load_model_registry(launch_entry, config.model_registry)
        self.loading_model = False
        self.load_status = f"active: {self.models[self.model_idx].label}"

        self.prompt_text = ""
        self.status = "ready -- describe a motion below and press Enter"
        self.generating = False
        self.history: list[str] = []

        self.live = LiveGenerationDebouncer()
        self.live.signature = (self.temperature, self.top_p, self.steps)
        self.stream_buffers = StudioStreamBuffers()
        self.smplx_fit = SmplxFitState()
        self.nodes = StudioSceneNodes(scene)
        self.follow_cam = True

        self.styled = False

    def apply_model_limits(self, hello: dict) -> None:
        self.downsample = int(hello["downsample"])
        self.trained_steps = int(hello["trained_steps"])
        self.max_steps = int(hello["max_steps"])
        self.default_steps = default_step_budget(self.trained_steps, self.max_steps)

    def apply_config(self, config: StudioConfig) -> str:
        previous = len(self.models)
        self.config = config
        self.window_frames = int(config.display.window_frames)
        self.window_steps = max(8, self.window_frames // self.downsample)
        self.fit_lookahead_chunks = int(config.display.fit_lookahead_chunks)
        self.stream_lead_frames = int(config.display.stream_lead_frames)
        self.skin_color = config.appearance.skin_color
        self.models, self.model_idx = load_model_registry(
            self.models[self.model_idx], config.model_registry
        )
        added = len(self.models) - previous
        return (
            f"settings reloaded -- "
            f"window {self.window_frames} frames, {len(self.models)} models"
            + (f" (+{added})" if added > 0 else "")
        )

    @property
    def gender(self) -> str:
        return self.config.genders[self.gender_idx]
