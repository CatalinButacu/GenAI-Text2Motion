from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import ViewerCamera

from text2motion.studio.avatar import append_skeleton_frames, build_skeleton_seq
from text2motion.studio.config import StudioConfig
from text2motion.studio.contracts import MotionClient, ViewerHost


@dataclass(frozen=True)
class ModelEntry:
    label: str
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str


def load_model_registry(launch: ModelEntry, registry: Path) -> tuple[list[ModelEntry], int]:
    entries: list[ModelEntry] = []
    if registry.is_file():
        raw = yaml.safe_load(registry.read_text(encoding="utf-8"))
        for name, e in raw.items():
            entries.append(
                ModelEntry(
                    label=e.get("label", name),
                    config=e["config"],
                    ckpt=e["ckpt"],
                    tokenizer_ckpt=e["tokenizer_ckpt"],
                    backbone=e["backbone"],
                )
            )
    else:
        print(f"[warn] model registry not found: {registry} -> launch args only")
    launch_ckpt = Path(launch.ckpt).resolve()
    for i, e in enumerate(entries):
        if Path(e.ckpt).resolve() == launch_ckpt:
            return entries, i
    entries.insert(0, launch)
    return entries, 0


class SceneNodes:
    def __init__(self, scene) -> None:
        self.scene = scene
        self.rest = None
        self.motion = None
        self.body = None
        self.follow_center: np.ndarray | None = None

    def _drop(self, node):
        if node is None:
            return None
        if self.scene.selected_object is node:
            self.scene.select(None)
        self.scene.remove(node)
        return None

    def clear_rest(self) -> None:
        self.rest = self._drop(self.rest)

    def clear_motion(self) -> None:
        self.motion = self._drop(self.motion)

    def clear_body(self) -> None:
        self.body = self._drop(self.body)

    def clear_all(self) -> None:
        self.clear_motion()
        self.clear_rest()
        self.clear_body()

    def show_rest(self, node) -> None:
        self.clear_rest()
        self.rest = node
        self.scene.add(node)

    def show_motion(self, node) -> None:
        self.clear_motion()
        self.motion = node
        self.scene.add(node)

    def show_body(self, node) -> None:
        self.body = node
        self.scene.add(node)

    @property
    def focus(self):
        return self.body or self.motion or self.rest

    @property
    def has_motion_or_body(self) -> bool:
        return self.body is not None or self.motion is not None


class FitState:
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

    @property
    def n_frames(self) -> int:
        return len(self.orient)

    @property
    def is_stale(self) -> bool:
        return self.done_version != self.version

    def bump(self) -> int:
        self.version += 1
        return self.version


class StreamBuffers:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.new_chunks: list[np.ndarray] = []
        self.pending_body_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self.pending_remesh: tuple[np.ndarray, np.ndarray, bool] | None = None
        self.accum_joints: np.ndarray | None = None
        self.body_verts: np.ndarray | None = None
        self.current_fit_body = False
        self.drop_skeleton = False
        self.started = False
        self.need_reset_view = False
        self.cancel = False
        self.gen_progress = 0.0
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

    def append_fitted_chunk(self, fit, verts, faces, res) -> None:
        with self.lock:
            self.pending_body_chunks.append((verts, faces))
            fit.orient.append(res.global_orient)
            fit.pose.append(res.body_pose)
            fit.transl.append(res.transl)
            if fit.betas is None:
                fit.betas = res.betas

    def reset_fit_track(self, fit) -> None:
        with self.lock:
            self.pending_body_chunks = []
            fit.reset_track()

    def accumulate(self, joints: np.ndarray) -> np.ndarray:
        stacked = (
            joints if self.accum_joints is None else np.concatenate([self.accum_joints, joints])
        )
        self.accum_joints = stacked
        return stacked

    def reset_for_new_take(self) -> None:
        self.accum_joints = None
        self.body_verts = None
        self.started = False
        self.cancel = False
        self.gen_progress = 0.0
        self.fit_progress = None

    @property
    def body_frames(self) -> int:
        return 0 if self.body_verts is None else len(self.body_verts)


class LiveRegen:
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


class Display:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    def _consume(self) -> None:
        chunks, body_chunks, remesh = self.state.buf.drain()

        if chunks:
            self._append_chunks(chunks)
        if body_chunks:
            self._append_body_chunks(body_chunks)
        if self.state.buf.drop_skeleton and not body_chunks:
            self.state.buf.drop_skeleton = False
            if self.state.nodes.body is not None:
                self.state.nodes.clear_motion()
                self.state.nodes.body.enabled = True
        if remesh is not None:
            self._apply_remesh(*remesh)
        if self.state.buf.need_reset_view:
            self.state.buf.need_reset_view = False
            node = self.state.nodes.focus
            if node is not None:
                self._center_view(node)

    def _sync_display_nodes(self) -> None:
        if self.state.nodes.motion is None or self.state.nodes.body is None:
            return
        body_frames = 0 if self.state.buf.body_verts is None else len(self.state.buf.body_verts)
        on_body = self.host.scene.current_frame_id < body_frames
        self.state.nodes.body.enabled = on_body
        self.state.nodes.motion.enabled = not on_body

    def _center_view(self, node) -> None:
        bounds = node.current_bounds
        if not np.isfinite(bounds).all():
            print(f"[warn] skipped centring on {node.name}: non-finite bounds")
            return
        self.host.center_view_on_node(node)

    def _append_chunks(self, chunks: list[np.ndarray]) -> None:
        new = np.concatenate(chunks, axis=0)
        self.state.buf.accum_joints = (
            new
            if self.state.buf.accum_joints is None
            else np.concatenate([self.state.buf.accum_joints, new], axis=0)
        )
        if self.state.nodes.motion is None:
            motion = build_skeleton_seq(new)
            motion.name = "Motion (streaming)"
            self.state.nodes.show_motion(motion)
        else:
            append_skeleton_frames(self.state.nodes.motion, new)
        print(
            f"[DISP] skeleton node added: accum={self.state.buf.accum_joints.shape} "
            f"scene_frames={self.host.scene.n_frames} started={self.state.buf.started}",
            flush=True,
        )
        if not self.state.buf.started:
            self.state.buf.started = True
            self.state.nodes.clear_body()
            self.state.nodes.clear_rest()
            self.state.nodes.follow_center = None
            self.host.scene.current_frame_id = 0
            self.host.toggle_animation(True)
            self.state.buf.need_reset_view = True

    def _append_body_chunks(self, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for verts, faces in chunks:
            if self.state.nodes.body is None:
                self.state.nodes.show_body(
                    Meshes(verts, faces, color=self.state.skin_color, name="Avatar (SMPL-X)")
                )
                self.state.buf.body_verts = verts
            else:
                self.state.nodes.body.add_frames(verts)
                self.state.buf.body_verts = self.state.nodes.body.vertices

    def _apply_remesh(self, verts: np.ndarray, faces: np.ndarray, is_rest: bool) -> None:
        if is_rest:
            if self.state.nodes.has_motion_or_body:
                return
            self.state.nodes.show_rest(
                Meshes(verts, faces, color=self.state.skin_color, name="Avatar (rest)")
            )
            return
        frame = self.host.scene.current_frame_id
        self.state.nodes.clear_all()
        self.state.nodes.show_body(
            Meshes(verts, faces, color=self.state.skin_color, name="Avatar (SMPL-X)")
        )
        self.state.buf.body_verts = verts
        self.host.scene.current_frame_id = min(frame, len(verts) - 1)

    def _current_center(self) -> np.ndarray | None:
        frame = self.host.scene.current_frame_id
        body_frames = 0 if self.state.buf.body_verts is None else len(self.state.buf.body_verts)
        if self.state.nodes.body is not None and (
            self.state.nodes.motion is None or frame < body_frames
        ):
            arr = self.state.buf.body_verts
        elif self.state.nodes.motion is not None and self.state.buf.accum_joints is not None:
            arr = self.state.buf.accum_joints
        else:
            return None
        return arr[min(frame, len(arr) - 1)].mean(axis=0)

    def _update_follow_cam(self) -> None:
        if not self.state.follow_cam:
            return
        camera = self.host.scene.camera
        if not isinstance(camera, ViewerCamera):
            return
        center = self._current_center()
        if center is None:
            return
        if self.state.nodes.follow_center is None:
            offset = camera.position - camera.target
            camera.target = center
            camera.position = center + offset
        else:
            delta = center - self.state.nodes.follow_center
            camera.position = camera.position + delta
            camera.target = camera.target + delta
        self.state.nodes.follow_center = center


class StudioState:
    def __init__(
        self,
        client: MotionClient,
        model_dir: str,
        device: str,
        args: argparse.Namespace,
        scene,
        config: StudioConfig,
    ) -> None:
        self.client = client
        self.config = config
        self.device = device
        self.model_dir = model_dir
        self.downsample = int(client.hello["downsample"])
        self.max_steps = int(client.hello["max_steps"])

        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.cfg_scale = float(args.cfg_scale)
        self.steps = min(int(args.steps), self.max_steps)

        self.gender_idx = 0
        self.fit_body = bool(model_dir)
        self.skin_color = config.appearance.skin_color
        self.user_betas = np.zeros(config.fit.num_betas, dtype=np.float32)

        self.backbone = args.backbone
        launch_entry = ModelEntry(
            label=f"launch args: {Path(args.ckpt).stem}",
            config=args.config,
            ckpt=args.ckpt,
            tokenizer_ckpt=args.tokenizer_ckpt,
            backbone=args.backbone,
        )
        self.models, self.model_idx = load_model_registry(launch_entry, config.model_registry)
        self.loading_model = False
        self.load_status = f"active: {self.models[self.model_idx].label}"

        self.prompt_text = ""
        self.status = "ready -- describe a motion below and press Enter"
        self.generating = False
        self.history: list[str] = []

        self.live = LiveRegen()
        self.live.signature = (self.temperature, self.top_p, self.steps)
        self.buf = StreamBuffers()
        self.fit = FitState()
        self.nodes = SceneNodes(scene)
        self.follow_cam = True

        self.styled = False

    @property
    def gender(self) -> str:
        return self.config.genders[self.gender_idx]
