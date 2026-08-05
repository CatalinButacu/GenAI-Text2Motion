from __future__ import annotations

import threading

import numpy as np


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
        self.scene.remove(node)  # aitviewer's Scene.remove already tolerates a missing node
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
        self.lock = threading.Lock()  # guards these buffers AND the paired FitState track
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
        with self.lock:  # body chunk and fit frame must stay index-aligned for the retrofit
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
        self.prev_prompt = ""  # doubles as the gui change-detector and the pending-edit flag
        self.dirty_at = 0.0
        self.last_dispatched = ""
        self.regen_pending = False
        self.signature: tuple | None = None

    def mark_dirty(self, text: str, now: float) -> None:
        self.prev_prompt = text
        self.dirty_at = now

    def debounce_elapsed(self, now: float, debounce_s: float) -> bool:
        return self.prev_prompt != "" and (now - self.dirty_at) >= debounce_s
