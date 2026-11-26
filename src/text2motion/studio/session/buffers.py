from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


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
