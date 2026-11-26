from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from text2motion.studio.session.buffers import StudioStreamBuffers


############################################################
#       StudioFitTrack
############################################################
class StudioFitTrack:
    def __init__(self, buffers: StudioStreamBuffers) -> None:
        self.buffers = buffers

    def append_fitted_chunk(self, fit, verts, faces, res) -> bool:
        with self.buffers.lock:
            if self.fit_is_stale():
                return False
            self.buffers.pending_body_chunks.append((verts, faces))
            fit.orient.append(res.global_orient)
            fit.pose.append(res.body_pose)
            fit.transl.append(res.transl)
            if fit.betas is None:
                fit.betas = res.betas
            return True

    def reset_fit_track(self, fit) -> None:
        with self.buffers.lock:
            self.buffers.pending_body_chunks = []
            self.buffers.fit_anchor = None
            fit.reset_track()

    def fit_is_stale(self) -> bool:
        return self.buffers.cancel or self.buffers.fit_take != self.buffers.take
