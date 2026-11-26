from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
