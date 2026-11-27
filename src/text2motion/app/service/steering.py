from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

############################################################
#       StreamSteering
############################################################
STEER_JUMP_FRAMES = 16


CM_PER_M = 100.0


@dataclass
class PendingSteer:
    prompt: str
    switch_token: int
    start: int
    end: int
    peak: float = 0.0
    measured: int = 0


@dataclass
class StreamSteering:
    every: int
    downsample: int
    text: torch.Tensor
    pending: list[PendingSteer] = field(default_factory=list)
    previous: np.ndarray | None = None
    offset: int = 0

    def text_at(self, index: int) -> torch.Tensor:
        return self.text

    def request(self, prompt: str, text: torch.Tensor, tokens_consumed: int) -> dict:
        if not self.every:
            return {
                "type": "steer_rejected",
                "reason": "the service runs without anchors; set anchor_every_tokens above 0",
            }
        self.text = text
        switch_token = -(-tokens_consumed // self.every) * self.every
        start = switch_token * self.downsample
        self.pending.append(
            PendingSteer(prompt, switch_token, start, start + STEER_JUMP_FRAMES - 1)
        )
        return {"type": "steer_accepted", "prompt": prompt, "switch_token": switch_token}

    def frame_steps(self, joints: np.ndarray) -> np.ndarray:
        steps = np.zeros(len(joints))
        steps[1:] = np.linalg.norm(joints[1:] - joints[:-1], axis=-1).max(axis=-1)
        if self.previous is not None:
            steps[0] = np.linalg.norm(joints[0] - self.previous, axis=-1).max()
        return steps

    def observe(self, joints: np.ndarray) -> list[dict]:
        steps = self.frame_steps(joints)
        frames = np.arange(self.offset, self.offset + len(joints))
        reports = []
        for item in list(self.pending):
            inside = (frames >= item.start) & (frames <= item.end)
            if inside.any():
                item.peak = max(item.peak, float(steps[inside].max()) * CM_PER_M)
                item.measured += int(inside.sum())
            if self.offset + len(joints) > item.end:
                reports.append(self.report(item))
                self.pending.remove(item)
        self.previous = joints[-1]
        self.offset += len(joints)
        return reports

    def flush(self) -> list[dict]:
        reports = [self.report(item) for item in self.pending]
        self.pending.clear()
        return reports

    def report(self, item: PendingSteer) -> dict:
        return {
            "type": "steered",
            "prompt": item.prompt,
            "switch_token": item.switch_token,
            "switch_frame": item.start,
            "frames_measured": item.measured,
            "joint_jump_cm": item.peak if item.measured else None,
        }
