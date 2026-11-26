from __future__ import annotations

import logging
import queue

import numpy as np

logger = logging.getLogger(__name__)


def log_action(name: str, detail: str = "") -> None:
    logger.info(f"[UI] {name}{f' {detail}' if detail else ''}")


def log_toggle(name: str, value: bool) -> None:
    logger.info(f"[UI] {name} -> {'ON' if value else 'OFF'}")


def log_result(name: str, ok: bool, detail: str = "") -> None:
    logger.info(f"[UI] {name} -> {'OK' if ok else 'FAILED'}{f' ({detail})' if detail else ''}")


def upsample_joints(previous: np.ndarray | None, joints: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return joints
    start = joints[:1] if previous is None else previous[None]
    anchors = np.concatenate([start, joints], axis=0)
    weights = (np.arange(1, factor + 1, dtype=np.float32) / factor)[None, :, None, None]
    begin = anchors[:-1, None]
    end = anchors[1:, None]
    blended = begin + (end - begin) * weights
    return blended.reshape(len(joints) * factor, *joints.shape[1:]).astype(joints.dtype)


def discard_pending_items(pending: queue.Queue) -> None:
    try:
        while True:
            pending.get_nowait()
    except queue.Empty:
        pass
