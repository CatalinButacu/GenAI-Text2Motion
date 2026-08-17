from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ViewerHost(Protocol):
    scene: Any
    playback_fps: float
    run_animations: bool
    gui_controls: dict[str, Callable[[], None]]

    def toggle_animation(self, run: bool) -> None: ...


@runtime_checkable
class MotionClient(Protocol):
    hello: dict[str, Any]

    def generate(
        self,
        prompt: str,
        steps: int,
        temperature: float,
        top_p: float,
        cfg_scale: float,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[np.ndarray]: ...

    def load(
        self, config: str, ckpt: str, tokenizer_ckpt: str, backbone: str
    ) -> dict[str, Any]: ...
