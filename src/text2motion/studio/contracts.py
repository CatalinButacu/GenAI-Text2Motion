from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np

from text2motion.motion.model import StreamedChunk


@runtime_checkable
class ViewerHost(Protocol):
    scene: Any
    playback_fps: float
    run_animations: bool
    gui_controls: dict[str, Callable[[], None]]
    viewports: Any
    viewport_mode: Any
    shadows_enabled: Any
    _past_frametimes: np.ndarray

    def toggle_animation(self, run: bool) -> None: ...

    def reload_settings(self) -> None: ...


@runtime_checkable
class MotionInferenceClientPort(Protocol):
    hello: dict[str, Any]
    last_stats: dict[str, Any] | None

    def generate(
        self,
        prompt: str,
        steps: int,
        temperature: float,
        top_p: float,
        cfg_scale: float,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[StreamedChunk]: ...

    def load(
        self, config: str, ckpt: str, tokenizer_ckpt: str, backbone: str
    ) -> dict[str, Any]: ...
