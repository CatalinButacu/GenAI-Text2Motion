from __future__ import annotations

from pathlib import Path

import numpy as np

from text2motion.dataset.sources.amass.extractor import AmassPoseExtractor
from text2motion.dataset.sources.base import RecordingSource

BODY_JOINTS = 22


class AmassSource(RecordingSource):
    name = "amass"

    def __init__(self, smplx_dir: Path | None = None, fps: int = 60, device: str = "cpu") -> None:
        self._smplx_dir = smplx_dir
        self._fps = fps
        self._device = device
        self._extractor = None

    def list_recordings(self, raw_dir: Path) -> list[Path]:
        return sorted(path for path in Path(raw_dir).rglob("*.npz"))

    def joints_of(self, recording: Path) -> np.ndarray | None:
        return self.get_extractor().extract(recording)

    def get_extractor(self):
        if self._extractor is None:
            if self._smplx_dir is None:
                raise ValueError("AmassSource needs smplx_dir to read recordings")
            self._extractor = AmassPoseExtractor(
                self._smplx_dir, joint_count=BODY_JOINTS, device=self._device, fps=self._fps
            )
        return self._extractor

    def recipe(self) -> dict:
        return self.get_extractor().recipe()

    def split_of(self, clip: str) -> str | None:
        return None
