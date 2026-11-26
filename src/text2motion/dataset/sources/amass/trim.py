from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from text2motion.dataset.sources.amass.names import dataset_name

_HEAD_TRIM_SECONDS = MappingProxyType(
    {
        "Eyes_Japan_Dataset": 3.0,
        "MPI_HDM05": 3.0,
        "TotalCapture": 1.0,
        "MPI_Limits": 1.0,
        "Transitions_mocap": 0.5,
    }
)


def head_trim_table() -> dict[str, float]:
    return dict(sorted(_HEAD_TRIM_SECONDS.items()))


def head_trim_frames(source_path: str | Path, fps: int) -> int:
    return int(round(_HEAD_TRIM_SECONDS.get(dataset_name(source_path), 0.0) * fps))
