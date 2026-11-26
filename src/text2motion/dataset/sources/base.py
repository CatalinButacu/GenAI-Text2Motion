from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

SUFFIX = re.compile(r"(_poses|_stageii)?\.(npy|npz)$")
PREFIXES = ("./", "pose_data/")
INDEX_MANIFEST = "index_manifest.json"
MIRROR_PREFIXES = ("", "M")
HML3D_FPS = 20


def recording_key(path: str) -> str:
    text = path.replace("\\", "/")
    for prefix in PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return SUFFIX.sub("", text)


class RecordingSource(ABC):
    name: str

    @abstractmethod
    def list_recordings(self, raw_dir: Path) -> list[Path]: ...

    @abstractmethod
    def split_of(self, clip: str) -> str | None: ...

    def captions_of(self, clip: str) -> tuple[str, ...]:
        return ()
