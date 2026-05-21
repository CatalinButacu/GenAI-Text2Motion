from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DIR = "data/humanml3d"
METADATA_RE = re.compile(r"#.*$")


def parse_annotation_line(line: str) -> str | None:
    line = line.strip()

    if not line:
        return None

    text = METADATA_RE.sub("", line).strip()

    return text if text else None


def normalize_amass_path(p: str) -> str:
    p = p.replace("\\", "/").lower()
    p = re.sub(r"_stageii|_stagei", "", p)
    p = re.sub(r"\.(npz|pkl|pt|npy)$", "", p)
    p = re.sub(r"_poses$", "", p)
    segs = p.split("/")

    return "/".join(re.sub(r"[\s_]+", "", s) for s in segs)


class HumanML3DLoader:
    def __init__(self, data_dir: str = DEFAULT_DIR) -> None:
        self.dir = Path(data_dir)
        self.texts_dir = self.dir / "texts"

    def is_available(self) -> bool:
        return self.texts_dir.exists() and any(self.texts_dir.glob("*.txt"))

    def load_texts(self) -> list[dict]:
        if not self.texts_dir.exists():
            return []

        results = []

        for path in sorted(self.texts_dir.glob("*.txt")):
            if path.stem.startswith("M"):  # skip mirrored augmentation clips
                continue

            anns = [t for ln in path.read_text(encoding="utf-8").splitlines()
                    if (t := parse_annotation_line(ln)) is not None]

            if anns:
                results.append({"clip_id": path.stem, "texts": anns})
        log.info("[HumanML3D] %d clip annotation sets", len(results))

        return results

    def load_split(self, split: str = "train") -> list[str]:
        path = self.dir / "split" / f"{split}.txt"

        if not path.exists():
            return []

        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def build_text_corpus(self) -> list[str]:
        return list(dict.fromkeys(t for item in self.load_texts() for t in item["texts"]))
