from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from text2motion.render.studio_viewer.constants import MODEL_REGISTRY


@dataclass(frozen=True)
class ModelEntry:
    label: str
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str


def load_model_registry(launch: ModelEntry) -> tuple[list[ModelEntry], int]:
    entries: list[ModelEntry] = []
    if MODEL_REGISTRY.is_file():
        raw = yaml.safe_load(MODEL_REGISTRY.read_text(encoding="utf-8"))
        for name, e in raw.items():
            entries.append(
                ModelEntry(
                    label=e.get("label", name),
                    config=e["config"],
                    ckpt=e["ckpt"],
                    tokenizer_ckpt=e["tokenizer_ckpt"],
                    backbone=e["backbone"],
                )
            )
    else:
        print(f"[warn] model registry not found: {MODEL_REGISTRY} -> launch args only")
    launch_ckpt = Path(launch.ckpt).resolve()
    for i, e in enumerate(entries):
        if Path(e.ckpt).resolve() == launch_ckpt:
            return entries, i
    entries.insert(0, launch)
    return entries, 0
