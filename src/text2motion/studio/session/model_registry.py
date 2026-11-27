from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from text2motion.studio.contracts import (
    StudioLaunchArguments,
)

logger = logging.getLogger(__name__)


############################################################
#       StudioModelEntry
############################################################
@dataclass(frozen=True)
class StudioModelEntry:
    name: str
    version: str
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}"


def live_step_budget(steps: int, cap: int) -> int:
    if cap == 0:
        return steps
    if steps == 0:
        return cap
    return min(steps, cap)


def resolve_step_budget(args: StudioLaunchArguments, downsample: int, max_steps: int) -> int:
    fps = max(1, int(args.fps))
    requested = int(args.steps)
    if float(args.seconds) > 0.0:
        requested = max(1, math.ceil(float(args.seconds) * fps / downsample))
    if requested == 0:
        if max_steps == 0:
            return 0
        raise SystemExit(
            f"indefinite generation needs a recurrent backbone, but this model caps at "
            f"{max_steps} steps (~{max_steps * downsample / fps:.1f}s). "
            f"Pass --seconds N (or --steps N) to set a finite budget, or run a mamba model."
        )
    if max_steps and requested > max_steps:
        raise SystemExit(
            f"requested {requested} steps but this model supports at most {max_steps} "
            f"(~{max_steps * downsample / fps:.1f}s)"
        )
    return requested


def load_model_registry(
    launch: StudioModelEntry | None, registry: Path
) -> tuple[list[StudioModelEntry], int]:
    entries: list[StudioModelEntry] = []
    if registry.is_file():
        raw = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
        defaults = raw.get("defaults", {})
        for entry in raw.get("models", []):
            values = defaults | entry
            entries.append(
                StudioModelEntry(
                    name=values["name"],
                    version=str(values["version"]),
                    config=values["config"],
                    ckpt=values["ckpt"],
                    tokenizer_ckpt=values["tokenizer_ckpt"],
                    backbone=values.get("backbone", values["name"]),
                )
            )
    else:
        logger.info(f"[warn] model registry not found: {registry} -> launch args only")
    if launch is None:
        return entries, 0
    launch_ckpt = Path(launch.ckpt).resolve()
    for i, e in enumerate(entries):
        if Path(e.ckpt).resolve() == launch_ckpt:
            return entries, i
    entries.insert(0, launch)
    return entries, 0
