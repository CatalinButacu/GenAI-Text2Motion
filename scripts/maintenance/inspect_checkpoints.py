#!/usr/bin/env python
"""Summarise motion checkpoints: epoch, val_loss, and key config, ranked best-first.

Scans the live checkpoints/ and archived _from_worktrees/ best_model.pt files so
competing checkpoints can be compared at a glance. Pass paths to target specific ones.

    python scripts/maintenance/inspect_checkpoints.py
    python scripts/maintenance/inspect_checkpoints.py checkpoints/motion_ssm/best_model.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
VAL_KEYS = ("val_loss", "best_loss", "best_val_loss")
CONFIG_FIELDS = (
    "arch", "d_model", "n_layers", "sbert_model",
    "learning_rate", "label_smoothing", "recon_loss_weight", "num_epochs",
)


def default_targets() -> list[Path]:
    live = [
        ROOT / "checkpoints/motion_ssm/best_model.pt",
        ROOT / "checkpoints/rvq_tokenizer/best_model.pt",
    ]
    archived = sorted((ROOT / "_from_worktrees").rglob("best_model.pt"))
    return [p for p in [*live, *archived] if p.exists()]


def resolve_targets(args: list[str]) -> list[Path]:
    if not args:
        return default_targets()
    targets: list[Path] = []

    for arg in args:
        path = Path(arg)

        if path.is_dir():
            targets.extend(sorted(path.rglob("best_model.pt")))
        elif path.exists():
            targets.append(path)

    return targets


def read_field(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def summarise(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt if isinstance(ckpt, dict) else {}
    config = meta.get("config")
    val = next((float(meta[k]) for k in VAL_KEYS if meta.get(k) is not None), None)
    label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

    return {
        "label": label,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "epoch": meta.get("epoch"),
        "val": val,
        "config": {name: read_field(config, name) for name in CONFIG_FIELDS},
    }


def main() -> int:
    targets = resolve_targets(sys.argv[1:])

    if not targets:
        print("No checkpoints found.")
        return 0

    rows = []

    for path in targets:
        try:
            rows.append(summarise(path))
        except (OSError, RuntimeError, EOFError) as err:
            print(f"SKIP {path}: {type(err).__name__}: {err}")

    rows.sort(key=lambda r: (r["val"] is None, r["val"] or 0.0))

    for row in rows:
        val = f"{row['val']:.4f}" if isinstance(row["val"], float) else "?"
        print(f"val={val}  epoch={row['epoch']}  {row['size_mb']}MB  {row['label']}")
        shown = "  ".join(f"{k}={v}" for k, v in row["config"].items() if v is not None)

        if shown:
            print(f"    {shown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
