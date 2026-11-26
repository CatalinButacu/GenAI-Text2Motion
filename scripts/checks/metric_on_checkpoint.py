# python scripts/checks/metric_on_checkpoint.py --checkpoint runs/checkpoints/tokenizer/eq_base_2ep_fsq_g8
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_check import (
    CLIP_COUNT,
    FEATURE_DIR,
    JOINT_COUNT,
    absolute_mpjpe_mm,
    project_recover,
    root_aligned_mpjpe_mm,
    validation_names,
)

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.training import with_tokenizer_kind
from text2motion.motion.normalization import MotionScaler
from text2motion.tokenization.contracts import TokenizerKind

CONFIG = Path("configs/tokenizer/fsq_g8_v1024.yaml")
DATA_ROOT = Path("data/processed/humanml3d")
RESULT_FILE = Path("docs/project/metric_on_checkpoint.csv")
FEATURE_DIM = 263
DOWNSAMPLE = 4


def load_module(checkpoint: Path):
    app = ApplicationBootstrap.from_config(CONFIG, device="cpu", seed=2026)
    app = with_tokenizer_kind(app, TokenizerKind.FINITE_SCALAR)
    module = app.create_tokenizer_network()
    module.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    module.eval()
    return module


def reconstruct(module, scaler: MotionScaler, feature: np.ndarray) -> np.ndarray:
    normalized = torch.from_numpy(np.asarray(scaler.normalize(feature), np.float32)).unsqueeze(0)
    with torch.no_grad():
        reconstruction = module(normalized)[0][0].numpy()
    return np.asarray(scaler.denormalize(reconstruction), np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.checkpoint)
    scaler = MotionScaler.load(DATA_ROOT, dim=FEATURE_DIM)
    rows = []
    for name in validation_names():
        feature = np.load(FEATURE_DIR / f"{name}.npy").astype(np.float64)
        feature = feature[: (feature.shape[0] // DOWNSAMPLE) * DOWNSAMPLE]
        reference = project_recover(feature, JOINT_COUNT)
        reconstruction = project_recover(reconstruct(module, scaler, feature), JOINT_COUNT)
        rows.append(
            {
                "clip": name,
                "frames": feature.shape[0],
                "mpjpe_absolute_mm": f"{absolute_mpjpe_mm(reference, reconstruction):.2f}",
                "mpjpe_root_aligned_mm": f"{root_aligned_mpjpe_mm(reference, reconstruction):.2f}",
            }
        )

    absolute = np.mean([float(row["mpjpe_absolute_mm"]) for row in rows])
    aligned = np.mean([float(row["mpjpe_root_aligned_mm"]) for row in rows])
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"checkpoint {args.checkpoint.name}  clips {CLIP_COUNT}  "
        f"MPJPE absolute {absolute:.1f} mm  root-aligned {aligned:.1f} mm  -> {RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
