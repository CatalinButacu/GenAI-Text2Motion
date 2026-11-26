# python scripts/maintenance/build_clean_train_split.py
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from text2motion.dataset.contracts import QualityLimits
from text2motion.dataset.core.quality import fit_quality_forest, quality_inliers, quality_metrics

SOURCE = Path("data/processed/humanml3d")
TARGET = Path("data/processed/humanml3d_clean")
FEATURES = "new_joint_vecs"
FRAME_RATE = 20
REPORT = Path("docs/project/clean_train_split.csv")
COPIED = ("Mean.npy", "Std.npy", "val.txt", "test.txt")


def base_name(name: str) -> str:
    return name[1:] if name.startswith("M") else name


def train_names() -> list[str]:
    return [
        line.strip()
        for line in (SOURCE / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metric_rows(names: list[str], limits: QualityLimits) -> np.ndarray:
    rows = []
    for name in names:
        feature = np.load(SOURCE / FEATURES / f"{name}.npy").astype(np.float64)
        rows.append(quality_metrics(feature, FRAME_RATE, limits))
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    limits = QualityLimits()
    names = [name for name in train_names() if not name.startswith("M")]
    metrics = metric_rows(names, limits)
    finite = np.isfinite(metrics).all(axis=1)
    forest = fit_quality_forest(metrics[finite], limits)
    inlier = quality_inliers(forest, metrics)

    kept = {name for name, keep in zip(names, inlier, strict=True) if keep}
    all_train = train_names()
    clean_train = [name for name in all_train if base_name(name) in kept]

    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "train.txt").write_text("\n".join(clean_train) + "\n", encoding="utf-8")
    for filename in COPIED:
        shutil.copyfile(SOURCE / filename, TARGET / filename)
    link = TARGET / FEATURES
    if not link.exists():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link.resolve()), str((SOURCE / FEATURES).resolve())],
            check=True,
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["measure", "value"])
        writer.writerow(["train base clips", len(names)])
        writer.writerow(["flagged outliers", int((~inlier).sum())])
        writer.writerow(["kept base clips", len(kept)])
        writer.writerow(["clean train lines (with mirror)", len(clean_train)])
        for label, column in (("skating", 0), ("floating_mm", 1), ("jerk", 2), ("yaw", 3)):
            writer.writerow([f"median {label} all", f"{np.nanmedian(metrics[:, column]):.4f}"])
            writer.writerow(
                [f"median {label} kept", f"{np.nanmedian(metrics[inlier, column]):.4f}"]
            )
    print(
        json.dumps(
            {
                "train_all": len(names),
                "flagged": int((~inlier).sum()),
                "clean_lines": len(clean_train),
            }
        )
    )


if __name__ == "__main__":
    main()
