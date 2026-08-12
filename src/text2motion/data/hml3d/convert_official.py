from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from .stats import fit_train_stats

FEATURE_DIM = 263
JOINTS_NUM = 22
SPLITS = ("train", "val", "test")


def _parquet_files(src: Path, split: str) -> list[Path]:
    matches = sorted(src.glob(f"**/{split}-*.parquet"))
    if not matches:
        matches = sorted(p for p in src.glob("**/*.parquet") if split in p.name)
    return matches


def write_motions(src: Path, out_dir: Path) -> dict[str, list[str]]:
    vec_dir = out_dir / "new_joint_vecs"
    vec_dir.mkdir(parents=True, exist_ok=True)
    split_names: dict[str, list[str]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        files = _parquet_files(src, split)
        if not files:
            raise FileNotFoundError(f"no parquet files for split {split!r} under {src}")

        for parquet_file in files:
            reader = pq.ParquetFile(parquet_file)
            desc = f"{split}:{parquet_file.name}"
            for batch in tqdm(reader.iter_batches(batch_size=256), desc=desc):
                rows = batch.to_pydict()
                for motion, meta in zip(rows["motion"], rows["meta_data"], strict=True):
                    name = meta["name"]
                    feature = np.asarray(motion, dtype=np.float32)
                    if feature.shape[-1] != FEATURE_DIM:
                        raise ValueError(f"{name}: expected {FEATURE_DIM}-dim, got {feature.shape}")
                    np.save(vec_dir / f"{name}.npy", feature)
                    split_names[split].append(name)

    for split, names in split_names.items():
        (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    return split_names


def compute_mean_std(out_dir: Path, train_names: list[str]) -> None:
    fit_train_stats(out_dir / "new_joint_vecs", train_names, out_dir, JOINTS_NUM)


def run(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split_names = write_motions(src, out)
    for split, names in split_names.items():
        print(f"{split}: {len(names)} clips")
    compute_mean_std(out, split_names["train"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TeoGchx/HumanML3D parquet to our layout.")
    parser.add_argument("--src", required=True, help="downloaded parquet repo dir")
    parser.add_argument(
        "--out", required=True, help="output dir (new_joint_vecs, Mean/Std, splits)"
    )
    args = parser.parse_args()
    run(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
