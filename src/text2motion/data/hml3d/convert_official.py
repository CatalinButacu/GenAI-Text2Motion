"""Convert the TeoGchx/HumanML3D parquet mirror into our regenerated-data layout.

TeoGchx/HumanML3D is the standard 263-dim HumanML3D (official splits: 23384 train / 4384 test / 1460
val, base + mirror). Each row carries ``caption``, ``motion`` (T, 263) and ``meta_data.name``. We
write ``new_joint_vecs/<name>.npy`` and the split lists, then compute ``Mean.npy``/``Std.npy`` from
the train split with the official ``cal_mean_variance`` per-group std smoothing. Texts are NOT taken
from here (the parquet lacks the POS ``tokens``/M-mirror captions) -- the dataset reads them from
``paths.texts_dir`` (the donor's official ``texts/``).

Run:
    python -m text2motion.data.hml3d.convert_official --src data/hml3d_official --out data/HumanML3D_official
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

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
    vec_dir = out_dir / "new_joint_vecs"
    clips = []
    for name in train_names:
        feature = np.load(vec_dir / f"{name}.npy")
        if np.isnan(feature).any():
            continue
        clips.append(feature)

    data = np.concatenate(clips, axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # Per-group std smoothing, copied EXACTLY from the official cal_mean_variance.ipynb.
    joints_num = JOINTS_NUM
    std[0:1] = std[0:1].mean() / 1.0
    std[1:3] = std[1:3].mean() / 1.0
    std[3:4] = std[3:4].mean() / 1.0
    std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3].mean() / 1.0
    rot_slice = slice(4 + (joints_num - 1) * 3, 4 + (joints_num - 1) * 9)
    std[rot_slice] = std[rot_slice].mean() / 1.0
    vel_slice = slice(4 + (joints_num - 1) * 9, 4 + (joints_num - 1) * 9 + joints_num * 3)
    std[vel_slice] = std[vel_slice].mean() / 1.0
    std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
        std[4 + (joints_num - 1) * 9 + joints_num * 3 :].mean() / 1.0
    )

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == std.shape[-1]

    np.save(out_dir / "Mean.npy", mean)
    np.save(out_dir / "Std.npy", std)
    print(f"Mean/Std saved from {len(clips)} train clips, feature shape {data.shape}")


def run(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split_names = write_motions(src, out)
    for split, names in split_names.items():
        print(f"{split}: {len(names)} clips")
    compute_mean_std(out, split_names["train"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TeoGchx/HumanML3D parquet to our layout.")
    parser.add_argument("--src", required=True, help="downloaded parquet repo dir")
    parser.add_argument("--out", required=True, help="output dir (new_joint_vecs, Mean/Std, splits)")
    args = parser.parse_args()
    run(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
