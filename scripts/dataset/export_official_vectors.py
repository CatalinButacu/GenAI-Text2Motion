# .\.venv\Scripts\python.exe scripts/dataset/export_official_vectors.py
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO / "data" / "raw" / "hml3d" / "data"
OFFICIAL = REPO / "data" / "HumanML3D_official"
OUT = OFFICIAL / "new_joint_vecs"
FEATURE_DIM = 263


def index_names() -> set[str]:
    index = pd.read_csv(OFFICIAL / "index.csv")
    return {str(name).removesuffix(".npy") for name in index["new_name"]}


def export_rows(wanted: set[str]) -> tuple[int, set[str]]:
    written = 0
    found: set[str] = set()
    OUT.mkdir(parents=True, exist_ok=True)
    for shard in sorted(PARQUET_DIR.glob("*.parquet")):
        for row in pq.read_table(shard, columns=["motion", "meta_data"]).to_pylist():
            name = str(row["meta_data"]["name"])
            if name not in wanted:
                continue
            vectors = np.asarray(row["motion"], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != FEATURE_DIM:
                raise ValueError(f"{name}: expected (T, {FEATURE_DIM}), got {vectors.shape}")
            if not np.isfinite(vectors).all():
                raise ValueError(f"{name}: non-finite values")
            np.save(OUT / f"{name}.npy", vectors)
            found.add(name)
            written += 1
    return written, wanted - found


def main() -> None:
    wanted = index_names()
    written, missing = export_rows(wanted)
    print("written", written, "missing", len(missing), sorted(missing))
    if written == 0:
        sys.exit("no clips written")


if __name__ == "__main__":
    main()
