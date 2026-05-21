from __future__ import annotations

import csv
import logging
from pathlib import Path

from .hml3d_helpers import normalize_amass_path

log = logging.getLogger(__name__)


def build_norm_map(local_files: list[Path], amass_data_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for f in local_files:
        rel = f.relative_to(amass_data_dir).as_posix()
        result[normalize_amass_path(rel)] = f

    return result


def parse_index_csv(idx_file: Path, split_ids: set, all_texts: dict,
                    norm_map: dict[str, Path]) -> list[dict]:
    samples: list[dict] = []

    with open(idx_file, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            clip_id = row["new_name"].replace(".npy", "")

            if clip_id not in split_ids:
                continue

            raw = row["source_path"].replace("./pose_data/", "")
            norm = normalize_amass_path(raw)

            if norm in norm_map and clip_id in all_texts:
                texts = all_texts[clip_id]
                samples.append({
                    "clip_id": clip_id,
                    "text": texts[0],          # canonical (used for vocab fallback)
                    "texts": list(texts),       # full list -- sampled per-epoch at __getitem__
                    "amass_path": str(norm_map[norm]),
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                })
    n_annotations = sum(len(s["texts"]) for s in samples)
    log.info("[HumanML3D] parsed %d SMPL-X mappings (%d total annotations, %.1fx supervision)",
             len(samples), n_annotations, n_annotations / max(len(samples), 1))

    return samples
