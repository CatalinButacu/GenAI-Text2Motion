from __future__ import annotations

import csv
import logging
from pathlib import Path

from .hml3d_helpers import normalizeAmassPath

log = logging.getLogger(__name__)


def buildNormMap(localFiles: list[Path], amassDataDir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for f in localFiles:
        rel = f.relative_to(amassDataDir).as_posix()
        result[normalizeAmassPath(rel)] = f

    return result


def parseIndexCsv(idxFile: Path, splitIds: set, allTexts: dict,
                    normMap: dict[str, Path]) -> list[dict]:
    samples: list[dict] = []

    with open(idxFile, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            clipId = row["new_name"].replace(".npy", "")

            if clipId not in splitIds:
                continue

            raw = row["source_path"].replace("./pose_data/", "")
            norm = normalizeAmassPath(raw)

            if norm in normMap and clipId in allTexts:
                texts = allTexts[clipId]
                samples.append({
                    "clip_id": clipId,
                    "text": texts[0],          # canonical (used for vocab fallback)
                    "texts": list(texts),       # full list -- sampled per-epoch at __getitem__
                    "amass_path": str(normMap[norm]),
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                })
    nAnnotations = sum(len(s["texts"]) for s in samples)
    log.info("[HumanML3D] parsed %d SMPL-X mappings (%d total annotations, %.1fx supervision)",
             len(samples), nAnnotations, nAnnotations / max(len(samples), 1))

    return samples
