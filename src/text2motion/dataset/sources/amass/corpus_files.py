from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from tqdm import tqdm

from text2motion.common.manifest import write_manifest
from text2motion.dataset.core.operations import DATA_CLIP_ERRORS, NPY_FILES
from text2motion.motion.storage import atomic_numpy_save

logger = logging.getLogger(__name__)

CORPUS_MANIFEST = "corpus_manifest.json"


def source_fingerprint(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def stored_fingerprints(manifest_path: Path) -> dict[str, list[int]]:
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8")).get("source_fingerprints", {})


def check_unique_names(names: list[str]) -> None:
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(
            f"{len(repeated)} corpus names come from more than one pose file: {repeated[:3]}"
        )


def corpus_feature_name(pose_path: Path, pose_root: Path) -> str:
    relative = pose_path.resolve().relative_to(pose_root.resolve())
    return "__".join(relative.with_suffix("").parts) + ".npy"


def select_pose_sources(pose_root: Path, limit: int) -> list[Path]:
    sources = sorted(pose_root.rglob(NPY_FILES))
    if not sources:
        raise FileNotFoundError(f"no pose files under {pose_root}")
    if limit:
        sources = sources[:limit]
    return sources


def reused_frames(save_path: Path, src: Path, previous: object, current: object) -> int:
    if previous != current:
        raise ValueError(
            f"{save_path} exists, but its source {src} changed or is not recorded; "
            "build into a new folder"
        )
    return int(np.load(save_path, mmap_mode="r").shape[0])


def convert_source(
    src: Path, save_path: Path, features_of: Callable[[Path], np.ndarray | None]
) -> tuple[str, int]:
    try:
        features = features_of(src)
    except DATA_CLIP_ERRORS as exc:
        logger.warning(f"FAILED {save_path.name}: {exc}")
        return "failed", 0
    if features is None:
        return "skipped_short", 0
    atomic_numpy_save(features, save_path)
    return "written", features.shape[0]


def convert_sources(
    sources: list[Path],
    pose_root: Path,
    vec_dir: Path,
    excluded: set[str],
    previous: Mapping[str, object],
    fingerprints: Mapping[str, object],
    features_of: Callable[[Path], np.ndarray | None],
) -> tuple[dict[str, int], list[str]]:
    counts = {"written": 0, "reused": 0, "skipped_heldout": 0, "skipped_short": 0, "frames": 0}
    failed: list[str] = []
    for src in tqdm(sources, desc="pose->263 (all AMASS)"):
        name = corpus_feature_name(src, pose_root)
        if name in excluded:
            counts["skipped_heldout"] += 1
            continue
        save_path = vec_dir / name
        if save_path.is_file():
            outcome = "reused"
            frames = reused_frames(save_path, src, previous.get(name), fingerprints[name])
        else:
            outcome, frames = convert_source(src, save_path, features_of)
        if outcome == "failed":
            failed.append(name)
            continue
        counts[outcome] += 1
        counts["frames"] += frames
    return counts, failed


def require_no_failures(failed: list[str]) -> None:
    if failed:
        raise RuntimeError(
            f"{len(failed)} source files failed to convert (names in {CORPUS_MANIFEST}); "
            f"first: {failed[:3]}"
        )


def write_corpus_summary(
    manifest_path: Path,
    recipe: dict[str, object],
    counts: Mapping[str, int],
    failed: list[str],
    fps: int,
    limit: int,
    fingerprints: Mapping[str, object],
) -> dict[str, float]:
    summary = {
        **counts,
        "failed": len(failed),
        "hours": round(counts["frames"] / fps / 3600, 2),
    }
    status = {
        **summary,
        "limit": limit,
        "complete": not failed and not limit,
        "failed_names": failed,
        "source_fingerprints": fingerprints,
    }
    write_manifest(manifest_path, recipe, status)
    return summary
