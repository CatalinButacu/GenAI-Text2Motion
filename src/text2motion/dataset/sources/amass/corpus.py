from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from text2motion.common.manifest import check_recipe_reuse, write_manifest
from text2motion.dataset.core.operations import DATA_CLIP_ERRORS, NPY_FILES
from text2motion.dataset.sources.amass.poses import POSE_MANIFEST
from text2motion.dataset.sources.amass.reader import pose_candidates, pose_folders
from text2motion.dataset.sources.amass.trim import head_trim_frames, head_trim_table
from text2motion.dataset.sources.humanml3d.source import check_hml3d_fps, reference_joints_path
from text2motion.motion.contracts import PreparationRequest, Split
from text2motion.motion.representation import (
    encode_joints,
    feature_params,
    target_offsets,
)
from text2motion.motion.storage import (
    FEATURES_DIR,
    SPLIT_FILES,
    atomic_numpy_save,
    base_clip_ids,
    sha256_file,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def heldout_sources(index_csv: Path, out_dir: Path, pose_root: Path) -> set[Path]:
    def read_heldout_ids(out_dir: Path) -> set[str]:
        held: set[str] = set()
        for split in (Split.VALIDATION, Split.TEST):
            held.update(base_clip_ids(out_dir, split))
        return held

    import pandas as pd

    held_ids = read_heldout_ids(out_dir)
    index = pd.read_csv(index_csv)
    without_row = sorted(held_ids - {Path(str(name)).stem for name in index["new_name"]})
    if without_row:
        raise RuntimeError(
            f"leakage guard failed: {len(without_row)} held-out clips have no row in "
            f"{index_csv}, so their source capture cannot be excluded; first: {without_row[:5]}"
        )
    excluded: set[Path] = set()
    unexcludable: list[str] = []
    for row in index.to_dict("records"):
        if Path(str(row["new_name"])).stem not in held_ids:
            continue
        source_path = str(row["source_path"])
        folders = pose_folders(source_path, pose_root)
        candidates = pose_candidates(source_path, pose_root)
        matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
        if matches:
            excluded.update(matches)
            continue
        if any(folder.is_dir() for folder in folders):
            unexcludable.append(source_path)

    if unexcludable:
        raise RuntimeError(
            f"leakage guard failed: {len(unexcludable)} held-out clips have their AMASS capture "
            f"folder present under {pose_root} but no filename matched the known spellings, so "
            f"they would NOT be excluded from the unlabeled pretraining corpus. Fix the spelling "
            f"candidates before pretraining. First offenders: {unexcludable[:5]}"
        )
    return excluded


CORPUS_FORMAT = "text2motion.amass_corpus"


CORPUS_MANIFEST = "corpus_manifest.json"


CORPUS_VERSION = 2


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


def heldout_feature_names(index_csv: Path, out_dir: Path, pose_root: Path) -> set[str]:
    heldout = heldout_sources(index_csv, out_dir, pose_root)
    heldout_sizes = {path.stat().st_size for path in heldout}
    heldout_hashes = {sha256_file(path) for path in heldout}
    names = {corpus_feature_name(path, pose_root) for path in heldout}
    for path in pose_root.rglob(NPY_FILES):
        if path.resolve() in heldout:
            continue
        if path.stat().st_size in heldout_sizes and sha256_file(path) in heldout_hashes:
            names.add(corpus_feature_name(path, pose_root))
    return names


def complete_pose_recipe(pose_root: Path) -> dict[str, object]:
    manifest_path = pose_root / POSE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} missing; run stage 'amass' (with --adopt_pose_folder for a legacy folder)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"{manifest_path} says the pose extraction did not finish; run it again")
    return manifest["recipe"]


def corpus_recipe(
    request: PreparationRequest,
    index_csv: Path,
    reference_path: Path,
    pose_recipe: dict[str, object],
    min_frames: int,
) -> dict[str, object]:
    dataset_dir = Path(request.out_dir)
    recipe = {
        "format": CORPUS_FORMAT,
        "version": CORPUS_VERSION,
        "pose_recipe": pose_recipe,
        "min_frames": min_frames,
        "fps": request.fps,
        "joint_count": request.joint_count,
        "mirror_x": True,
        "head_trim_seconds": head_trim_table(),
        "reference_joints_sha256": sha256_file(reference_path),
        "index_sha256": sha256_file(index_csv),
        "val_split_sha256": sha256_file(dataset_dir / SPLIT_FILES[Split.VALIDATION]),
        "test_split_sha256": sha256_file(dataset_dir / SPLIT_FILES[Split.TEST]),
    }
    return json.loads(json.dumps(recipe))


def corpus_features(
    src: Path, pose_root: Path, request: PreparationRequest, params, min_frames: int
) -> np.ndarray | None:
    joints = np.load(src)[:, : request.joint_count]
    trim = head_trim_frames(src.relative_to(pose_root), request.fps)
    joints = joints[trim:].copy()
    if joints.shape[0] - 1 < min_frames:
        return None
    joints[..., 0] *= -1
    features, _, _, _ = encode_joints(joints, params)
    if not np.isfinite(features).all():
        raise ValueError("non-finite features")
    return features.astype(np.float32)


def require_corpus_inputs(request: PreparationRequest, regen_dir: Path) -> tuple[Path, Path]:
    pose_root = regen_dir / "pose_data"
    if not pose_root.is_dir():
        raise FileNotFoundError(
            f"{pose_root} missing -- run 'text2motion prepare --stage amass' first"
        )
    if request.index_csv is None:
        raise ValueError("index_csv must be set (the leakage guard needs the official index)")
    check_hml3d_fps(request.fps)
    return pose_root, Path(request.index_csv)


def corpus_params(reference_path: Path, request: PreparationRequest):
    joint_count = request.joint_count
    reference = np.load(reference_path)[:, :joint_count].reshape(-1, joint_count, 3)
    return feature_params(target_offsets(reference), request.fps)


def excluded_feature_names(
    index_csv: Path, dataset_dir: Path, pose_root: Path, vec_dir: Path, existing: set[str]
) -> set[str]:
    excluded = heldout_feature_names(index_csv, dataset_dir, pose_root)
    present = sorted(excluded & existing)
    if present:
        raise ValueError(
            f"{len(present)} heldout files are already in {vec_dir} (first: {present[:3]}); "
            "build into a new folder"
        )
    return excluded


def select_pose_sources(pose_root: Path, limit: int) -> list[Path]:
    sources = sorted(pose_root.rglob(NPY_FILES))
    if not sources:
        raise FileNotFoundError(f"no pose files under {pose_root}")
    if limit:
        sources = sources[:limit]
    return sources


def named_fingerprints(sources: list[Path], pose_root: Path) -> dict[str, list[int]]:
    names = [corpus_feature_name(src, pose_root) for src in sources]
    check_unique_names(names)
    return {name: source_fingerprint(src) for name, src in zip(names, sources, strict=True)}


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


def read_corpus_inputs(
    request: PreparationRequest, regen_dir: Path, min_frames: int
) -> tuple[Path, Path, dict[str, object], Callable[[Path], np.ndarray | None]]:
    pose_root, index_csv = require_corpus_inputs(request, regen_dir)
    reference_path = reference_joints_path(regen_dir / "joints", request.reference_clip)
    params = corpus_params(reference_path, request)
    pose_recipe = complete_pose_recipe(pose_root)
    recipe = corpus_recipe(request, index_csv, reference_path, pose_recipe, min_frames)
    features_of = partial(
        corpus_features, pose_root=pose_root, request=request, params=params, min_frames=min_frames
    )
    return pose_root, index_csv, recipe, features_of


def check_corpus_folder(
    out_dir: Path, recipe: dict[str, object], index_csv: Path, dataset_dir: Path, pose_root: Path
) -> set[str]:
    vec_dir = out_dir / FEATURES_DIR
    existing = {path.name for path in vec_dir.glob(NPY_FILES)}
    check_recipe_reuse(out_dir / CORPUS_MANIFEST, recipe, bool(existing))
    return excluded_feature_names(index_csv, dataset_dir, pose_root, vec_dir, existing)


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


def build_amass_pretraining_corpus(
    request: PreparationRequest,
    regen_dir: Path,
    out_dir: Path,
    min_frames: int,
    limit: int = 0,
) -> dict[str, float]:
    pose_root, index_csv, recipe, features_of = read_corpus_inputs(request, regen_dir, min_frames)
    dataset_dir = Path(request.out_dir)
    excluded = check_corpus_folder(out_dir, recipe, index_csv, dataset_dir, pose_root)
    vec_dir = out_dir / FEATURES_DIR
    manifest_path = out_dir / CORPUS_MANIFEST
    previous = stored_fingerprints(manifest_path)
    sources = select_pose_sources(pose_root, limit)
    fingerprints = named_fingerprints(sources, pose_root)
    vec_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, recipe, {"complete": False, "source_fingerprints": previous})
    counts, failed = convert_sources(
        sources, pose_root, vec_dir, excluded, previous, fingerprints, features_of
    )
    summary = write_corpus_summary(
        manifest_path, recipe, counts, failed, request.fps, limit, fingerprints
    )
    require_no_failures(failed)
    return summary
