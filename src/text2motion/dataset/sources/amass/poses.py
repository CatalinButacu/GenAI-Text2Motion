from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from text2motion.common.manifest import check_recipe_reuse, write_manifest
from text2motion.dataset.contracts import AMASS_LABEL_FPS
from text2motion.dataset.sources.amass.reader import AmassReader, resolve_pose_path
from text2motion.dataset.sources.amass.source import AmassSource
from text2motion.motion.storage import atomic_numpy_save

ADOPT_TOLERANCE = 1e-4


def pose_path_for(src: Path, amass_root: Path, pose_root: Path) -> Path:
    return (pose_root / src.relative_to(amass_root)).with_suffix(".npy")


def same_pose(extracted: np.ndarray | None, saved: np.ndarray) -> bool:
    if extracted is None or extracted.shape != saved.shape:
        return False
    return bool(np.allclose(extracted, saved, rtol=0.0, atol=ADOPT_TOLERANCE))


POSE_MANIFEST = "pose_manifest.json"


POSE_FORMAT = "text2motion.amass_poses"


POSE_VERSION = 2


CLOCKED_POSES = "pose_data_clocked"


ADOPT_SAMPLE = 20


def adopt_pose_folder(
    extractor: AmassSource,
    amass_root: Path,
    pose_root: Path,
    npz_files: list[Path],
    recipe: dict[str, object],
) -> None:
    manifest_path = pose_root / POSE_MANIFEST
    if manifest_path.is_file():
        raise ValueError(f"{manifest_path} exists; adoption is only for a folder without one")
    pairs = [(src, pose_path_for(src, amass_root, pose_root)) for src in npz_files]
    pairs = [(src, saved) for src, saved in pairs if saved.is_file()]
    if not pairs:
        raise FileNotFoundError(f"{pose_root} has no pose files to adopt")
    rng = np.random.default_rng(2026)
    picks = sorted(rng.choice(len(pairs), min(ADOPT_SAMPLE, len(pairs)), replace=False))
    sample = [pairs[index] for index in picks]
    mismatched = [
        saved.name
        for src, saved in sample
        if not same_pose(extractor.joints_of(src), np.load(saved))
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} of {len(sample)} sampled poses differ from a new extraction "
            f"(first: {mismatched[:3]}); re-extract into a new folder"
        )
    adopted = [str(saved.relative_to(pose_root)) for _, saved in sample]
    write_manifest(manifest_path, recipe, {"complete": False, "adopted_sample": adopted})


def write_clocked_poses(
    amass_raw: Path,
    smplx_models: Path,
    pose_data: Path,
    out_dir: Path,
    rates: list[tuple[str, int]],
    fps: int,
) -> dict[int, int]:
    reader = AmassReader(smplx_models, fps)
    counts: dict[int, int] = {}
    for source_path, stored_fps in tqdm(rates, desc="clocked poses"):
        pose_path = resolve_pose_path(source_path, pose_data)
        if pose_path is None:
            raise FileNotFoundError(f"clock table names {source_path} but no pose file")
        relative = pose_path.relative_to(pose_data)
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        counts[stored_fps] = counts.get(stored_fps, 0) + 1
        if target.is_file():
            continue
        if stored_fps == AMASS_LABEL_FPS:
            os.link(pose_path, target)
            continue
        joints = reader.raw_joints(amass_raw / relative.with_suffix(".npz"), stored_fps)
        if joints is None:
            raise ValueError(f"{relative} has no motion at the stored rate")
        np.save(target, joints)
    return counts


def open_pose_manifest(
    extractor: AmassSource,
    amass_root: Path,
    pose_root: Path,
    npz_files: list[Path],
    recipe: dict[str, object],
    adopt: bool,
) -> dict[str, object]:
    if adopt:
        adopt_pose_folder(extractor, amass_root, pose_root, npz_files, recipe)
    manifest_path = pose_root / POSE_MANIFEST
    check_recipe_reuse(manifest_path, recipe, any(pose_root.rglob("*.npy")))
    previous = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {key: value for key, value in previous.items() if key == "adopted_sample"}
    write_manifest(manifest_path, recipe, {**manifest, "complete": False})
    return manifest


def write_missing_poses(
    extractor: AmassSource,
    amass_root: Path,
    pose_root: Path,
    npz_files: list[Path],
) -> dict[str, int]:
    counts = {"written": 0, "reused": 0, "no_motion": 0}
    for src in tqdm(npz_files, desc="AMASS->pose"):
        save_path = pose_path_for(src, amass_root, pose_root)
        if save_path.is_file():
            counts["reused"] += 1
            continue
        joints = extractor.joints_of(src)
        if joints is None:
            counts["no_motion"] += 1
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_numpy_save(joints, save_path)
        counts["written"] += 1
    return counts


def extract_poses(
    amass_root: Path, smplx_models: Path, pose_root: Path, fps: int, adopt: bool = False
) -> dict[str, int]:
    extractor = AmassSource(smplx_models, fps=fps)
    recipe = {"format": POSE_FORMAT, "version": POSE_VERSION, **extractor.recipe()}
    npz_files = sorted(
        Path(r) / f for r, _, fs in os.walk(amass_root) for f in fs if f.endswith(".npz")
    )
    manifest = open_pose_manifest(extractor, amass_root, pose_root, npz_files, recipe, adopt)
    counts = write_missing_poses(extractor, amass_root, pose_root, npz_files)
    write_manifest(pose_root / POSE_MANIFEST, recipe, {**manifest, **counts, "complete": True})
    return counts
