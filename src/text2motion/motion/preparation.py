from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.motion.amass import (
    AmassPoseExtractor,
    head_trim_frames,
    head_trim_table,
    heldout_sources,
    resolve_pose_path,
)
from text2motion.motion.contracts import PreparationRequest, PreparationStage, Split
from text2motion.motion.normalization import fit_train_stats, read_split_names
from text2motion.motion.representation import (
    DIM,
    JOINTS,
    encode_joints,
    feature_params,
    recover_from_ric,
    target_offsets,
)
from text2motion.motion.storage import (
    FEATURES_DIR,
    SPLIT_FILES,
    atomic_numpy_save,
    sha256_file,
)

logger = logging.getLogger(__name__)

NPY_FILES = "*.npy"
CORPUS_MANIFEST = "corpus_manifest.json"
CORPUS_FORMAT = "text2motion.amass_corpus"
CORPUS_VERSION = 2
POSE_MANIFEST = "pose_manifest.json"
INDEX_MANIFEST = "index_manifest.json"
FEATURES_MANIFEST = "features_manifest.json"
POSE_FORMAT = "text2motion.amass_poses"
POSE_VERSION = 2
_ADOPT_SAMPLE = 20
_ADOPT_TOLERANCE = 1e-4
_REFERENCE_CLIP_ID = "000021"
_HML3D_FPS = 20
_RIGHT_CHAIN = (2, 5, 8, 11, 14, 17, 19, 21)
_LEFT_CHAIN = (1, 4, 7, 10, 13, 16, 18, 20)
_LEFT_HAND_CHAIN = (22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30)
_RIGHT_HAND_CHAIN = (43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51)


def __parquet_files(src: Path, split: str) -> list[Path]:
    matches = sorted(src.glob(f"**/{split}-*.parquet"))
    if not matches:
        matches = sorted(p for p in src.glob("**/*.parquet") if split in p.name)
    return matches


def __parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "the official-release converter needs pyarrow: run uv sync --extra prep"
        ) from exc
    return pq


def checked_release_motion(name: str, motion) -> np.ndarray:
    feature = np.asarray(motion, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] == 0 or feature.shape[-1] != DIM:
        raise ValueError(f"{name}: expected [frames, {DIM}], got {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError(f"{name}: the motion holds NaN or Inf values")
    return feature


def check_hml3d_fps(fps: int) -> None:
    if fps != _HML3D_FPS:
        raise ValueError(
            f"the HumanML3D index frames and feature constants are {_HML3D_FPS} fps; got fps={fps}"
        )


def write_motions(src: Path, out_dir: Path) -> dict[str, list[str]]:
    pq = __parquet_module()

    vec_dir = out_dir / FEATURES_DIR
    vec_dir.mkdir(parents=True, exist_ok=True)
    split_names: dict[str, list[str]] = {split: [] for split in Split}
    seen: dict[str, str] = {}

    for split in Split:
        files = __parquet_files(src, split)
        if not files:
            raise FileNotFoundError(f"no parquet files for split {split!r} under {src}")

        for parquet_file in files:
            reader = pq.ParquetFile(parquet_file)
            desc = f"{split}:{parquet_file.name}"
            for batch in tqdm(reader.iter_batches(batch_size=256), desc=desc):
                rows = batch.to_pydict()
                for motion, meta in zip(rows["motion"], rows["meta_data"], strict=True):
                    name = meta["name"]
                    if name in seen:
                        raise ValueError(
                            f"{name} appears in {seen[name]!r} and again in {split!r}; "
                            "one feature file cannot belong to two splits"
                        )
                    seen[name] = split
                    atomic_numpy_save(checked_release_motion(name, motion), vec_dir / f"{name}.npy")
                    split_names[split].append(name)

    for split, names in split_names.items():
        (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    return split_names


def compute_mean_std(out_dir: Path, train_names: list[str]) -> None:
    fit_train_stats(out_dir / FEATURES_DIR, train_names, out_dir, JOINTS)


def convert_official_release(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split_names = write_motions(src, out)
    for split, names in split_names.items():
        logger.info(f"{split}: {len(names)} clips")
    compute_mean_std(out, split_names[Split.TRAIN])


def swap_left_right(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    tmp = data[:, _RIGHT_CHAIN]
    data[:, _RIGHT_CHAIN] = data[:, _LEFT_CHAIN]
    data[:, _LEFT_CHAIN] = tmp
    if data.shape[1] > 24:
        tmp = data[:, _RIGHT_HAND_CHAIN]
        data[:, _RIGHT_HAND_CHAIN] = data[:, _LEFT_HAND_CHAIN]
        data[:, _LEFT_HAND_CHAIN] = tmp
    return data


############################################################
#       Hml3dPreparationLayout
############################################################
@dataclass(frozen=True)
class Hml3dPreparationLayout:
    out_dir: Path

    @property
    def pose_data(self) -> Path:
        return self.out_dir / "pose_data"

    @property
    def joints(self) -> Path:
        return self.out_dir / "joints"

    @property
    def new_joints(self) -> Path:
        return self.out_dir / "new_joints"

    @property
    def new_joint_vecs(self) -> Path:
        return self.out_dir / FEATURES_DIR

    def mkdirs(self) -> None:
        for d in (self.pose_data, self.joints, self.new_joints, self.new_joint_vecs):
            d.mkdir(parents=True, exist_ok=True)


def __pose_path(src: Path, amass_root: Path, pose_root: Path) -> Path:
    return (pose_root / src.relative_to(amass_root)).with_suffix(".npy")


def __same_pose(extracted: np.ndarray | None, saved: np.ndarray) -> bool:
    if extracted is None or extracted.shape != saved.shape:
        return False
    return bool(np.allclose(extracted, saved, rtol=0.0, atol=_ADOPT_TOLERANCE))


def adopt_pose_folder(
    extractor: AmassPoseExtractor,
    amass_root: Path,
    pose_root: Path,
    npz_files: list[Path],
    recipe: dict[str, object],
) -> None:
    manifest_path = pose_root / POSE_MANIFEST
    if manifest_path.is_file():
        raise ValueError(f"{manifest_path} exists; adoption is only for a folder without one")
    pairs = [(src, __pose_path(src, amass_root, pose_root)) for src in npz_files]
    pairs = [(src, saved) for src, saved in pairs if saved.is_file()]
    if not pairs:
        raise FileNotFoundError(f"{pose_root} has no pose files to adopt")
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(pairs), min(_ADOPT_SAMPLE, len(pairs)), replace=False))
    sample = [pairs[index] for index in picks]
    mismatched = [
        saved.name
        for src, saved in sample
        if not __same_pose(extractor.extract(src), np.load(saved))
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} of {len(sample)} sampled poses differ from a new extraction "
            f"(first: {mismatched[:3]}); re-extract into a new folder"
        )
    adopted = [str(saved.relative_to(pose_root)) for _, saved in sample]
    write_manifest(manifest_path, recipe, {"complete": False, "adopted_sample": adopted})


def extract_amass_to_y_up_joints(
    request: PreparationRequest, layout: Hml3dPreparationLayout, adopt: bool = False
) -> dict[str, int]:
    if request.amass_dir is None:
        raise ValueError("amass_dir must be set to the AMASS root for stage 'amass'")
    if request.smplx_models is None:
        raise ValueError("smplx_models must be set (SMPL-X bodies) for stage 'amass'")

    amass_root = Path(request.amass_dir)
    pose_root = layout.pose_data
    extractor = AmassPoseExtractor(
        Path(request.smplx_models),
        joint_count=request.joint_count,
        device=request.device,
        fps=request.fps,
    )
    recipe = {"format": POSE_FORMAT, "version": POSE_VERSION, **extractor.recipe()}
    npz_files = sorted(
        Path(r) / f for r, _, fs in os.walk(amass_root) for f in fs if f.endswith(".npz")
    )
    if adopt:
        adopt_pose_folder(extractor, amass_root, pose_root, npz_files, recipe)
    manifest_path = pose_root / POSE_MANIFEST
    check_recipe_reuse(manifest_path, recipe, any(pose_root.rglob(NPY_FILES)))
    previous = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {key: value for key, value in previous.items() if key == "adopted_sample"}
    write_manifest(manifest_path, recipe, {**manifest, "complete": False})

    counts = {"written": 0, "reused": 0, "no_motion": 0}
    for src in tqdm(npz_files, desc="AMASS->pose"):
        save_path = __pose_path(src, amass_root, pose_root)
        if save_path.is_file():
            counts["reused"] += 1
            continue
        joints = extractor.extract(src)
        if joints is None:
            counts["no_motion"] += 1
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_numpy_save(joints, save_path)
        counts["written"] += 1

    write_manifest(manifest_path, recipe, {**manifest, **counts, "complete": True})
    return counts


def build_indexed_joint_clips(
    request: PreparationRequest, layout: Hml3dPreparationLayout, pose_root: Path | None = None
) -> None:
    if request.index_csv is None:
        raise ValueError("index_csv must be set for stage 'index'")
    fps = request.fps
    import pandas as pd

    index_file = pd.read_csv(request.index_csv)
    pose_root = pose_root if pose_root is not None else layout.pose_data

    written = 0
    missing: list[str] = []
    invalid: list[str] = []
    for i in tqdm(range(index_file.shape[0]), desc="index->joints"):
        source_path = index_file.loc[i]["source_path"]
        new_name = index_file.loc[i]["new_name"]
        start_frame = int(index_file.loc[i]["start_frame"])
        end_frame = int(index_file.loc[i]["end_frame"])

        load_path = resolve_pose_path(source_path, pose_root)
        if load_path is None:
            missing.append(str(new_name))
            continue
        data = np.load(load_path)

        if "humanact12" not in source_path:
            trim = head_trim_frames(source_path, fps)
            if trim:
                data = data[trim:]
            if start_frame < 0 or end_frame <= start_frame:
                invalid.append(str(new_name))
                continue
            data = data[start_frame:end_frame]
            data[..., 0] *= -1
        if data.shape[0] < 2:
            invalid.append(str(new_name))
            continue

        data_m = swap_left_right(data)
        atomic_numpy_save(data, layout.joints / f"{new_name}")
        atomic_numpy_save(data_m, layout.joints / f"M{new_name}")
        written += 1

    recipe = {"index_sha256": sha256_file(Path(request.index_csv)), "fps": fps}
    status = {
        "written": written,
        "missing": len(missing),
        "missing_names": missing,
        "invalid_names": invalid,
        "complete": not missing and not invalid,
    }
    write_manifest(layout.out_dir / INDEX_MANIFEST, recipe, status)
    if invalid:
        raise ValueError(
            f"index->joints: {len(invalid)} rows have bad crop bounds or fewer than 2 frames "
            f"(names in {INDEX_MANIFEST}); first: {invalid[:3]}"
        )
    if missing:
        logger.warning(
            f"index->joints: {len(missing)} of {len(index_file)} clips have no pose file "
            f"(names in {INDEX_MANIFEST}); first: {missing[:3]}"
        )
    logger.info(f"index->joints: wrote {written} clips")


def __hml3d_features(source: np.ndarray, params, joint_count: int):
    data, _, _, _ = encode_joints(source, params)
    if not np.isfinite(data).all():
        raise ValueError("non-finite features")
    rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joint_count)
    return data, rec.squeeze().numpy()


def encode_joint_clips_to_hml3d263(
    request: PreparationRequest, layout: Hml3dPreparationLayout
) -> None:
    joint_count = request.joint_count
    ref_path = __reference_joints_path(layout.joints)
    reference = np.load(ref_path)[:, :joint_count]
    reference = reference.reshape(len(reference), -1, 3)
    params = feature_params(target_offsets(reference))

    source_list = sorted(p.name for p in layout.joints.glob(NPY_FILES))
    frame_num = 0
    failed: list[str] = []
    for source_file in tqdm(source_list, desc="joints->263"):
        source_data = np.load(layout.joints / source_file)[:, :joint_count]
        try:
            data, recovered = __hml3d_features(source_data, params, joint_count)
        except Exception as exc:
            failed.append(source_file)
            logger.warning(f"FAILED {source_file}: {exc}")
            continue
        atomic_numpy_save(recovered, layout.new_joints / source_file)
        atomic_numpy_save(data, layout.new_joint_vecs / source_file)
        frame_num += data.shape[0]

    recipe = {
        "reference_joints_sha256": sha256_file(ref_path),
        "joint_count": joint_count,
        "fps": request.fps,
    }
    status = {
        "written": len(source_list) - len(failed),
        "frames": frame_num,
        "failed_names": failed,
        "complete": not failed,
    }
    write_manifest(layout.out_dir / FEATURES_MANIFEST, recipe, status)
    logger.info(f"joints->263: {status['written']} clips, {frame_num} frames")
    if failed:
        raise RuntimeError(
            f"{len(failed)} joint clips failed to encode (names in {FEATURES_MANIFEST}); "
            f"first: {failed[:3]}"
        )


def compute_hml3d_normalization_stats(
    layout: Hml3dPreparationLayout, joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    train_names = read_split_names(layout.out_dir / SPLIT_FILES[Split.TRAIN])
    return fit_train_stats(layout.new_joint_vecs, train_names, layout.out_dir, joint_count)


def run_hml3d_preparation(
    request: PreparationRequest,
    stage: PreparationStage | str = PreparationStage.ALL,
    adopt_pose_folder: bool = False,
) -> None:
    stage = PreparationStage(stage)
    check_hml3d_fps(request.fps)
    layout = Hml3dPreparationLayout(Path(request.out_dir))
    layout.mkdirs()

    if stage in (PreparationStage.AMASS, PreparationStage.ALL):
        logger.info(extract_amass_to_y_up_joints(request, layout, adopt_pose_folder))
    if stage in (PreparationStage.INDEX, PreparationStage.ALL):
        build_indexed_joint_clips(request, layout)
    if stage in (PreparationStage.FEATURE, PreparationStage.ALL):
        encode_joint_clips_to_hml3d263(request, layout)
    if stage in (PreparationStage.STATS, PreparationStage.ALL):
        compute_hml3d_normalization_stats(layout, request.joint_count)


def source_fingerprint(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def __stored_fingerprints(manifest_path: Path) -> dict[str, list[int]]:
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
        if path.stat().st_size in heldout_sizes and sha256_file(path) in heldout_hashes:
            names.add(corpus_feature_name(path, pose_root))
    return names


def __reference_joints_path(joints_dir: Path) -> Path:
    preferred = joints_dir / f"{_REFERENCE_CLIP_ID}.npy"
    if not preferred.is_file():
        raise FileNotFoundError(
            f"{preferred} missing; the official HumanML3D skeleton reference is clip "
            f"{_REFERENCE_CLIP_ID}, and another clip would change every bone length"
        )
    return preferred


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


def check_recipe_reuse(manifest_path: Path, recipe: dict[str, object], has_outputs: bool) -> None:
    folder = manifest_path.parent
    if not manifest_path.is_file():
        if has_outputs:
            raise ValueError(
                f"{folder} has outputs but no {manifest_path.name} (legacy folder); "
                "build into a new folder"
            )
        return
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))["recipe"]
    changed = sorted(key for key in set(stored) | set(recipe) if stored.get(key) != recipe.get(key))
    if changed:
        raise ValueError(
            f"{folder} was built with another recipe (different: {changed}); "
            "build into a new folder"
        )


def write_manifest(
    manifest_path: Path, recipe: dict[str, object], status: dict[str, object]
) -> None:
    manifest = {"recipe": recipe, **status}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def __corpus_features(
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


def build_amass_pretraining_corpus(
    request: PreparationRequest,
    regen_dir: Path,
    out_dir: Path,
    min_frames: int,
    limit: int = 0,
) -> dict[str, float]:
    pose_root = regen_dir / "pose_data"
    if not pose_root.is_dir():
        raise FileNotFoundError(
            f"{pose_root} missing -- run 'text2motion prepare --stage amass' first"
        )
    if request.index_csv is None:
        raise ValueError("index_csv must be set (the leakage guard needs the official index)")
    check_hml3d_fps(request.fps)
    index_csv = Path(request.index_csv)
    reference_path = __reference_joints_path(regen_dir / "joints")
    joint_count = request.joint_count
    reference = np.load(reference_path)[:, :joint_count].reshape(-1, joint_count, 3)
    params = feature_params(target_offsets(reference))

    pose_recipe = complete_pose_recipe(pose_root)
    recipe = corpus_recipe(request, index_csv, reference_path, pose_recipe, min_frames)
    vec_dir = out_dir / FEATURES_DIR
    existing = {path.name for path in vec_dir.glob(NPY_FILES)}
    manifest_path = out_dir / CORPUS_MANIFEST
    check_recipe_reuse(manifest_path, recipe, bool(existing))
    excluded = heldout_feature_names(index_csv, Path(request.out_dir), pose_root)
    present = sorted(excluded & existing)
    if present:
        raise ValueError(
            f"{len(present)} heldout files are already in {vec_dir} (first: {present[:3]}); "
            "build into a new folder"
        )
    previous_fingerprints = __stored_fingerprints(manifest_path)
    sources = sorted(pose_root.rglob(NPY_FILES))
    if not sources:
        raise FileNotFoundError(f"no pose files under {pose_root}")
    if limit:
        sources = sources[:limit]
    names = [corpus_feature_name(src, pose_root) for src in sources]
    check_unique_names(names)
    fingerprints = {name: source_fingerprint(src) for name, src in zip(names, sources, strict=True)}
    vec_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path, recipe, {"complete": False, "source_fingerprints": previous_fingerprints}
    )
    counts = {"written": 0, "reused": 0, "skipped_heldout": 0, "skipped_short": 0, "frames": 0}
    failed: list[str] = []
    for src in tqdm(sources, desc="pose->263 (all AMASS)"):
        name = corpus_feature_name(src, pose_root)
        if name in excluded:
            counts["skipped_heldout"] += 1
            continue
        save_path = vec_dir / name
        if save_path.is_file():
            if previous_fingerprints.get(name) != fingerprints[name]:
                raise ValueError(
                    f"{save_path} exists, but its source {src} changed or is not recorded; "
                    "build into a new folder"
                )
            counts["reused"] += 1
            counts["frames"] += int(np.load(save_path, mmap_mode="r").shape[0])
            continue
        try:
            features = __corpus_features(src, pose_root, request, params, min_frames)
        except Exception as exc:
            failed.append(name)
            logger.warning(f"FAILED {name}: {exc}")
            continue
        if features is None:
            counts["skipped_short"] += 1
            continue
        atomic_numpy_save(features, save_path)
        counts["written"] += 1
        counts["frames"] += features.shape[0]

    summary = {
        **counts,
        "failed": len(failed),
        "hours": round(counts["frames"] / request.fps / 3600, 2),
    }
    write_manifest(
        manifest_path,
        recipe,
        {
            **summary,
            "limit": limit,
            "complete": not failed and not limit,
            "failed_names": failed,
            "source_fingerprints": fingerprints,
        },
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} source files failed to convert (names in {CORPUS_MANIFEST}); "
            f"first: {failed[:3]}"
        )
    return summary
