from __future__ import annotations

import json
import logging
import os
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
from text2motion.motion.storage import FEATURES_DIR, SPLIT_FILES, sha256_file

logger = logging.getLogger(__name__)

CORPUS_MANIFEST = "corpus_manifest.json"
CORPUS_FORMAT = "text2motion.amass_corpus"
CORPUS_VERSION = 1
_REFERENCE_CLIP_ID = "000021"
_RIGHT_CHAIN = [2, 5, 8, 11, 14, 17, 19, 21]
_LEFT_CHAIN = [1, 4, 7, 10, 13, 16, 18, 20]
_LEFT_HAND_CHAIN = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
_RIGHT_HAND_CHAIN = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]


def __parquet_files(src: Path, split: str) -> list[Path]:
    matches = sorted(src.glob(f"**/{split}-*.parquet"))
    if not matches:
        matches = sorted(p for p in src.glob("**/*.parquet") if split in p.name)
    return matches


def write_motions(src: Path, out_dir: Path) -> dict[str, list[str]]:
    import pyarrow.parquet as pq

    vec_dir = out_dir / FEATURES_DIR
    vec_dir.mkdir(parents=True, exist_ok=True)
    split_names: dict[str, list[str]] = {split: [] for split in Split}

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
                    feature = np.asarray(motion, dtype=np.float32)
                    if feature.shape[-1] != DIM:
                        raise ValueError(f"{name}: expected {DIM}-dim, got {feature.shape}")
                    np.save(vec_dir / f"{name}.npy", feature)
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


def extract_amass_to_y_up_joints(
    request: PreparationRequest, layout: Hml3dPreparationLayout
) -> None:
    if request.amass_dir is None:
        raise ValueError("amass_dir must be set to the AMASS root for stage 'amass'")
    if request.smplx_models is None:
        raise ValueError("smplx_models must be set (SMPL-X bodies) for stage 'amass'")

    amass_root = Path(request.amass_dir)
    extractor = AmassPoseExtractor(
        Path(request.smplx_models),
        joint_count=request.joint_count,
        device=request.device,
        fps=request.fps,
    )

    npz_files = [Path(r) / f for r, _, fs in os.walk(amass_root) for f in fs if f.endswith(".npz")]
    for src in tqdm(npz_files, desc="AMASS->pose"):
        rel = src.relative_to(amass_root)
        save_path = (layout.pose_data / rel).with_suffix(".npy")
        if save_path.is_file():
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joints = extractor.extract(src)
        if joints is not None:
            np.save(save_path, joints)


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
    missing = 0
    for i in tqdm(range(index_file.shape[0]), desc="index->joints"):
        source_path = index_file.loc[i]["source_path"]
        new_name = index_file.loc[i]["new_name"]
        start_frame = int(index_file.loc[i]["start_frame"])
        end_frame = int(index_file.loc[i]["end_frame"])

        load_path = resolve_pose_path(source_path, pose_root)
        if load_path is None:
            missing += 1
            continue
        data = np.load(load_path)

        if "humanact12" not in source_path:
            trim = head_trim_frames(source_path, fps)
            if trim:
                data = data[trim:]
            data = data[start_frame:end_frame]
            data[..., 0] *= -1

        data_m = swap_left_right(data)
        np.save(layout.joints / new_name, data)
        np.save(layout.joints / ("M" + new_name), data_m)
        written += 1

    logger.info(
        f"index->joints: wrote {written} clips, skipped {missing} missing (of {len(index_file)})"
    )


def encode_joint_clips_to_hml3d263(
    request: PreparationRequest, layout: Hml3dPreparationLayout
) -> None:
    joint_count = request.joint_count
    ref_path = layout.joints / f"{_REFERENCE_CLIP_ID}.npy"
    if not ref_path.is_file():
        available = sorted(layout.joints.glob("[0-9]*.npy"))
        if not available:
            raise FileNotFoundError(f"no joints in {layout.joints}; run stage 'index' first")
        ref_path = available[0]
        logger.info(f"reference skeleton {_REFERENCE_CLIP_ID} missing; using {ref_path.name}")
    reference = np.load(ref_path)[:, :joint_count]
    reference = reference.reshape(len(reference), -1, 3)
    params = feature_params(target_offsets(reference))

    source_list = sorted(p.name for p in layout.joints.glob("*.npy"))
    frame_num = 0
    for source_file in tqdm(source_list, desc="joints->263"):
        source_data = np.load(layout.joints / source_file)[:, :joint_count]
        try:
            data, _, _, _ = encode_joints(source_data, params)
            rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joint_count)
            np.save(layout.new_joints / source_file, rec.squeeze().numpy())
            np.save(layout.new_joint_vecs / source_file, data)
            frame_num += data.shape[0]
        except Exception as exc:
            logger.info(f"FAILED {source_file}: {exc}")

    logger.info(
        f"Total clips: {len(source_list)}, Frames: {frame_num}, "
        f"Duration: {frame_num / request.fps / 60:.4f}m"
    )


def compute_hml3d_normalization_stats(
    layout: Hml3dPreparationLayout, joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    train_names = read_split_names(layout.out_dir / SPLIT_FILES[Split.TRAIN])
    return fit_train_stats(layout.new_joint_vecs, train_names, layout.out_dir, joint_count)


def run_hml3d_preparation(
    request: PreparationRequest, stage: PreparationStage | str = PreparationStage.ALL
) -> None:
    stage = PreparationStage(stage)
    layout = Hml3dPreparationLayout(Path(request.out_dir))
    layout.mkdirs()

    if stage in (PreparationStage.AMASS, PreparationStage.ALL):
        extract_amass_to_y_up_joints(request, layout)
    if stage in (PreparationStage.INDEX, PreparationStage.ALL):
        build_indexed_joint_clips(request, layout)
    if stage in (PreparationStage.FEATURE, PreparationStage.ALL):
        encode_joint_clips_to_hml3d263(request, layout)
    if stage in (PreparationStage.STATS, PreparationStage.ALL):
        compute_hml3d_normalization_stats(layout, request.joint_count)


def corpus_feature_name(pose_path: Path, pose_root: Path) -> str:
    relative = pose_path.resolve().relative_to(pose_root.resolve())
    return "__".join(relative.with_suffix("").parts) + ".npy"


def heldout_feature_names(index_csv: Path, out_dir: Path, pose_root: Path) -> set[str]:
    heldout = heldout_sources(index_csv, out_dir, pose_root)
    heldout_sizes = {path.stat().st_size for path in heldout}
    heldout_hashes = {sha256_file(path) for path in heldout}
    names = {corpus_feature_name(path, pose_root) for path in heldout}
    for path in pose_root.rglob("*.npy"):
        if path.stat().st_size in heldout_sizes and sha256_file(path) in heldout_hashes:
            names.add(corpus_feature_name(path, pose_root))
    return names


def __reference_joints_path(joints_dir: Path) -> Path:
    preferred = joints_dir / f"{_REFERENCE_CLIP_ID}.npy"
    if preferred.is_file():
        return preferred
    candidates = sorted(joints_dir.glob("[0-9]*.npy"))
    if not candidates:
        raise FileNotFoundError(f"no reference joints in {joints_dir}")
    return candidates[0]


def corpus_recipe(
    request: PreparationRequest, index_csv: Path, reference_path: Path, min_frames: int
) -> dict[str, object]:
    dataset_dir = Path(request.out_dir)
    recipe = {
        "format": CORPUS_FORMAT,
        "version": CORPUS_VERSION,
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


def check_corpus_reuse(out_dir: Path, recipe: dict[str, object]) -> None:
    manifest_path = out_dir / CORPUS_MANIFEST
    if not manifest_path.is_file():
        if any((out_dir / FEATURES_DIR).glob("*.npy")):
            raise ValueError(
                f"{out_dir} has features but no {CORPUS_MANIFEST} (legacy corpus); "
                "build into a new folder"
            )
        return
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))["recipe"]
    changed = sorted(key for key in set(stored) | set(recipe) if stored.get(key) != recipe.get(key))
    if changed:
        raise ValueError(
            f"{out_dir} was built with another recipe (different: {changed}); "
            "build into a new folder"
        )


def write_corpus_manifest(
    out_dir: Path, recipe: dict[str, object], status: dict[str, object]
) -> None:
    manifest = {"recipe": recipe, **status}
    (out_dir / CORPUS_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def __corpus_features(
    src: Path, pose_root: Path, request: PreparationRequest, params, min_frames: int
) -> np.ndarray | None:
    joints = np.load(src)[:, : request.joint_count]
    trim = head_trim_frames(src.relative_to(pose_root), request.fps)
    joints = joints[trim:].copy()
    if joints.shape[0] < min_frames:
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
    index_csv = Path(request.index_csv)
    reference_path = __reference_joints_path(regen_dir / "joints")
    joint_count = request.joint_count
    reference = np.load(reference_path)[:, :joint_count].reshape(-1, joint_count, 3)
    params = feature_params(target_offsets(reference))

    recipe = corpus_recipe(request, index_csv, reference_path, min_frames)
    check_corpus_reuse(out_dir, recipe)
    vec_dir = out_dir / FEATURES_DIR
    excluded = heldout_feature_names(index_csv, Path(request.out_dir), pose_root)
    present = sorted(excluded & {path.name for path in vec_dir.glob("*.npy")})
    if present:
        raise ValueError(
            f"{len(present)} heldout files are already in {vec_dir} (first: {present[:3]}); "
            "build into a new folder"
        )
    vec_dir.mkdir(parents=True, exist_ok=True)
    write_corpus_manifest(out_dir, recipe, {"complete": False})

    sources = sorted(pose_root.rglob("*.npy"))
    if limit:
        sources = sources[:limit]
    counts = {"written": 0, "reused": 0, "skipped_heldout": 0, "skipped_short": 0, "frames": 0}
    failed: list[str] = []
    for src in tqdm(sources, desc="pose->263 (all AMASS)"):
        name = corpus_feature_name(src, pose_root)
        if name in excluded:
            counts["skipped_heldout"] += 1
            continue
        save_path = vec_dir / name
        if save_path.is_file():
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
        temporary = save_path.with_name(f".{save_path.stem}.tmp.npy")
        np.save(temporary, features)
        temporary.replace(save_path)
        counts["written"] += 1
        counts["frames"] += features.shape[0]

    summary = {
        **counts,
        "failed": len(failed),
        "hours": round(counts["frames"] / request.fps / 3600, 2),
    }
    write_corpus_manifest(
        out_dir, recipe, {**summary, "complete": not failed, "failed_names": failed}
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} source files failed to convert (names in {CORPUS_MANIFEST}); "
            f"first: {failed[:3]}"
        )
    return summary
