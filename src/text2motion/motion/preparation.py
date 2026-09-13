from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.motion.amass import (
    AmassPoseExtractor,
    head_trim_frames,
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
from text2motion.motion.storage import FEATURES_DIR, SPLIT_FILES

_REFERENCE_CLIP_ID = "000021"
_RIGHT_CHAIN = [2, 5, 8, 11, 14, 17, 19, 21]
_LEFT_CHAIN = [1, 4, 7, 10, 13, 16, 18, 20]
_LEFT_HAND_CHAIN = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
_RIGHT_HAND_CHAIN = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]


def _parquet_files(src: Path, split: str) -> list[Path]:
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
        print(f"{split}: {len(names)} clips")
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


def extract_amass_to_y_up_joints(request: PreparationRequest, layout: Hml3dPreparationLayout) -> None:
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

    print(f"index->joints: wrote {written} clips, skipped {missing} missing (of {len(index_file)})")


def encode_joint_clips_to_hml3d263(request: PreparationRequest, layout: Hml3dPreparationLayout) -> None:
    joint_count = request.joint_count
    ref_path = layout.joints / f"{_REFERENCE_CLIP_ID}.npy"
    if not ref_path.is_file():
        available = sorted(layout.joints.glob("[0-9]*.npy"))
        if not available:
            raise FileNotFoundError(f"no joints in {layout.joints}; run stage 'index' first")
        ref_path = available[0]
        print(f"reference skeleton {_REFERENCE_CLIP_ID} missing; using {ref_path.name}")
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
            print(f"FAILED {source_file}: {exc}")

    print(
        f"Total clips: {len(source_list)}, Frames: {frame_num}, "
        f"Duration: {frame_num / request.fps / 60:.4f}m"
    )


def compute_hml3d_normalization_stats(layout: Hml3dPreparationLayout, joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    train_names = read_split_names(layout.out_dir / SPLIT_FILES[Split.TRAIN])
    return fit_train_stats(layout.new_joint_vecs, train_names, layout.out_dir, joint_count)


def run_hml3d_preparation(request: PreparationRequest, stage: PreparationStage | str = PreparationStage.ALL) -> None:
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


def resolve_heldout(request: PreparationRequest, pose_root: Path) -> set[Path]:
    if request.index_csv is None:
        raise ValueError("index_csv must be set (the leakage guard needs the official index)")
    return heldout_sources(Path(request.index_csv), Path(request.out_dir), pose_root)


def build_amass_pretraining_corpus(
    request: PreparationRequest,
    regen_dir: Path,
    out_dir: Path,
    min_frames: int,
    limit: int = 0,
) -> dict[str, float]:
    pose_root = regen_dir / "pose_data"
    joints_dir = regen_dir / "joints"
    if not pose_root.is_dir():
        raise FileNotFoundError(
            f"{pose_root} missing -- run 'text2motion prepare --stage amass' first"
        )

    ref_path = joints_dir / f"{_REFERENCE_CLIP_ID}.npy"
    if not ref_path.is_file():
        candidates = sorted(joints_dir.glob("[0-9]*.npy"))
        if not candidates:
            raise FileNotFoundError(f"no reference joints in {joints_dir}")
        ref_path = candidates[0]
    joint_count = request.joint_count
    reference = np.load(ref_path)[:, :joint_count].reshape(-1, joint_count, 3)
    params = feature_params(target_offsets(reference))

    excluded = resolve_heldout(request, pose_root)
    vec_dir = out_dir / FEATURES_DIR
    vec_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(pose_root.rglob("*.npy"))
    if limit:
        sources = sources[:limit]
    written = skipped_heldout = skipped_short = failed = total_frames = 0
    for src in tqdm(sources, desc="pose->263 (all AMASS)"):
        if src.resolve() in excluded:
            skipped_heldout += 1
            continue
        rel = src.relative_to(pose_root)
        flat_name = "__".join(rel.with_suffix("").parts) + ".npy"
        save_path = vec_dir / flat_name
        if save_path.is_file():
            written += 1
            continue
        joints = np.load(src)[:, :joint_count]
        trim = head_trim_frames(rel, request.fps)
        if trim:
            joints = joints[trim:]
        if joints.shape[0] < min_frames:
            skipped_short += 1
            continue
        joints = joints.copy()
        joints[..., 0] *= -1
        try:
            features, _, _, _ = encode_joints(joints, params)
            if not np.isfinite(features).all():
                raise ValueError("non-finite features")
            np.save(save_path, features.astype(np.float32))
            written += 1
            total_frames += features.shape[0]
        except Exception as exc:
            failed += 1
            print(f"FAILED {rel}: {exc}")

    summary = {
        "written": written,
        "skipped_heldout": skipped_heldout,
        "skipped_short": skipped_short,
        "failed": failed,
        "frames": total_frames,
        "hours": round(total_frames / request.fps / 3600, 2),
    }
    return summary
