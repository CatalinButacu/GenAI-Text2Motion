from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from text2motion.shared.config import Config, Hml3dReprCfg, PathsCfg, load_config

from . import param_util
from .feature import build_tgt_offsets, default_params, process_file, recover_from_ric
from .raw_pose import AmassPoseExtractor

_DATASET_HEAD_TRIM_S = {
    "Eyes_Japan_Dataset": 3.0,
    "MPI_HDM05": 3.0,
    "TotalCapture": 1.0,
    "MPI_Limits": 1.0,
    "Transitions_mocap": 0.5,
}

_AMASS_DATASET_RENAME = {
    "MPI_HDM05": "HDM05",
    "BioMotionLab_NTroje": "BMLrub",
    "MPI_Limits": "PosePrior",
    "MPI_mosh": "MoSh",
    "DFaust_67": "DFaust",
    "SSM_synced": "SSM",
    "TCD_handMocap": "TCDHands",
    "Transitions_mocap": "Transitions",
    "Eyes_Japan_Dataset": "EyesJapanDataset",
}


def _resolve_pose_path(source_path: str, pose_root: Path) -> Path | None:
    parts = [p for p in Path(source_path).parts if p not in (".", "pose_data")]
    if not parts:
        return None
    dataset = _AMASS_DATASET_RENAME.get(parts[0], parts[0])
    rel = Path(dataset, *parts[1:])
    stem = rel.stem
    base = stem[:-6] if stem.endswith("_poses") else stem
    folder = pose_root / rel.parent
    for cand in (f"{base}_stageii.npy", f"{base}_poses.npy", f"{stem}.npy", f"{base}.npy"):
        if (folder / cand).is_file():
            return folder / cand
    return None


def swap_left_right(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = param_util.mirror_right_chain
    left_chain = param_util.mirror_left_chain
    left_hand_chain = param_util.mirror_left_hand_chain
    right_hand_chain = param_util.mirror_right_hand_chain
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


@dataclass(frozen=True)
class RegenLayout:
    out_dir: Path

    @property
    def pose_data(self) -> Path:
        return self.out_dir / "pose_data"  # stage 1: raw AMASS forward

    @property
    def joints(self) -> Path:
        return self.out_dir / "joints"  # stage 2: cropped + mirrored joints

    @property
    def new_joints(self) -> Path:
        return self.out_dir / "new_joints"  # stage 3: recovered (T, 22, 3)

    @property
    def new_joint_vecs(self) -> Path:
        return self.out_dir / "new_joint_vecs"  # stage 3: 263 features

    def mkdirs(self) -> None:
        for d in (self.pose_data, self.joints, self.new_joints, self.new_joint_vecs):
            d.mkdir(parents=True, exist_ok=True)


def stage_amass_to_pose(paths: PathsCfg, layout: RegenLayout, device: str) -> None:
    if paths.amass_dir is None:
        raise ValueError("paths.amass_dir must be set to the AMASS root for stage 'amass'")
    amass_root = Path(paths.amass_dir)
    if paths.smplx_models is None:
        raise ValueError("paths.smplx_models must be set (SMPL-X bodies) for stage 'amass'")

    extractor = AmassPoseExtractor(Path(paths.smplx_models), device=device)

    npz_files = [Path(r) / f for r, _, fs in os.walk(amass_root) for f in fs if f.endswith(".npz")]
    for src in tqdm(npz_files, desc="AMASS->pose"):
        rel = src.relative_to(amass_root)
        save_path = (layout.pose_data / rel).with_suffix(".npy")
        if save_path.is_file():
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joints = extractor.amass_to_pose(src)
        if joints is not None:
            np.save(save_path, joints)


def _head_trim(source_path: str, fps: int) -> int:
    for name, seconds in _DATASET_HEAD_TRIM_S.items():
        if name in source_path:
            return int(round(seconds * fps))
    return 0


def stage_index_to_joints(
    paths: PathsCfg, layout: RegenLayout, fps: int, pose_root: Path | None = None
) -> None:
    if paths.hml3d_index_csv is None:
        raise ValueError("paths.hml3d_index_csv must be set for stage 'index'")
    index_file = pd.read_csv(paths.hml3d_index_csv)
    pose_root = pose_root if pose_root is not None else layout.pose_data

    written = 0
    missing = 0
    for i in tqdm(range(index_file.shape[0]), desc="index->joints"):
        source_path = index_file.loc[i]["source_path"]
        new_name = index_file.loc[i]["new_name"]
        start_frame = int(index_file.loc[i]["start_frame"])
        end_frame = int(index_file.loc[i]["end_frame"])

        load_path = _resolve_pose_path(source_path, pose_root)
        if load_path is None:
            missing += 1
            continue
        data = np.load(load_path)

        if "humanact12" not in source_path:
            trim = _head_trim(source_path, fps)
            if trim:
                data = data[trim:]
            data = data[start_frame:end_frame]
            data[..., 0] *= -1

        data_m = swap_left_right(data)
        np.save(layout.joints / new_name, data)
        np.save(layout.joints / ("M" + new_name), data_m)
        written += 1

    print(f"index->joints: wrote {written} clips, skipped {missing} missing (of {len(index_file)})")


def stage_joints_to_feature(layout: RegenLayout, repr_cfg: Hml3dReprCfg) -> None:
    joints_num = repr_cfg.num_joints
    ref_path = layout.joints / f"{param_util.t2m_tgt_skel_id}.npy"
    if not ref_path.is_file():
        available = sorted(layout.joints.glob("[0-9]*.npy"))
        if not available:
            raise FileNotFoundError(f"no joints in {layout.joints}; run stage 'index' first")
        ref_path = available[0]
        print(f"reference skeleton {param_util.t2m_tgt_skel_id} missing; using {ref_path.name}")
    reference = np.load(ref_path)[:, :joints_num]
    reference = reference.reshape(len(reference), -1, 3)
    tgt_offsets = build_tgt_offsets(reference)
    params = default_params(tgt_offsets)

    source_list = sorted(p.name for p in layout.joints.glob("*.npy"))
    frame_num = 0
    for source_file in tqdm(source_list, desc="joints->263"):
        source_data = np.load(layout.joints / source_file)[:, :joints_num]
        try:
            data, _, _, _ = process_file(source_data, params)
            rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            np.save(layout.new_joints / source_file, rec.squeeze().numpy())
            np.save(layout.new_joint_vecs / source_file, data)
            frame_num += data.shape[0]
        except Exception as exc:  # report loudly, do not silently produce a fallback feature
            print(f"FAILED {source_file}: {exc}")

    print(
        f"Total clips: {len(source_list)}, Frames: {frame_num}, "
        f"Duration: {frame_num / repr_cfg.fps / 60:.4f}m"
    )


def stage_mean_std(layout: RegenLayout, joints_num: int) -> tuple[np.ndarray, np.ndarray]:
    file_list = sorted(p.name for p in layout.new_joint_vecs.glob("*.npy"))
    data_list = []
    for file in file_list:
        arr = np.load(layout.new_joint_vecs / file)
        if np.isnan(arr).any():
            print(f"NaN feature, skipping: {file}")
            continue
        data_list.append(arr)

    data = np.concatenate(data_list, axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[0:1] = std[0:1].mean() / 1.0
    std[1:3] = std[1:3].mean() / 1.0
    std[3:4] = std[3:4].mean() / 1.0
    std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3].mean() / 1.0
    std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
        std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9].mean() / 1.0
    )
    std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3] = (
        std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0
    )
    std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
        std[4 + (joints_num - 1) * 9 + joints_num * 3 :].mean() / 1.0
    )

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == std.shape[-1]

    np.save(layout.out_dir / "Mean.npy", mean)
    np.save(layout.out_dir / "Std.npy", std)
    print(f"Mean/Std saved: feature shape {data.shape}")
    return mean, std


def run(cfg: Config, stage: str) -> None:
    if cfg.paths.hml3d_out_dir is None:
        raise ValueError("paths.hml3d_out_dir must be set (regenerated output root)")
    layout = RegenLayout(Path(cfg.paths.hml3d_out_dir))
    layout.mkdirs()

    if stage in ("amass", "all"):
        stage_amass_to_pose(cfg.paths, layout, cfg.device)
    if stage in ("index", "all"):
        stage_index_to_joints(cfg.paths, layout, cfg.hml3d.fps)
    if stage in ("feature", "all"):
        stage_joints_to_feature(layout, cfg.hml3d)
    if stage in ("stats", "all"):
        stage_mean_std(layout, cfg.hml3d.num_joints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate standard HumanML3D-263 from AMASS.")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["amass", "index", "feature", "stats", "all"],
        help="which pipeline stage(s) to run",
    )
    args = parser.parse_args()
    run(load_config(args.config), args.stage)


if __name__ == "__main__":
    main()
