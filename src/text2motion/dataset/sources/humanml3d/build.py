from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.common.manifest import write_manifest
from text2motion.dataset.contracts import AMASS_LABEL_FPS
from text2motion.dataset.core.operations import DATA_CLIP_ERRORS, NPY_FILES
from text2motion.dataset.sources.amass.clock import USABLE_STATUS, stored_fps_table
from text2motion.dataset.sources.amass.poses import (
    CLOCKED_POSES,
    extract_poses,
    write_clocked_poses,
)
from text2motion.dataset.sources.humanml3d.index_stage import build_indexed_joint_clips
from text2motion.dataset.sources.humanml3d.layout import Hml3dPreparationLayout
from text2motion.dataset.sources.humanml3d.official import OFFICIAL_SPLIT_FILES
from text2motion.dataset.sources.humanml3d.source import (
    HumanML3DSource,
    check_hml3d_fps,
    reference_joints_path,
)
from text2motion.dataset.splits.plan import SplitPlan
from text2motion.dataset.splits.view import ALLOWED_TRAIN_FILE
from text2motion.motion.contracts import PreparationRequest, Split
from text2motion.motion.normalization import fit_train_stats
from text2motion.motion.representation import (
    encode_joints,
    feature_params,
    recover_from_ric,
    target_offsets,
)
from text2motion.motion.storage import (
    SPLIT_FILES,
    atomic_numpy_save,
    sha256_file,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

FEATURES_MANIFEST = "features_manifest.json"


def joints_to_263(source: np.ndarray, params, joint_count: int):
    data, _, _, _ = encode_joints(source, params)
    if not np.isfinite(data).all():
        raise ValueError("non-finite features")
    rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joint_count)
    return data, rec.squeeze().numpy()


def reference_params(ref_path: Path, joint_count: int, fps: int):
    reference = np.load(ref_path)[:, :joint_count]
    reference = reference.reshape(len(reference), -1, 3)
    return feature_params(target_offsets(reference), fps)


def write_encoded_clips(
    source_list: list[str], layout: Hml3dPreparationLayout, params, joint_count: int
) -> tuple[int, list[str]]:
    frame_num = 0
    failed: list[str] = []
    for source_file in tqdm(source_list, desc="joints->263"):
        source_data = np.load(layout.joints / source_file)[:, :joint_count]
        try:
            data, recovered = joints_to_263(source_data, params, joint_count)
        except DATA_CLIP_ERRORS as exc:
            failed.append(source_file)
            logger.warning(f"FAILED {source_file}: {exc}")
            continue
        atomic_numpy_save(recovered, layout.new_joints / source_file)
        atomic_numpy_save(data, layout.new_joint_vecs / source_file)
        frame_num += data.shape[0]
    return frame_num, failed


def encode_joint_clips_to_hml3d263(
    request: PreparationRequest, layout: Hml3dPreparationLayout
) -> None:
    joint_count = request.joint_count
    ref_path = reference_joints_path(layout.joints, request.reference_clip)
    params = reference_params(ref_path, joint_count, request.fps)

    source_list = sorted(p.name for p in layout.joints.glob(NPY_FILES))
    frame_num, failed = write_encoded_clips(source_list, layout, params, joint_count)

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


def fill_from_official(official_dir: Path, layout: Hml3dPreparationLayout, fps: int) -> int:
    return HumanML3DSource(official_dir).fill_missing(
        layout.out_dir, layout.joints, fps, splits=("train", "val", "test")
    )


def open_layout(request: PreparationRequest) -> Hml3dPreparationLayout:
    check_hml3d_fps(request.fps)
    layout = Hml3dPreparationLayout(Path(request.out_dir))
    layout.mkdirs()
    return layout


def unprotected_clip_rates(clock_csv: Path, official_dir: Path) -> list[tuple[str, int]]:
    with open(clock_csv, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] in USABLE_STATUS]
    protected = SplitPlan.from_official(official_dir).protected_sources
    return [
        (row["source_path"], int(row["stored_fps"]))
        for row in rows
        if row["source_path"] not in protected
    ]


def require_amass_inputs(request: PreparationRequest) -> tuple[Path, Path]:
    if request.amass_dir is None:
        raise ValueError("amass_dir must be set to the AMASS root for the clock and joints stages")
    if request.smplx_models is None:
        raise ValueError("smplx_models must be set (SMPL-X bodies) for the clock and joints stages")
    return Path(request.amass_dir), Path(request.smplx_models)


def write_clock_table(request: PreparationRequest) -> None:
    if request.clock_csv is None or request.index_csv is None:
        raise ValueError("clock_csv and index_csv must be set for stage 'clock'")
    amass_dir, smplx_models = require_amass_inputs(request)
    layout = open_layout(request)
    official_dir = Path(request.index_csv).parent
    logger.info(
        extract_poses(
            amass_dir, smplx_models, layout.pose_data, request.fps, request.adopt_pose_folder
        )
    )
    rows = [
        row.as_record()
        for row in stored_fps_table(official_dir, layout.pose_data, AMASS_LABEL_FPS // request.fps)
    ]
    Path(request.clock_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(request.clock_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    statuses = [str(row["status"]) for row in rows]
    print("files", len(rows), {status: statuses.count(status) for status in sorted(set(statuses))})


def copy_official_splits(official_dir: Path, out_dir: Path) -> None:
    for split in OFFICIAL_SPLIT_FILES:
        shutil.copyfile(official_dir / split, out_dir / split)


def write_allowed_train_list(official_dir: Path, layout: Hml3dPreparationLayout) -> None:
    allowed = sorted(SplitPlan.from_official(official_dir).training_names())
    with_joints = [name for name in allowed if (layout.joints / f"{name}.npy").is_file()]
    (layout.out_dir / ALLOWED_TRAIN_FILE).write_text(
        "\n".join(with_joints) + "\n", encoding="utf-8"
    )
    print("allowed training clips with joints:", len(with_joints), flush=True)


def build_joint_clips(request: PreparationRequest) -> None:
    if request.clock_csv is None or request.index_csv is None:
        raise ValueError("clock_csv and index_csv must be set for stage 'joints'")
    amass_dir, smplx_models = require_amass_inputs(request)
    layout = open_layout(request)
    official_dir = Path(request.index_csv).parent
    copy_official_splits(official_dir, layout.out_dir)
    logger.info(
        extract_poses(
            amass_dir, smplx_models, layout.pose_data, request.fps, request.adopt_pose_folder
        )
    )
    counts = write_clocked_poses(
        amass_dir,
        smplx_models,
        layout.pose_data,
        layout.out_dir / CLOCKED_POSES,
        unprotected_clip_rates(Path(request.clock_csv), official_dir),
        request.fps,
    )
    print("clocked pose files by stored fps:", counts, flush=True)
    shutil.rmtree(layout.joints)
    layout.mkdirs()
    build_indexed_joint_clips(request, layout, layout.out_dir / CLOCKED_POSES)
    if request.fill:
        filled = fill_from_official(official_dir, layout, request.fps)
        print("clips filled from official features:", filled, flush=True)
        write_allowed_train_list(official_dir, layout)


def encode_hml3d_features(request: PreparationRequest) -> None:
    encode_joint_clips_to_hml3d263(request, open_layout(request))


def fit_hml3d_statistics(request: PreparationRequest) -> None:
    layout = open_layout(request)
    train_split = layout.out_dir / SPLIT_FILES[Split.TRAIN]
    if not train_split.is_file():
        raise FileNotFoundError(
            f"{train_split} missing: normalization statistics must be fitted on the train "
            f"split alone. Fitting over every feature file would include val and test, "
            f"leaking held-out data into the scaler used by every downstream model."
        )
    allowed_path = layout.out_dir / ALLOWED_TRAIN_FILE
    if not allowed_path.is_file():
        raise FileNotFoundError(f"{allowed_path} missing: rebuild the joints before the statistics")
    train_names = [
        line.strip()
        for line in allowed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fit_train_stats(layout.new_joint_vecs, train_names, layout.out_dir, request.joint_count)
