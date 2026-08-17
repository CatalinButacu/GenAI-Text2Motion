from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.motion.dataset import (
    FEATURES_DIR,
    SPLIT_FILES,
    Split,
    fit_train_stats,
    read_split_names,
)
from text2motion.motion.model import SMPLX_MODEL_TYPE, Gender
from text2motion.motion.representation import (
    DIM,
    FPS,
    JOINTS,
    build_tgt_offsets,
    default_params,
    mirror_left_chain,
    mirror_left_hand_chain,
    mirror_right_chain,
    mirror_right_hand_chain,
    process_file,
    recover_from_ric,
    t2m_tgt_skel_id,
)


class Stage(StrEnum):
    AMASS = "amass"
    INDEX = "index"
    FEATURE = "feature"
    STATS = "stats"
    ALL = "all"


@dataclass(frozen=True)
class PreparationRequest:
    out_dir: Path
    amass_dir: Path | None = None
    smplx_models: Path | None = None
    index_csv: Path | None = None
    device: str = "cpu"
    fps: int = FPS
    joints_num: int = JOINTS


DATASET_HEAD_TRIM_S = {
    "Eyes_Japan_Dataset": 3.0,
    "MPI_HDM05": 3.0,
    "TotalCapture": 1.0,
    "MPI_Limits": 1.0,
    "Transitions_mocap": 0.5,
}

AMASS_DATASET_RENAME = {
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

SHORT_TO_OFFICIAL = {short: official for official, short in AMASS_DATASET_RENAME.items()}


def dataset_of(source_path: str | Path) -> str:
    parts = [p for p in Path(source_path).parts if p not in (".", "pose_data")]
    if not parts:
        return ""
    return SHORT_TO_OFFICIAL.get(parts[0], parts[0])


def head_trim_frames(source_path: str | Path, fps: int) -> int:
    return int(round(DATASET_HEAD_TRIM_S.get(dataset_of(source_path), 0.0) * fps))


def pose_path_candidates(source_path: str, pose_root: Path) -> tuple[Path | None, list[Path]]:
    parts = [p for p in Path(source_path).parts if p not in (".", "pose_data")]
    if not parts:
        return None, []
    dataset = AMASS_DATASET_RENAME.get(parts[0], parts[0])
    rel = Path(dataset, *parts[1:])
    stem = rel.stem
    base = stem[:-6] if stem.endswith("_poses") else stem
    folder = pose_root / rel.parent

    stems = list(dict.fromkeys([base, base.replace(" ", "_"), stem, stem.replace(" ", "_")]))
    names = [f"{s}{suffix}.npy" for s in stems for suffix in ("_stageii", "_poses", "")]
    return folder, [folder / name for name in dict.fromkeys(names)]


def resolve_pose_path(source_path: str, pose_root: Path) -> Path | None:
    folder, candidates = pose_path_candidates(source_path, pose_root)
    if folder is None:
        return None
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def read_heldout_ids(out_dir: Path) -> set[str]:
    held: set[str] = set()
    for split in (Split.VALIDATION, Split.TEST):
        names = (out_dir / SPLIT_FILES[split.value]).read_text().splitlines()
        held |= {
            n.strip()[1:] if n.strip().startswith("M") else n.strip() for n in names if n.strip()
        }
    return held


def heldout_sources(index_csv: Path, out_dir: Path, pose_root: Path) -> set[Path]:
    held_ids = read_heldout_ids(out_dir)
    import pandas as pd

    index = pd.read_csv(index_csv)

    excluded: set[Path] = set()
    unexcludable: list[str] = []
    for i in range(index.shape[0]):
        if Path(str(index.loc[i]["new_name"])).stem not in held_ids:
            continue
        source_path = str(index.loc[i]["source_path"])
        resolved = resolve_pose_path(source_path, pose_root)
        if resolved is not None:
            excluded.add(resolved.resolve())
            continue
        folder, _ = pose_path_candidates(source_path, pose_root)
        if folder is not None and folder.is_dir():
            unexcludable.append(source_path)

    if unexcludable:
        raise RuntimeError(
            f"leakage guard failed: {len(unexcludable)} held-out clips have their AMASS capture "
            f"folder present under {pose_root} but no filename matched the known spellings, so "
            f"they would NOT be excluded from the unlabeled pretraining corpus. Fix the spelling "
            f"candidates before pretraining. First offenders: {unexcludable[:5]}"
        )
    return excluded


TRANS_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)
EX_FPS = 20
NUM_BETAS = 16
NUM_JOINTS_OUT = 22
_GENDERS = [g.value for g in Gender]


def _resolve_model_root(smplx_dir: Path) -> Path:
    if (smplx_dir / "smplx").is_dir():
        return smplx_dir

    if smplx_dir.name.lower() == "smplx":
        return smplx_dir.parent

    return smplx_dir


class AmassPoseExtractor:
    def __init__(
        self,
        smplx_dir: Path,
        num_betas: int = NUM_BETAS,
        num_joints_out: int = NUM_JOINTS_OUT,
        device: str = "cpu",
        chunk_frames: int = 256,
    ) -> None:
        import smplx

        self.device = torch.device(device)
        self.num_betas = num_betas
        self.num_joints_out = num_joints_out
        self.chunk_frames = chunk_frames
        model_root = _resolve_model_root(Path(smplx_dir))
        self._models: dict[str, object] = {}

        for gender in _GENDERS:
            model_file = model_root / "smplx" / f"SMPLX_{gender.upper()}.npz"
            if not model_file.is_file():
                raise FileNotFoundError(f"SMPL-X model not found: {model_file}")
            self._models[gender] = (
                smplx.create(
                    model_path=str(model_root),
                    model_type=SMPLX_MODEL_TYPE,
                    gender=gender,
                    use_pca=False,
                    flat_hand_mean=True,
                    num_betas=num_betas,
                )
                .to(self.device)
                .eval()
            )

        self._n_expr = int(self._models[Gender.NEUTRAL].num_expression_coeffs)

    def _select_model(self, gender: object):
        g = gender.decode() if isinstance(gender, bytes) else str(gender)
        if g not in self._models:
            g = Gender.FEMALE
        return self._models[g]

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _forward_chunk(
        self,
        bm: object,
        root: np.ndarray,
        body: np.ndarray,
        hand: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        n = end - start
        zeros3 = torch.zeros(n, 3, device=self.device)
        broadcast_betas = np.broadcast_to(betas, (n, betas.shape[0])).copy()
        body_out = bm(
            global_orient=self._to_device(root[start:end]),
            body_pose=self._to_device(body[start:end]),
            left_hand_pose=self._to_device(hand[start:end, :45]),
            right_hand_pose=self._to_device(hand[start:end, 45:]),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=torch.zeros(n, self._n_expr, device=self.device),
            betas=self._to_device(broadcast_betas),
            transl=self._to_device(trans[start:end]),
        )
        return body_out.joints[:, : self.num_joints_out, :].detach().cpu().numpy()

    @torch.no_grad()
    def amass_to_pose(self, src_path: str | Path) -> np.ndarray | None:
        bdata = np.load(src_path, allow_pickle=True)

        if "trans" not in bdata:
            return None

        if "mocap_frame_rate" in bdata:
            fps = float(bdata["mocap_frame_rate"])
        elif "mocap_framerate" in bdata:
            fps = float(bdata["mocap_framerate"])
        else:
            return None

        bm = self._select_model(bdata["gender"])
        down_sample = max(int(fps / EX_FPS), 1)
        sl = slice(None, None, down_sample)

        root = np.asarray(bdata["root_orient"][sl], dtype=np.float32)
        body = np.asarray(bdata["pose_body"][sl], dtype=np.float32)
        hand = np.asarray(bdata["pose_hand"][sl], dtype=np.float32)
        trans = np.asarray(bdata["trans"][sl], dtype=np.float32)
        betas = np.asarray(bdata["betas"][: self.num_betas], dtype=np.float32)
        total = root.shape[0]
        if total == 0:
            return None

        joint_chunks: list[np.ndarray] = []
        for start in range(0, total, self.chunk_frames):
            end = min(start + self.chunk_frames, total)
            chunk_joints = self._forward_chunk(bm, root, body, hand, trans, betas, start, end)
            joint_chunks.append(chunk_joints)

        joints = np.concatenate(joint_chunks, axis=0)
        return np.dot(joints, TRANS_MATRIX)


FEATURE_DIM = DIM
JOINTS_NUM = JOINTS
SPLITS = tuple(Split)


def _parquet_files(src: Path, split: str) -> list[Path]:
    matches = sorted(src.glob(f"**/{split}-*.parquet"))
    if not matches:
        matches = sorted(p for p in src.glob("**/*.parquet") if split in p.name)
    return matches


def write_motions(src: Path, out_dir: Path) -> dict[str, list[str]]:
    import pyarrow.parquet as pq

    vec_dir = out_dir / FEATURES_DIR
    vec_dir.mkdir(parents=True, exist_ok=True)
    split_names: dict[str, list[str]] = {split: [] for split in SPLITS}

    for split in SPLITS:
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
                    if feature.shape[-1] != FEATURE_DIM:
                        raise ValueError(f"{name}: expected {FEATURE_DIM}-dim, got {feature.shape}")
                    np.save(vec_dir / f"{name}.npy", feature)
                    split_names[split].append(name)

    for split, names in split_names.items():
        (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    return split_names


def compute_mean_std(out_dir: Path, train_names: list[str]) -> None:
    fit_train_stats(out_dir / FEATURES_DIR, train_names, out_dir, JOINTS_NUM)


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
    right_chain = mirror_right_chain
    left_chain = mirror_left_chain
    left_hand_chain = mirror_left_hand_chain
    right_hand_chain = mirror_right_hand_chain
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


def stage_amass_to_pose(request: PreparationRequest, layout: RegenLayout) -> None:
    if request.amass_dir is None:
        raise ValueError("amass_dir must be set to the AMASS root for stage 'amass'")
    if request.smplx_models is None:
        raise ValueError("smplx_models must be set (SMPL-X bodies) for stage 'amass'")

    amass_root = Path(request.amass_dir)
    extractor = AmassPoseExtractor(Path(request.smplx_models), device=request.device)

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


def stage_index_to_joints(
    request: PreparationRequest, layout: RegenLayout, pose_root: Path | None = None
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


def stage_joints_to_feature(request: PreparationRequest, layout: RegenLayout) -> None:
    joints_num = request.joints_num
    ref_path = layout.joints / f"{t2m_tgt_skel_id}.npy"
    if not ref_path.is_file():
        available = sorted(layout.joints.glob("[0-9]*.npy"))
        if not available:
            raise FileNotFoundError(f"no joints in {layout.joints}; run stage 'index' first")
        ref_path = available[0]
        print(f"reference skeleton {t2m_tgt_skel_id} missing; using {ref_path.name}")
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
        except Exception as exc:
            print(f"FAILED {source_file}: {exc}")

    print(
        f"Total clips: {len(source_list)}, Frames: {frame_num}, "
        f"Duration: {frame_num / request.fps / 60:.4f}m"
    )


def stage_mean_std(layout: RegenLayout, joints_num: int) -> tuple[np.ndarray, np.ndarray]:
    train_names = read_split_names(layout.out_dir / SPLIT_FILES[Split.TRAIN])
    return fit_train_stats(layout.new_joint_vecs, train_names, layout.out_dir, joints_num)


def regenerate(request: PreparationRequest, stage: Stage | str = Stage.ALL) -> None:
    stage = Stage(stage)
    layout = RegenLayout(Path(request.out_dir))
    layout.mkdirs()

    if stage in (Stage.AMASS, Stage.ALL):
        stage_amass_to_pose(request, layout)
    if stage in (Stage.INDEX, Stage.ALL):
        stage_index_to_joints(request, layout)
    if stage in (Stage.FEATURE, Stage.ALL):
        stage_joints_to_feature(request, layout)
    if stage in (Stage.STATS, Stage.ALL):
        stage_mean_std(layout, request.joints_num)


def resolve_heldout(request: PreparationRequest, pose_root: Path) -> set[Path]:
    if request.index_csv is None:
        raise ValueError("index_csv must be set (the leakage guard needs the official index)")
    return heldout_sources(Path(request.index_csv), Path(request.out_dir), pose_root)


def build_corpus(
    request: PreparationRequest,
    regen_dir: Path,
    out_dir: Path,
    min_frames: int,
    limit: int = 0,
) -> dict[str, float]:
    pose_root = regen_dir / "pose_data"
    joints_dir = regen_dir / "joints"
    if not pose_root.is_dir():
        raise FileNotFoundError(f"{pose_root} missing -- run regenerate.py --stage amass first")

    ref_path = joints_dir / f"{t2m_tgt_skel_id}.npy"
    if not ref_path.is_file():
        candidates = sorted(joints_dir.glob("[0-9]*.npy"))
        if not candidates:
            raise FileNotFoundError(f"no reference joints in {joints_dir}")
        ref_path = candidates[0]
    joints_num = request.joints_num
    reference = np.load(ref_path)[:, :joints_num].reshape(-1, joints_num, 3)
    params = default_params(build_tgt_offsets(reference))

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
        joints = np.load(src)[:, :joints_num]
        trim = head_trim_frames(rel, request.fps)
        if trim:
            joints = joints[trim:]
        if joints.shape[0] < min_frames:
            skipped_short += 1
            continue
        joints = joints.copy()
        joints[..., 0] *= -1
        try:
            features, _, _, _ = process_file(joints, params)
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
