from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.model import SMPLX_MODEL_TYPE, Gender
from text2motion.motion.representation import FPS, JOINTS
from text2motion.motion.storage import sha256_file, split_clip_ids

_HEAD_TRIM_SECONDS = MappingProxyType(
    {
        "Eyes_Japan_Dataset": 3.0,
        "MPI_HDM05": 3.0,
        "TotalCapture": 1.0,
        "MPI_Limits": 1.0,
        "Transitions_mocap": 0.5,
    }
)

_FOLDER_NAMES = MappingProxyType(
    {
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
)

_OFFICIAL_NAMES = MappingProxyType({folder: official for official, folder in _FOLDER_NAMES.items()})
_MODEL_BETAS = 16
_CHUNK_SIZE = 256


def dataset_name(source_path: str | Path) -> str:
    parts = [part for part in Path(source_path).parts if part not in (".", "pose_data")]
    if not parts:
        return ""
    return _OFFICIAL_NAMES.get(parts[0], parts[0])


def head_trim_table() -> dict[str, float]:
    return dict(sorted(_HEAD_TRIM_SECONDS.items()))


def head_trim_frames(source_path: str | Path, fps: int) -> int:
    return int(round(_HEAD_TRIM_SECONDS.get(dataset_name(source_path), 0.0) * fps))


def pose_folders_and_candidates(source_path: str, pose_root: Path) -> tuple[list[Path], list[Path]]:
    parts = [part for part in Path(source_path).parts if part not in (".", "pose_data")]
    if not parts:
        return [], []
    datasets = list(dict.fromkeys([_FOLDER_NAMES.get(parts[0], parts[0]), parts[0]]))
    stem = Path(parts[-1]).stem
    base = stem[:-6] if stem.endswith("_poses") else stem
    stems = list(dict.fromkeys([base, base.replace(" ", "_"), stem, stem.replace(" ", "_")]))
    names = list(
        dict.fromkeys(
            f"{candidate}{suffix}.npy"
            for candidate in stems
            for suffix in ("_stageii", "_poses", "")
        )
    )
    folders = [pose_root / Path(dataset, *parts[1:-1]) for dataset in datasets]
    return folders, [folder / name for folder in folders for name in names]


def resolve_pose_path(source_path: str, pose_root: Path) -> Path | None:
    _, candidates = pose_folders_and_candidates(source_path, pose_root)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def read_heldout_ids(out_dir: Path) -> set[str]:
    held: set[str] = set()
    for split in (Split.VALIDATION, Split.TEST):
        held.update(split_clip_ids(out_dir, split))
    return held


def heldout_sources(index_csv: Path, out_dir: Path, pose_root: Path) -> set[Path]:
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
        folders, candidates = pose_folders_and_candidates(source_path, pose_root)
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


def resolve_gender(value: object) -> Gender:
    raw = value.item() if isinstance(value, np.ndarray) else value
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    try:
        return Gender(text.strip().lower())
    except ValueError as error:
        raise ValueError(
            f"unknown AMASS gender {value!r}; expected one of {list(Gender)}"
        ) from error


def resolve_model_root(smplx_dir: Path) -> Path:
    if (smplx_dir / "smplx").is_dir():
        return smplx_dir
    if smplx_dir.name.lower() == "smplx":
        return smplx_dir.parent
    return smplx_dir


############################################################
#       AmassPoseExtractor
############################################################
class AmassPoseExtractor:
    def __init__(
        self,
        smplx_dir: Path,
        beta_count: int = _MODEL_BETAS,
        joint_count: int = JOINTS,
        device: str = "cpu",
        fps: int = FPS,
        chunk_size: int = _CHUNK_SIZE,
    ) -> None:
        import smplx

        self.device = torch.device(device)
        self._beta_count = beta_count
        self._joint_count = joint_count
        self._fps = fps
        self._chunk_size = chunk_size
        model_root = resolve_model_root(Path(smplx_dir))
        self._models: dict[str, Any] = {}
        self._model_files: dict[str, Path] = {}

        for gender in Gender:
            model_file = model_root / "smplx" / f"SMPLX_{gender.value.upper()}.npz"
            if not model_file.is_file():
                raise FileNotFoundError(f"SMPL-X model not found: {model_file}")
            self._model_files[gender.value] = model_file
            self._models[gender.value] = (
                smplx.create(
                    model_path=str(model_root),
                    model_type=SMPLX_MODEL_TYPE,
                    gender=gender.value,
                    use_pca=False,
                    flat_hand_mean=True,
                    num_betas=beta_count,
                )
                .to(self.device)
                .eval()
            )

        self._expression_count = int(self._models[Gender.NEUTRAL.value].num_expression_coeffs)

    def recipe(self) -> dict[str, object]:
        return {
            "fps": self._fps,
            "joint_count": self._joint_count,
            "beta_count": self._beta_count,
            "flat_hand_mean": True,
            "axes": "y_up",
            "smplx_sha256": {
                gender: sha256_file(path) for gender, path in sorted(self._model_files.items())
            },
        }

    def __model(self, gender: object):
        return self._models[resolve_gender(gender).value]

    def to_float_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def __forward(
        self,
        model: Any,
        root: torch.Tensor,
        body: torch.Tensor,
        hand: torch.Tensor,
        trans: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        frames = root.shape[0]
        zeros3 = torch.zeros(frames, 3, dtype=root.dtype, device=self.device)
        output = model(
            global_orient=root,
            body_pose=body,
            left_hand_pose=hand[:, :45],
            right_hand_pose=hand[:, 45:],
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=torch.zeros(
                frames, self._expression_count, dtype=root.dtype, device=self.device
            ),
            betas=betas.expand(frames, -1),
            transl=trans,
        )
        return output.joints[:, : self._joint_count, :]

    def __stride(self, source_fps: float, src_path: str | Path) -> int:
        ratio = source_fps / self._fps
        stride = round(ratio)
        if stride < 1 or abs(ratio - stride) > 1e-6:
            raise ValueError(
                f"{src_path}: {source_fps} fps is not an integer multiple of {self._fps} fps; "
                "a frame stride would change the motion speed"
            )
        return stride

    @torch.inference_mode()
    def extract(self, src_path: str | Path) -> np.ndarray | None:
        with np.load(src_path, allow_pickle=True) as data:
            if "trans" not in data:
                return None
            fps_key = next(
                (key for key in ("mocap_frame_rate", "mocap_framerate") if key in data),
                None,
            )
            if fps_key is None:
                raise ValueError(f"{src_path} has motion but no frame-rate field")
            sample = slice(None, None, self.__stride(float(data[fps_key]), src_path))
            root = np.asarray(data["root_orient"][sample], dtype=np.float32)
            if root.shape[0] == 0:
                return None
            body = np.asarray(data["pose_body"][sample], dtype=np.float32)
            hand = np.asarray(data["pose_hand"][sample], dtype=np.float32)
            trans = np.asarray(data["trans"][sample], dtype=np.float32)
            if np.asarray(data["betas"]).shape[0] < self._beta_count:
                raise ValueError(
                    f"{src_path}: {np.asarray(data['betas']).shape[0]} betas, the model needs "
                    f"{self._beta_count}"
                )
            betas = np.asarray(data["betas"][: self._beta_count], dtype=np.float32)
            model = self.__model(data["gender"])

        root_device = self.to_float_tensor(root)
        body_device = self.to_float_tensor(body)
        hand_device = self.to_float_tensor(hand)
        trans_device = self.to_float_tensor(trans)
        betas_device = self.to_float_tensor(betas)

        chunks: list[torch.Tensor] = []
        for start in range(0, root.shape[0], self._chunk_size):
            end = min(start + self._chunk_size, root.shape[0])
            chunks.append(
                self.__forward(
                    model,
                    root_device[start:end],
                    body_device[start:end],
                    hand_device[start:end],
                    trans_device[start:end],
                    betas_device,
                )
            )
        joints = torch.cat(chunks, dim=0)
        y_up_joints = torch.stack((joints[..., 0], joints[..., 2], joints[..., 1]), dim=-1)
        if not torch.isfinite(y_up_joints).all():
            raise ValueError(f"{src_path}: the extracted joints hold NaN or Inf values")
        return y_up_joints.cpu().numpy()
