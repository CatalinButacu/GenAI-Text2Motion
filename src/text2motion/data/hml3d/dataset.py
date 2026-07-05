from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.shared.config import DataCfg, Hml3dReprCfg, PathsCfg

_SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}


@dataclass(frozen=True)
class TextAnnotation:
    caption: str
    tokens: list[str]
    start_time: float
    end_time: float


def parse_text_file(path: Path) -> list[TextAnnotation]:
    annotations: list[TextAnnotation] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            raise ValueError(f"malformed caption line in {path}: {raw_line!r}")
        caption, tokens, start, end = parts[0], parts[1], parts[2], parts[3]
        start_time = float(start)
        end_time = float(end)
        annotations.append(
            TextAnnotation(
                caption=caption,
                tokens=tokens.split(" ") if tokens else [],
                start_time=0.0 if np.isnan(start_time) else start_time,  # official: NaN tag -> 0.0
                end_time=0.0 if np.isnan(end_time) else end_time,
            )
        )
    if not annotations:
        raise ValueError(f"no captions found in {path}")
    return annotations


class Hml3dMotionTextDataset(Dataset):
    def __init__(
        self,
        paths: PathsCfg,
        repr_cfg: Hml3dReprCfg,
        data_cfg: DataCfg,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {list(_SPLIT_FILES)}, got {split!r}")
        if paths.hml3d_out_dir is None:
            raise ValueError("paths.hml3d_out_dir must be set (regenerated output root)")

        self._dim = repr_cfg.dim
        self._fps = repr_cfg.fps
        self._min_len = data_cfg.min_motion_len
        self._max_len = data_cfg.max_motion_len
        self._train = split == "train"
        self._mirror = data_cfg.mirror_augment and self._train
        self._rng = random.Random(seed)

        out_dir = Path(paths.hml3d_out_dir)
        self._vec_dir = out_dir / "new_joint_vecs"
        self._text_dir = Path(paths.texts_dir) if paths.texts_dir is not None else out_dir / "texts"

        self.mean, self.std = self._load_stats(out_dir)

        split_path = out_dir / _SPLIT_FILES[split]
        if not split_path.is_file():
            raise FileNotFoundError(
                f"split file not found: {split_path} (copy the official {split} list there)"
            )
        listed = [n.strip() for n in split_path.read_text().splitlines() if n.strip()]
        base_names = list(dict.fromkeys(n[1:] if n.startswith("M") else n for n in listed))

        self._ids = self._index_clips(base_names)
        if not self._ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")

    def _load_stats(self, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        mean_path, std_path = out_dir / "Mean.npy", out_dir / "Std.npy"
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Mean.npy/Std.npy missing under {out_dir}; run regenerate.py --stage stats first"
            )
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        if mean.shape[-1] != self._dim or std.shape[-1] != self._dim:
            raise ValueError(f"Mean/Std must be {self._dim}-dim, got {mean.shape} / {std.shape}")
        return mean, std

    def _index_clips(self, names: list[str]) -> list[str]:
        kept: list[str] = []
        for name in names:
            for clip_id in self._variants(name):
                vec_path = self._vec_dir / f"{clip_id}.npy"
                text_path = self._text_dir / f"{clip_id}.txt"
                if not vec_path.is_file() or not text_path.is_file():
                    continue
                length = int(np.load(vec_path, mmap_mode="r").shape[0])
                if length >= self._min_len:
                    kept.append(clip_id)
        return kept

    def _variants(self, name: str) -> list[str]:
        return [name, f"M{name}"] if self._mirror else [name]

    def __len__(self) -> int:
        return len(self._ids)

    def normalize(self, feat: np.ndarray) -> np.ndarray:
        return (feat - self.mean) / self.std

    def denormalize(self, feat: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(feat, torch.Tensor):
            mean = torch.as_tensor(self.mean, device=feat.device, dtype=feat.dtype)
            std = torch.as_tensor(self.std, device=feat.device, dtype=feat.dtype)
            return feat * std + mean
        return feat * self.std + self.mean

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        clip_id = self._ids[idx]
        feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
        if feat.shape[-1] != self._dim:
            raise ValueError(f"{clip_id}: expected {self._dim}-dim feature, got {feat.shape}")
        if not np.isfinite(feat).all():  # NaN/Inf would poison gradients (data audit) -- fail loud
            raise ValueError(
                f"{clip_id}: non-finite values in feature; drop it from the split list"
            )

        annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
        ann, feat = self._pick_caption_segment(annotations, feat)

        feat = self._fit_to_max_len(feat)
        feat = self.normalize(feat)
        return torch.from_numpy(feat), feat.shape[0], ann.caption

    def _segment(self, ann: TextAnnotation, total: int) -> tuple[int, int] | None:
        if ann.start_time == 0.0 and ann.end_time == 0.0:
            return None
        start = max(int(ann.start_time * self._fps), 0)
        end = min(int(ann.end_time * self._fps), total)
        return start, end

    def _pick_caption_segment(
        self, annotations: list[TextAnnotation], feat: np.ndarray
    ) -> tuple[TextAnnotation, np.ndarray]:
        usable = []
        for ann in annotations:
            segment = self._segment(ann, feat.shape[0])
            if segment is None or segment[1] - segment[0] >= self._min_len:
                usable.append((ann, segment))
        if not usable:  # every caption covers a sub-min segment; full clip is the least-bad pairing
            return self._rng.choice(annotations), feat
        ann, segment = self._rng.choice(usable)
        if segment is not None:
            feat = feat[segment[0] : segment[1]]
        return ann, feat

    def _fit_to_max_len(self, feat: np.ndarray) -> np.ndarray:
        overflow = feat.shape[0] - self._max_len
        if overflow <= 0:
            return feat
        start = self._rng.randint(0, overflow) if self._train else 0
        return feat[start : start + self._max_len]


def collate_pad(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    motions, lengths, captions = zip(*batch, strict=True)
    max_len = max(motion.shape[0] for motion in motions)
    feature_dim = motions[0].shape[1]
    padded = motions[0].new_zeros(len(motions), max_len, feature_dim)
    for row, motion in enumerate(motions):
        padded[row, : motion.shape[0]] = motion
    return padded, torch.tensor(lengths, dtype=torch.long), list(captions)


def build_dataloader(
    paths: PathsCfg,
    repr_cfg: Hml3dReprCfg,
    data_cfg: DataCfg,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = Hml3dMotionTextDataset(paths, repr_cfg, data_cfg, split=split, seed=seed)
    use_shuffle = (split == "train") if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        num_workers=num_workers,
        collate_fn=collate_pad,
        drop_last=(split == "train"),
    )
