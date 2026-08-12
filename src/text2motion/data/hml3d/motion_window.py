from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.shared.config import DataCfg, Hml3dReprCfg, PathsCfg
from text2motion.shared.seed import item_rng

from .stats import MotionScaler

_SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}


class MotionWindowDataset(Dataset):
    def __init__(
        self,
        paths: PathsCfg,
        repr_cfg: Hml3dReprCfg,
        data_cfg: DataCfg,
        split: str,
        window: int,
        seed: int = 42,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {list(_SPLIT_FILES)}, got {split!r}")
        if paths.hml3d_out_dir is None:
            raise ValueError("paths.hml3d_out_dir must be set (regenerated output root)")

        self._dim = repr_cfg.dim
        self._window = window
        self._train = split == "train"
        self._mirror = data_cfg.mirror_augment and self._train
        self._seed = seed
        self._epoch = 0

        out_dir = Path(paths.hml3d_out_dir)
        self._vec_dir = out_dir / "new_joint_vecs"
        self.scaler = MotionScaler.load(out_dir, dim=self._dim)

        listed = [
            n.strip() for n in (out_dir / _SPLIT_FILES[split]).read_text().splitlines() if n.strip()
        ]
        base_names = list(dict.fromkeys(n[1:] if n.startswith("M") else n for n in listed))
        self._ids = self._index_clips(base_names)
        if not self._ids:
            raise RuntimeError(f"no clips >= window {window} for split {split!r}")

    def _index_clips(self, names: list[str]) -> list[str]:
        variants = (lambda n: [n, f"M{n}"]) if self._mirror else (lambda n: [n])
        self._cache: list[np.ndarray] = []
        kept: list[str] = []
        non_finite: list[str] = []
        for name in names:
            for clip_id in variants(name):
                vec_path = self._vec_dir / f"{clip_id}.npy"
                if not vec_path.is_file():
                    continue
                feat = np.load(vec_path).astype(np.float32)
                if feat.shape[0] >= self._window:
                    if not np.isfinite(feat).all():  # NaN/Inf clips poison gradients (data audit)
                        non_finite.append(clip_id)
                        continue
                    kept.append(clip_id)
                    self._cache.append(feat)
        if non_finite:
            raise ValueError(
                f"{len(non_finite)} non-finite clips in this split; drop them from the split list: "
                f"{non_finite[:20]}{' ...' if len(non_finite) > 20 else ''}"
            )
        return kept

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def std(self) -> np.ndarray:
        return self.scaler.std

    def denormalize(self, feat: torch.Tensor) -> torch.Tensor:
        return self.scaler.denormalize(feat)

    def __getitem__(self, idx: int) -> torch.Tensor:
        feat = self._cache[idx]
        overflow = feat.shape[0] - self._window
        if self._train:
            start = item_rng(self._seed, self._epoch, idx).randint(0, overflow)
        else:
            start = overflow // 2
        window = self.scaler.normalize(feat[start : start + self._window])
        return torch.from_numpy(window)


def build_window_loader(
    paths: PathsCfg,
    repr_cfg: Hml3dReprCfg,
    data_cfg: DataCfg,
    split: str,
    window: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = MotionWindowDataset(paths, repr_cfg, data_cfg, split, window, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train"),
    )
