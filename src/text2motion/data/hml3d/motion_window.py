"""Fixed-length motion windows for tokenizer (RVQ/FSQ) training -- motion only, no text.

The motion tokenizer is trained on fixed-length windows randomly cropped from each clip (the T2M-GPT
/ MoMask recipe), which keeps batches rectangular and the conv encoder's downsampling exact. Yields
normalized ``(window, 263)`` tensors; mirror augmentation on the train split only. Shares the
regenerated layout + ``Mean.npy``/``Std.npy`` with the text dataset. See ``.claude/skills/
motion-tokenizer``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.shared.config import DataCfg, Hml3dReprCfg, PathsCfg

_SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}


class MotionWindowDataset(Dataset):
    """Normalized fixed-length motion windows from the regenerated HumanML3D-263."""

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
        self._rng = random.Random(seed)

        out_dir = Path(paths.hml3d_out_dir)
        self._vec_dir = out_dir / "new_joint_vecs"
        self.mean = np.load(out_dir / "Mean.npy").astype(np.float32)
        self.std = np.load(out_dir / "Std.npy").astype(np.float32)
        if self.mean.shape[-1] != self._dim:
            raise ValueError(f"Mean must be {self._dim}-dim, got {self.mean.shape}")

        listed = [n.strip() for n in (out_dir / _SPLIT_FILES[split]).read_text().splitlines() if n.strip()]
        base_names = list(dict.fromkeys(n[1:] if n.startswith("M") else n for n in listed))
        self._ids = self._index_clips(base_names)
        if not self._ids:
            raise RuntimeError(f"no clips >= window {window} for split {split!r}")

    def _index_clips(self, names: list[str]) -> list[str]:
        """Load every clip >= window into memory once (avoids per-step disk reads). Stores the raw
        features in ``self._cache`` parallel to the returned ids; normalization happens at access."""
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

    def __len__(self) -> int:
        return len(self._ids)

    def denormalize(self, feat: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=feat.device, dtype=feat.dtype)
        std = torch.as_tensor(self.std, device=feat.device, dtype=feat.dtype)
        return feat * std + mean

    def __getitem__(self, idx: int) -> torch.Tensor:
        feat = self._cache[idx]
        overflow = feat.shape[0] - self._window
        start = self._rng.randint(0, overflow) if self._train else overflow // 2
        window = (feat[start : start + self._window] - self.mean) / self.std
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
