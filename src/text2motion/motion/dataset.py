from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.motion.model import MotionBatch, MotionClip
from text2motion.motion.representation import DIM, FPS, Source, Track, normalization_stats

FEATURES_DIR = "new_joint_vecs"
TEXTS_DIR = "texts"
MEAN_FILE = "Mean.npy"
STD_FILE = "Std.npy"
STATS_PROVENANCE_FILE = "stats_provenance.json"
CLIP_LENGTHS_FILE = "clip_lengths.json"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


SPLIT_FILES = {split: f"{split.value}.txt" for split in Split}


@dataclass(frozen=True)
class DataConfig:
    track: Track = Track.HML3D_263
    sources: tuple[Source, ...] = (Source.HUMANML3D,)
    mirror_augment: bool = True
    max_motion_len: int = 196
    min_motion_len: int = 40


def item_rng(seed: int, epoch: int, index: int) -> np.random.RandomState:
    item_seed = np.random.SeedSequence((seed, epoch, index)).generate_state(1)[0]
    return np.random.RandomState(item_seed)


def _read_length_cache(cache_path: Path) -> dict[str, list[int]]:
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in raw.items()}


def clip_lengths(vec_dir: Path, clip_ids: list[str]) -> dict[str, int]:
    cache_path = vec_dir.parent / CLIP_LENGTHS_FILE
    cache = _read_length_cache(cache_path)

    lengths: dict[str, int] = {}
    dirty = False
    for clip_id in clip_ids:
        path = vec_dir / f"{clip_id}.npy"
        if not path.is_file():
            continue
        size = path.stat().st_size
        entry = cache.get(clip_id)
        if entry is not None and entry[1] == size:
            lengths[clip_id] = entry[0]
            continue
        length = int(np.load(path, mmap_mode="r").shape[0])
        lengths[clip_id] = length
        cache[clip_id] = [length, size]
        dirty = True

    if dirty:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return lengths


@dataclass(frozen=True)
class MotionScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, out_dir: Path, dim: int | None = None) -> MotionScaler:
        mean_path = out_dir / MEAN_FILE
        std_path = out_dir / STD_FILE
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Mean.npy/Std.npy missing under {out_dir}. For the standard 263 track these ship "
                f"with the official HumanML3D conversion; regenerating them changes the scaler "
                f"every trained checkpoint depends on."
            )
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        if mean.shape != std.shape:
            raise ValueError(f"Mean/Std shape mismatch: {mean.shape} vs {std.shape}")
        if dim is not None and mean.shape[-1] != dim:
            raise ValueError(f"Mean/Std must be {dim}-dim, got {mean.shape}")
        return cls(mean=mean, std=std)

    def normalize(self, feats):
        if isinstance(feats, torch.Tensor):
            return (feats - self._as_tensor(self.mean, feats)) / self._as_tensor(self.std, feats)
        return (feats - self.mean) / self.std

    def denormalize(self, feats):
        if isinstance(feats, torch.Tensor):
            return feats * self._as_tensor(self.std, feats) + self._as_tensor(self.mean, feats)
        return feats * self.std + self.mean

    @staticmethod
    def _as_tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(array, device=like.device, dtype=like.dtype)


def read_split_names(split_path: Path) -> list[str]:
    if not split_path.is_file():
        raise FileNotFoundError(
            f"{split_path} missing: normalization statistics must be fitted on the train split "
            f"alone. Fitting over every available feature file would include val and test, "
            f"leaking held-out data into the scaler used by every downstream model."
        )
    return [name.strip() for name in split_path.read_text().splitlines() if name.strip()]


def load_clips(vec_dir: Path, names: list[str]) -> list[np.ndarray]:
    clips = []
    for name in sorted(set(names)):
        path = vec_dir / f"{name}.npy"
        if not path.is_file():
            continue
        array = np.load(path)
        if np.isnan(array).any():
            print(f"NaN feature, skipping: {name}")
            continue
        clips.append(array)
    return clips


def fit_train_stats(
    vec_dir: Path, train_names: list[str], out_dir: Path, joints_num: int
) -> tuple[np.ndarray, np.ndarray]:
    clips = load_clips(vec_dir, train_names)
    mean, std = normalization_stats(clips, joints_num)

    mean_path = out_dir / MEAN_FILE
    provenance = {
        "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(vec_dir),
        "split": Split.TRAIN.value,
        "requested_clips": len(set(train_names)),
        "used_clips": len(clips),
        "frames": int(sum(clip.shape[0] for clip in clips)),
        "joints_num": joints_num,
        "overwrote_existing": mean_path.is_file(),
    }

    np.save(mean_path, mean)
    np.save(out_dir / STD_FILE, std)
    provenance_path = out_dir / STATS_PROVENANCE_FILE
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Mean/Std saved from {len(clips)} train clips -> {provenance_path}")
    return mean, std


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
                start_time=0.0 if np.isnan(start_time) else start_time,
                end_time=0.0 if np.isnan(end_time) else end_time,
            )
        )
    if not annotations:
        raise ValueError(f"no captions found in {path}")
    return annotations


class MotionTextDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data: DataConfig,
        split: Split | str = Split.TRAIN,
        texts_dir: Path | None = None,
        seed: int = 42,
    ) -> None:
        split = Split(split)

        self._min_len = data.min_motion_len
        self._max_len = data.max_motion_len
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train
        self._seed = seed
        self._epoch = 0

        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self._text_dir = Path(texts_dir) if texts_dir is not None else root / TEXTS_DIR

        self.scaler = MotionScaler.load(root, dim=DIM)

        split_path = root / SPLIT_FILES[split]
        if not split_path.is_file():
            raise FileNotFoundError(
                f"split file not found: {split_path} (copy the official {split} list there)"
            )
        listed = [n.strip() for n in split_path.read_text().splitlines() if n.strip()]
        base_names = list(dict.fromkeys(n[1:] if n.startswith("M") else n for n in listed))

        self._ids = self._index_clips(base_names)
        if not self._ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")

    @property
    def clip_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def std(self) -> np.ndarray:
        return self.scaler.std

    def _index_clips(self, names: list[str]) -> list[str]:
        candidates = [clip_id for name in names for clip_id in self._variants(name)]
        lengths = clip_lengths(self._vec_dir, candidates)

        kept: list[str] = []
        for name in names:
            for clip_id in self._variants(name):
                text_path = self._text_dir / f"{clip_id}.txt"
                length = lengths.get(clip_id)
                if length is None or not text_path.is_file():
                    continue
                if length >= self._min_len:
                    kept.append(clip_id)
        return kept

    def _variants(self, name: str) -> list[str]:
        return [name, f"M{name}"] if self._mirror else [name]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._ids)

    def normalize(self, feat):
        return self.scaler.normalize(feat)

    def denormalize(self, feat):
        return self.scaler.denormalize(feat)

    def __getitem__(self, idx: int) -> MotionClip:
        clip_id = self._ids[idx]
        feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
        if feat.shape[-1] != DIM:
            raise ValueError(f"{clip_id}: expected {DIM}-dim feature, got {feat.shape}")
        if not np.isfinite(feat).all():
            raise ValueError(
                f"{clip_id}: non-finite values in feature; drop it from the split list"
            )

        rng = item_rng(self._seed, self._epoch, idx)
        annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
        annotation, feat = self._pick_caption_segment(annotations, feat, rng)

        feat = self._fit_to_max_len(feat, rng)
        feat = self.normalize(feat)
        return MotionClip(
            features=torch.from_numpy(feat),
            frame_count=feat.shape[0],
            caption=annotation.caption,
            clip_id=clip_id,
        )

    def _segment(self, annotation: TextAnnotation, total: int) -> tuple[int, int] | None:
        if annotation.start_time == 0.0 and annotation.end_time == 0.0:
            return None
        start = max(int(annotation.start_time * FPS), 0)
        end = min(int(annotation.end_time * FPS), total)
        return start, end

    def _pick_caption_segment(
        self, annotations: list[TextAnnotation], feat: np.ndarray, rng: np.random.RandomState
    ) -> tuple[TextAnnotation, np.ndarray]:
        usable = []
        for annotation in annotations:
            segment = self._segment(annotation, feat.shape[0])
            if segment is None or segment[1] - segment[0] >= self._min_len:
                usable.append((annotation, segment))
        if not usable:
            return annotations[rng.randint(len(annotations))], feat
        annotation, segment = usable[rng.randint(len(usable))]
        if segment is not None:
            feat = feat[segment[0] : segment[1]]
        return annotation, feat

    def _fit_to_max_len(self, feat: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        overflow = feat.shape[0] - self._max_len
        if overflow <= 0:
            return feat
        start = int(rng.randint(overflow + 1)) if self._train else 0
        return feat[start : start + self._max_len]


class MotionWindowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data: DataConfig,
        split: Split | str,
        window: int,
        seed: int = 42,
    ) -> None:
        split = Split(split)

        self._window = window
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train
        self._seed = seed
        self._epoch = 0

        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self.scaler = MotionScaler.load(root, dim=DIM)

        listed = [
            n.strip() for n in (root / SPLIT_FILES[split]).read_text().splitlines() if n.strip()
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
                    if not np.isfinite(feat).all():
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

    def __getitem__(self, idx: int) -> MotionClip:
        feat = self._cache[idx]
        overflow = feat.shape[0] - self._window
        if self._train:
            start = int(item_rng(self._seed, self._epoch, idx).randint(overflow + 1))
        else:
            start = overflow // 2
        window = self.scaler.normalize(feat[start : start + self._window])
        return MotionClip(
            features=torch.from_numpy(window),
            frame_count=window.shape[0],
            clip_id=self._ids[idx],
        )


def collate_clips(clips: list[MotionClip]) -> MotionBatch:
    max_len = max(clip.frame_count for clip in clips)
    feature_dim = clips[0].features.shape[1]
    padded = clips[0].features.new_zeros(len(clips), max_len, feature_dim)
    for row, clip in enumerate(clips):
        padded[row, : clip.frame_count] = clip.features
    return MotionBatch(
        features=padded,
        lengths=torch.tensor([clip.frame_count for clip in clips], dtype=torch.long),
        captions=[clip.caption or "" for clip in clips],
    )


class MotionRepository:
    def __init__(
        self,
        root: Path,
        data: DataConfig | None = None,
        texts_dir: Path | None = None,
        seed: int = 42,
    ) -> None:
        if root is None:
            raise ValueError("motion repository root must be set (regenerated 263 output dir)")
        self.root = Path(root)
        self.data = data or DataConfig()
        self.texts_dir = Path(texts_dir) if texts_dir is not None else None
        self.seed = seed

    def scaler(self) -> MotionScaler:
        return MotionScaler.load(self.root, dim=DIM)

    def split(self, split: Split | str) -> MotionTextDataset:
        return MotionTextDataset(
            self.root, self.data, split=split, texts_dir=self.texts_dir, seed=self.seed
        )

    def windows(self, split: Split | str, window: int) -> MotionWindowDataset:
        return MotionWindowDataset(self.root, self.data, split, window, seed=self.seed)

    def loader(
        self,
        split: Split | str,
        batch_size: int,
        shuffle: bool | None = None,
        num_workers: int = 0,
    ) -> DataLoader:
        split = Split(split)
        return DataLoader(
            self.split(split),
            batch_size=batch_size,
            shuffle=(split == Split.TRAIN) if shuffle is None else shuffle,
            num_workers=num_workers,
            collate_fn=collate_clips,
            drop_last=(split == Split.TRAIN),
        )

    def window_loader(
        self,
        split: Split | str,
        window: int,
        batch_size: int,
        num_workers: int = 0,
    ) -> DataLoader:
        split = Split(split)
        return DataLoader(
            self.windows(split, window),
            batch_size=batch_size,
            shuffle=(split == Split.TRAIN),
            num_workers=num_workers,
            collate_fn=collate_clips,
            drop_last=(split == Split.TRAIN),
        )
