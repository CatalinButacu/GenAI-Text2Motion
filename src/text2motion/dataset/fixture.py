from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import MappingProxyType

import numpy as np

from text2motion.dataset.splits.view import ALLOWED_TRAIN_FILE
from text2motion.motion.contracts import Split
from text2motion.motion.representation import DIM
from text2motion.motion.storage import (
    FEATURES_DIR,
    MEAN_FILE,
    SPLIT_FILES,
    STD_FILE,
    TEXTS_DIR,
)

FIXTURE_CAPTIONS = (
    "a person walks forward#a/DET person/NOUN walk/VERB forward/ADV#0.0#0.0",
    "a person kicks with the right leg#a/DET person/NOUN kick/VERB with/ADP the/DET right/ADJ leg/NOUN#0.0#0.0",
    "a person waves with the left hand#a/DET person/NOUN wave/VERB with/ADP the/DET left/ADJ hand/NOUN#0.0#0.0",
)
FIXTURE_SPLIT_SIZES = MappingProxyType({Split.TRAIN: 6, Split.VALIDATION: 2, Split.TEST: 2})
FIXTURE_FRAMES = 64
FIXTURE_ID_PREFIX = "F"


def lines(items: Iterable[str]) -> str:
    return "".join(f"{item}\n" for item in items)


def build_fixture_dataset(root: Path, seed: int = 2026) -> dict[Split, list[str]]:
    root = Path(root)
    (root / FEATURES_DIR).mkdir(parents=True)
    (root / TEXTS_DIR).mkdir()
    generator = np.random.default_rng(seed)
    clips: dict[str, np.ndarray] = {}
    splits: dict[Split, list[str]] = {}
    for split, size in FIXTURE_SPLIT_SIZES.items():
        names = [f"{FIXTURE_ID_PREFIX}{split.value[:2]}{index:02d}" for index in range(size)]
        splits[split] = names
        for index, name in enumerate(names):
            steps = generator.standard_normal((FIXTURE_FRAMES, DIM)).astype(np.float32)
            clips[name] = np.cumsum(0.05 * steps, axis=0) + 0.1 * steps
            captions = (FIXTURE_CAPTIONS[index % 3], FIXTURE_CAPTIONS[(index + 1) % 3])
            (root / TEXTS_DIR / f"{name}.txt").write_text(lines(captions), "utf-8")
    for name, clip in clips.items():
        np.save(root / FEATURES_DIR / f"{name}.npy", clip)
    train = np.concatenate([clips[name] for name in splits[Split.TRAIN]])
    np.save(root / MEAN_FILE, train.mean(axis=0).astype(np.float32))
    np.save(root / STD_FILE, train.std(axis=0).astype(np.float32) + 1e-3)
    for split, names in splits.items():
        (root / SPLIT_FILES[split]).write_text(lines(names), encoding="utf-8")
    (root / ALLOWED_TRAIN_FILE).write_text(lines(splits[Split.TRAIN]), "utf-8")
    return splits
