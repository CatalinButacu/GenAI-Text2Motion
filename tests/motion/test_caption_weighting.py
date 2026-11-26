from pathlib import Path

import numpy as np
import pytest

from text2motion.app.config.loading import from_mapping
from text2motion.dataset.contracts import CaptionWeighting, MotionDataConfig
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.motion.contracts import Split
from text2motion.motion.representation import DIM
from text2motion.motion.storage import (
    FEATURES_DIR,
    MEAN_FILE,
    SPLIT_FILES,
    STD_FILE,
    TEXTS_DIR,
    parse_text_file,
)

WALK = "a person walks#a/DET person/NOUN walk/VERB#0.0#0.0"
KICK = "a person kicks#a/DET person/NOUN kick/VERB#0.0#0.0"
RARE = MotionDataConfig(mirror_augment=False, caption_weighting=CaptionWeighting.RARE_VERB)


def write_clips(root: Path, kick_clips: int, split: Split = Split.TRAIN) -> None:
    (root / FEATURES_DIR).mkdir()
    (root / TEXTS_DIR).mkdir()
    np.save(root / MEAN_FILE, np.zeros(DIM, np.float32))
    np.save(root / STD_FILE, np.ones(DIM, np.float32))
    names = [f"C{index}" for index in range(20)]
    for index, name in enumerate(names):
        np.save(root / FEATURES_DIR / f"{name}.npy", np.zeros((60, DIM), np.float32))
        lines = [WALK, KICK] if index < kick_clips else [WALK, WALK]
        (root / TEXTS_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / SPLIT_FILES[split]).write_text("\n".join(names) + "\n", encoding="utf-8")
    if split is Split.TRAIN:
        (root / "train_allowed.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def kick_share(dataset: CaptionedMotionDataset, draws: int) -> float:
    kicks = 0
    for epoch in range(draws):
        dataset.set_epoch(epoch)
        kicks += dataset[0].caption == "a person kicks"
    return kicks / draws


def test_rare_verb_weighting_prefers_the_caption_with_the_rarer_verb(tmp_path):
    write_clips(tmp_path, kick_clips=5)
    dataset = CaptionedMotionDataset(tmp_path, RARE, Split.TRAIN, seed=2026)

    assert dataset.verb_clip_counts == {"walk": 20, "kick": 5}
    weights = dataset.caption_weights(parse_text_file(tmp_path / TEXTS_DIR / "C0.txt"))
    np.testing.assert_allclose(weights, [1 / 3, 2 / 3])
    assert abs(kick_share(dataset, 600) - 2 / 3) < 0.06


def test_uniform_weighting_keeps_the_even_pick(tmp_path):
    write_clips(tmp_path, kick_clips=5)
    dataset = CaptionedMotionDataset(
        tmp_path, MotionDataConfig(mirror_augment=False), Split.TRAIN, seed=2026
    )

    assert dataset.verb_clip_counts is None
    assert abs(kick_share(dataset, 600) - 1 / 2) < 0.06


def test_validation_split_ignores_rare_verb_weighting(tmp_path):
    write_clips(tmp_path, kick_clips=5, split=Split.VALIDATION)
    dataset = CaptionedMotionDataset(tmp_path, RARE, Split.VALIDATION, seed=2026)

    assert dataset.verb_clip_counts is None


def test_rare_verb_weighting_without_a_frequent_verb_stops(tmp_path):
    write_clips(tmp_path, kick_clips=0)
    for path in (tmp_path / TEXTS_DIR).iterdir():
        path.write_text("a person#a/DET person/NOUN#0.0#0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rare_verb caption weighting needs a verb"):
        CaptionedMotionDataset(tmp_path, RARE, Split.TRAIN, seed=2026)


def test_yaml_value_selects_rare_verb_weighting():
    data = from_mapping(MotionDataConfig, {"caption_weighting": "rare_verb"})

    assert data.caption_weighting is CaptionWeighting.RARE_VERB
