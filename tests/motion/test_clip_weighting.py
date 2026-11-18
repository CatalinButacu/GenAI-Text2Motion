from pathlib import Path

import numpy as np
import pytest

from text2motion.app.config import from_mapping
from text2motion.dataset.contracts import CaptionWeighting, ClipWeighting, MotionDataConfig
from text2motion.dataset.datasets import (
    MOTION_CLASS_COLUMN,
    CaptionedMotionDataset,
    MotionDataModule,
)
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
CLIPS = 20
KICK_CLIPS = 5


def write_clips(root: Path, split: Split = Split.TRAIN) -> Path:
    (root / FEATURES_DIR).mkdir()
    (root / TEXTS_DIR).mkdir()
    np.save(root / MEAN_FILE, np.zeros(DIM, np.float32))
    np.save(root / STD_FILE, np.ones(DIM, np.float32))
    names = [f"C{index}" for index in range(CLIPS)]
    for index, name in enumerate(names):
        np.save(root / FEATURES_DIR / f"{name}.npy", np.zeros((60, DIM), np.float32))
        lines = [WALK, KICK] if index < KICK_CLIPS else [WALK, WALK]
        (root / TEXTS_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / SPLIT_FILES[split]).write_text("\n".join(names) + "\n", encoding="utf-8")
    classes = root / "classes.csv"
    classes.write_text(f"verb,{MOTION_CLASS_COLUMN}\nwalk,0\nkick,1\n", encoding="utf-8")
    return classes


def weighted_config(classes: Path) -> MotionDataConfig:
    return MotionDataConfig(
        mirror_augment=False, clip_weighting=ClipWeighting.MOTION_CLASS, motion_classes=classes
    )


def test_a_clip_weight_follows_its_rarest_motion_class(tmp_path):
    dataset = CaptionedMotionDataset(tmp_path, weighted_config(write_clips(tmp_path)), seed=2026)

    weights = dataset.clip_weights
    assert weights is not None
    np.testing.assert_allclose(weights[:KICK_CLIPS], KICK_CLIPS**-0.5)
    np.testing.assert_allclose(weights[KICK_CLIPS:], CLIPS**-0.5)


def test_the_train_loader_draws_the_rare_class_more_often(tmp_path):
    module = MotionDataModule(tmp_path, weighted_config(write_clips(tmp_path)), seed=2026)
    sampler = module.captioned_batch_loader(Split.TRAIN, 4).sampler

    draws = [index for _ in range(300) for index in sampler]
    kick_share = np.mean([index < KICK_CLIPS for index in draws])
    rare, common = KICK_CLIPS * KICK_CLIPS**-0.5, (CLIPS - KICK_CLIPS) * CLIPS**-0.5
    assert abs(kick_share - rare / (rare + common)) < 0.03
    assert len(list(sampler)) == CLIPS


def test_an_unshuffled_train_loader_keeps_every_clip_once(tmp_path):
    module = MotionDataModule(tmp_path, weighted_config(write_clips(tmp_path)), seed=2026)
    loader = module.captioned_batch_loader(Split.TRAIN, 4, shuffle=False)

    assert list(loader.sampler) == list(range(CLIPS))


def test_validation_ignores_motion_class_weighting(tmp_path):
    classes = write_clips(tmp_path, split=Split.VALIDATION)
    dataset = CaptionedMotionDataset(
        tmp_path, weighted_config(classes), Split.VALIDATION, seed=2026
    )

    assert dataset.clip_weights is None


def test_uniform_weighting_has_no_clip_weights(tmp_path):
    write_clips(tmp_path)
    dataset = CaptionedMotionDataset(tmp_path, MotionDataConfig(mirror_augment=False), seed=2026)

    assert dataset.clip_weights is None


def test_a_clip_without_a_class_verb_gets_the_largest_class_weight(tmp_path):
    classes = write_clips(tmp_path)
    (tmp_path / TEXTS_DIR / "C19.txt").write_text(
        "a person#a/DET person/NOUN#0.0#0.0\n", encoding="utf-8"
    )
    dataset = CaptionedMotionDataset(tmp_path, weighted_config(classes), seed=2026)

    weights = dataset.clip_weights
    assert weights is not None
    np.testing.assert_allclose(weights[-1], (CLIPS - 1) ** -0.5)


def test_uniform_replacement_draws_every_clip_alike_with_replacement(tmp_path):
    write_clips(tmp_path)
    config = MotionDataConfig(
        mirror_augment=False, clip_weighting=ClipWeighting.UNIFORM_REPLACEMENT
    )
    module = MotionDataModule(tmp_path, config, seed=2026)
    sampler = module.captioned_batch_loader(Split.TRAIN, 4).sampler

    draws = [index for _ in range(300) for index in sampler]
    kick_share = np.mean([index < KICK_CLIPS for index in draws])
    assert abs(kick_share - KICK_CLIPS / CLIPS) < 0.03
    assert len(list(sampler)) == CLIPS
    assert len(set(sampler)) < CLIPS


def test_motion_class_weighting_without_a_class_verb_stops(tmp_path):
    classes = write_clips(tmp_path)
    classes.write_text(f"verb,{MOTION_CLASS_COLUMN}\njump,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no train clip has a verb"):
        CaptionedMotionDataset(tmp_path, weighted_config(classes), seed=2026)


def test_motion_class_weighting_without_a_verb_map_stops():
    with pytest.raises(ValueError, match="needs the motion_classes verb map"):
        MotionDataConfig(clip_weighting=ClipWeighting.MOTION_CLASS)


def test_yaml_values_select_motion_class_weighting():
    data = from_mapping(
        MotionDataConfig,
        {"clip_weighting": "motion_class", "motion_classes": "doc/paper/pics/verb_cluster_map.csv"},
    )

    assert data.clip_weighting is ClipWeighting.MOTION_CLASS
    assert Path(data.motion_classes or "") == Path("doc/paper/pics/verb_cluster_map.csv")


WALK_KICK = "a person walks and kicks#a/DET person/NOUN walk/VERB and/CCONJ kick/VERB#0.0#0.0"
WAIT = "a person waits#a/DET person/NOUN wait/VERB#0.0#0.0"
WALK_ONLY_CLIPS = 30
MIXED_CLIPS = 20
WALK_GROUP = WALK_ONLY_CLIPS + MIXED_CLIPS
KICK_GROUP = MIXED_CLIPS
VERB_CONFIG = MotionDataConfig(mirror_augment=False, clip_weighting=ClipWeighting.VERB)


def write_verb_clips(root: Path) -> None:
    (root / FEATURES_DIR).mkdir()
    (root / TEXTS_DIR).mkdir()
    np.save(root / MEAN_FILE, np.zeros(DIM, np.float32))
    np.save(root / STD_FILE, np.ones(DIM, np.float32))
    captions = [[WALK]] * WALK_ONLY_CLIPS + [[WALK, WALK_KICK]] * MIXED_CLIPS + [[WAIT]]
    names = [f"V{index}" for index in range(len(captions))]
    for name, lines in zip(names, captions, strict=True):
        np.save(root / FEATURES_DIR / f"{name}.npy", np.zeros((60, DIM), np.float32))
        (root / TEXTS_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / SPLIT_FILES[Split.TRAIN]).write_text("\n".join(names) + "\n", encoding="utf-8")


def test_each_verb_group_gets_the_same_total_weight(tmp_path):
    write_verb_clips(tmp_path)
    weights = CaptionedMotionDataset(tmp_path, VERB_CONFIG, seed=2026).clip_weights

    assert weights is not None
    np.testing.assert_allclose(weights[:WALK_ONLY_CLIPS], 1.0 / WALK_GROUP)
    mixed = weights[WALK_ONLY_CLIPS : WALK_ONLY_CLIPS + MIXED_CLIPS]
    np.testing.assert_allclose(mixed, 1.0 / WALK_GROUP + 1.0 / KICK_GROUP)
    np.testing.assert_allclose(weights[:-1].sum(), 2.0)


def test_a_caption_counts_for_its_rarest_verb_only(tmp_path):
    write_verb_clips(tmp_path)
    dataset = CaptionedMotionDataset(tmp_path, VERB_CONFIG, seed=2026)
    annotations = parse_text_file(tmp_path / TEXTS_DIR / f"V{WALK_ONLY_CLIPS}.txt")

    walk_share, kick_share = dataset.caption_weights(annotations)
    expected = (1.0 / KICK_GROUP) / (1.0 / KICK_GROUP + 1.0 / WALK_GROUP)
    np.testing.assert_allclose(kick_share, expected)
    np.testing.assert_allclose(walk_share + kick_share, 1.0)


def test_a_caption_without_a_frequent_verb_gets_the_largest_group_weight(tmp_path):
    write_verb_clips(tmp_path)
    weights = CaptionedMotionDataset(tmp_path, VERB_CONFIG, seed=2026).clip_weights

    assert weights is not None
    np.testing.assert_allclose(weights[-1], 1.0 / WALK_GROUP)


def test_the_verb_loader_shows_kick_captions_as_often_as_walk_captions(tmp_path):
    write_verb_clips(tmp_path)
    module = MotionDataModule(tmp_path, VERB_CONFIG, seed=2026)
    loader = module.captioned_batch_loader(Split.TRAIN, 51)
    dataset = loader.dataset
    assert isinstance(dataset, CaptionedMotionDataset)

    captions = []
    for epoch in range(60):
        dataset.set_epoch(epoch)
        captions += [caption for batch in loader for caption in batch.captions]
    kick_share = np.mean([caption == "a person walks and kicks" for caption in captions])
    assert abs(kick_share - 1.0 / (2.0 + 1.0 / WALK_GROUP)) < 0.03
    assert len(list(loader.sampler)) == WALK_ONLY_CLIPS + MIXED_CLIPS + 1


def test_verb_weighting_without_a_frequent_verb_stops(tmp_path):
    write_clips(tmp_path)
    (tmp_path / TEXTS_DIR / "C19.txt").write_text(
        "a person#a/DET person/NOUN#0.0#0.0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="verb clip weighting needs a verb"):
        CaptionedMotionDataset(tmp_path, VERB_CONFIG, seed=2026)


def test_verb_weighting_refuses_rare_verb_captions():
    with pytest.raises(ValueError, match="picks its own caption"):
        MotionDataConfig(
            clip_weighting=ClipWeighting.VERB, caption_weighting=CaptionWeighting.RARE_VERB
        )


def test_yaml_value_selects_verb_weighting():
    data = from_mapping(MotionDataConfig, {"clip_weighting": "verb"})

    assert data.clip_weighting is ClipWeighting.VERB
