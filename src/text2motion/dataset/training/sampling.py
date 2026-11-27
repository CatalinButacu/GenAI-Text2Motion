from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from text2motion.dataset.contracts import CaptionWeighting, ClipWeighting, MotionDataConfig
from text2motion.dataset.training.captions import rarest_verb, usable_annotations
from text2motion.dataset.training.clip_index import ClipIndex
from text2motion.dataset.training.labels import (
    MOTION_CLASS_POWER,
    NO_MOTION_CLASS,
    NO_VERB_GROUP,
    RARE_VERB_MIN_CLIPS,
    VERB_BALANCE_MIN_CLIPS,
    caption_verbs,
    read_motion_classes,
)


def crop_to_max_frames(
    feat: np.ndarray, max_len: int, train: bool, rng: np.random.RandomState
) -> np.ndarray:
    overflow = feat.shape[0] - max_len
    if overflow <= 0:
        return feat
    start = int(rng.randint(overflow + 1)) if train else 0
    return feat[start : start + max_len]


def verbs_per_clip(index: ClipIndex) -> list[frozenset[str]]:
    found = []
    for clip_id in index.ids:
        annotations = index.annotations[clip_id]
        found.append(frozenset().union(*(caption_verbs(a) for a in annotations)))
    return found


def clip_classes(index: ClipIndex, classes: dict[str, int]) -> list[frozenset[int]]:
    found = []
    for verbs in verbs_per_clip(index):
        labels = frozenset(classes[verb] for verb in verbs if verb in classes)
        found.append(labels or frozenset({NO_MOTION_CLASS}))
    return found


def caption_groups(index: ClipIndex, counts: dict[str, int], min_len: int) -> list[list[str]]:
    lengths = index.frame_counts()
    groups = []
    for clip_id in index.ids:
        annotations = index.annotations[clip_id]
        usable = usable_annotations(annotations, lengths[clip_id], min_len)
        groups.append([rarest_verb(annotation, counts) for annotation, _ in usable])
    return groups


def verb_group_sizes(groups_of_clips: list[list[str]]) -> dict[str, int]:
    sizes = Counter(group for groups in groups_of_clips for group in groups)
    verb_sizes = [size for group, size in sizes.items() if group != NO_VERB_GROUP]
    sizes[NO_VERB_GROUP] = max(verb_sizes)
    return dict(sizes)


def verb_group_weights(groups_of_clips: list[list[str]], sizes: dict[str, int]) -> np.ndarray:
    weights = [sum(1.0 / sizes[group] for group in groups) for groups in groups_of_clips]
    return np.asarray(weights, dtype=np.float64)


def motion_class_weights(classes_of_clips: list[frozenset[int]]) -> np.ndarray:
    sizes = Counter(label for labels in classes_of_clips for label in labels)
    labelled = [size for label, size in sizes.items() if label != NO_MOTION_CLASS]
    if not labelled:
        raise ValueError("no train clip has a verb of the motion class map")
    sizes[NO_MOTION_CLASS] = max(labelled)
    rarest = [min(sizes[label] for label in labels) for labels in classes_of_clips]
    return np.asarray(rarest, dtype=np.float64) ** -MOTION_CLASS_POWER


def count_verb_clips(index: ClipIndex, min_clips: int, weighting: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip_id in index.ids:
        if clip_id.startswith("M"):
            continue
        annotations = index.annotations[clip_id]
        for verb in frozenset().union(*(caption_verbs(a) for a in annotations)):
            counts[verb] = counts.get(verb, 0) + 1
    kept = {verb: n for verb, n in counts.items() if n >= min_clips}
    if not kept:
        raise ValueError(
            f"{weighting} weighting needs a verb in at least {min_clips} "
            "train clips; no verb has that many"
        )
    return kept


############################################################
#       TrainWeighting
############################################################
@dataclass(frozen=True)
class TrainWeighting:
    verb_clip_counts: dict[str, int] | None = None
    verb_group_sizes: dict[str, int] | None = None
    clip_weights: np.ndarray | None = None


def verb_weighting(index: ClipIndex, min_len: int) -> TrainWeighting:
    counts = count_verb_clips(index, VERB_BALANCE_MIN_CLIPS, "verb clip")
    groups = caption_groups(index, counts, min_len)
    sizes = verb_group_sizes(groups)
    return TrainWeighting(counts, sizes, verb_group_weights(groups, sizes))


def train_weighting(index: ClipIndex, data: MotionDataConfig) -> TrainWeighting:
    counts: dict[str, int] | None = None
    if data.caption_weighting == CaptionWeighting.RARE_VERB:
        counts = count_verb_clips(index, RARE_VERB_MIN_CLIPS, "rare_verb caption")
    if data.clip_weighting == ClipWeighting.VERB:
        return verb_weighting(index, data.min_motion_len)
    weights: np.ndarray | None = None
    if data.clip_weighting == ClipWeighting.MOTION_CLASS:
        if data.motion_classes is None:
            raise ValueError("motion_class clip weighting needs the motion_classes verb map")
        classes = clip_classes(index, read_motion_classes(data.motion_classes))
        weights = motion_class_weights(classes)
    if data.clip_weighting == ClipWeighting.UNIFORM_REPLACEMENT:
        weights = np.ones(len(index.ids), dtype=np.float64)
    return TrainWeighting(counts, None, weights)
