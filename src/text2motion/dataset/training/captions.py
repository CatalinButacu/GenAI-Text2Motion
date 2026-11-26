from __future__ import annotations

import numpy as np

from text2motion.dataset.training.labels import (
    NO_VERB_GROUP,
    RARE_VERB_POWER,
    caption_verbs,
)
from text2motion.motion.contracts import TextAnnotation
from text2motion.motion.representation import FPS


def annotation_frame_range(annotation: TextAnnotation, total: int) -> tuple[int, int] | None:
    if annotation.start_time == 0.0 and annotation.end_time == 0.0:
        return None
    start = max(int(annotation.start_time * FPS), 0)
    end = min(int(annotation.end_time * FPS), total)
    return start, end


def usable_annotations(
    annotations: list[TextAnnotation], total: int, min_len: int
) -> list[tuple[TextAnnotation, tuple[int, int] | None]]:
    usable = []
    for annotation in annotations:
        segment = annotation_frame_range(annotation, total)
        if segment is None or segment[1] - segment[0] >= min_len:
            usable.append((annotation, segment))
    return usable


def rarest_verb(annotation: TextAnnotation, counts: dict[str, int]) -> str:
    kept = [verb for verb in caption_verbs(annotation) if verb in counts]
    return min(kept, key=lambda verb: (counts[verb], verb), default=NO_VERB_GROUP)


############################################################
#       CaptionPicker
############################################################
class CaptionPicker:
    def __init__(
        self,
        min_len: int,
        verb_clip_counts: dict[str, int] | None,
        verb_group_sizes: dict[str, int] | None,
    ) -> None:
        self._min_len = min_len
        self._verb_clip_counts = verb_clip_counts
        self._verb_group_sizes = verb_group_sizes

    @property
    def verb_clip_counts(self) -> dict[str, int] | None:
        return None if self._verb_clip_counts is None else dict(self._verb_clip_counts)

    def select_captioned_segment(
        self, annotations: list[TextAnnotation], feat: np.ndarray, rng: np.random.RandomState
    ) -> tuple[TextAnnotation, np.ndarray]:
        usable = usable_annotations(annotations, feat.shape[0], self._min_len)
        if not usable:
            raise ValueError("no caption of this clip covers a segment of the minimum length")
        annotation, segment = usable[self.caption_index([a for a, _ in usable], rng)]
        return annotation, feat if segment is None else feat[segment[0] : segment[1]]

    def caption_index(self, annotations: list[TextAnnotation], rng: np.random.RandomState) -> int:
        if self._verb_clip_counts is None:
            return rng.randint(len(annotations))
        return int(rng.choice(len(annotations), p=self.caption_weights(annotations)))

    def caption_weights(self, annotations: list[TextAnnotation]) -> np.ndarray:
        counts = self._verb_clip_counts
        if counts is None:
            return np.full(len(annotations), 1.0 / len(annotations))
        sizes = self._verb_group_sizes
        if sizes is not None:
            groups = [rarest_verb(annotation, counts) for annotation in annotations]
            weights = np.asarray([1.0 / sizes[group] for group in groups], dtype=np.float64)
            return weights / weights.sum()
        common = max(counts.values())
        rarest = [
            min((counts[verb] for verb in caption_verbs(a) if verb in counts), default=common)
            for a in annotations
        ]
        weights = np.asarray(rarest, dtype=np.float64) ** -RARE_VERB_POWER
        return weights / weights.sum()
