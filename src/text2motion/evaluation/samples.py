from __future__ import annotations

import zlib

import numpy as np

from text2motion.evaluation.contracts import GUO_MAX_FRAMES, GUO_MIN_FRAMES, EvaluationSample
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.motion.contracts import Split, TextAnnotation
from text2motion.motion.representation import FPS
from text2motion.motion.storage import FEATURES_DIR, parse_text_file


def deterministic_generation_seed(clip_id: str, run: int = 0) -> int:
    return zlib.crc32(clip_id.encode("utf-8")) + run


def chosen_caption_index(sample: EvaluationSample, run: int) -> int:
    seed = deterministic_generation_seed(sample.sample_id, run)
    return int(np.random.default_rng(seed).integers(len(sample.captions)))


def guo_pieces(
    clip_id: str, motion: np.ndarray, annotations: list[TextAnnotation]
) -> list[tuple[str, np.ndarray, list[TextAnnotation]]]:
    whole = [item for item in annotations if item.start_time == 0.0 and item.end_time == 0.0]
    tagged = [item for item in annotations if item not in whole]
    pieces = [(clip_id, motion, whole)] if whole else []
    for index, item in enumerate(tagged):
        segment = motion[int(item.start_time * FPS) : int(item.end_time * FPS)]
        pieces.append((f"{clip_id}#{index}", segment, [item]))
    return pieces


def listed_ids(
    resources: GuoEvaluationResources, split: Split | str, clip_list: tuple[str, ...] | None
) -> list[str]:
    ids = resources.clip_ids(split)
    if clip_list is None:
        return ids
    unknown = sorted(set(clip_list) - set(ids))
    if unknown:
        raise ValueError(f"{len(unknown)} listed clips are not in the {split} split: {unknown[:5]}")
    kept = set(clip_list)
    return [clip_id for clip_id in ids if clip_id in kept]


def load_annotated_motion(
    resources: GuoEvaluationResources, clip_id: str, dropped: dict[str, int]
) -> tuple[np.ndarray, list[TextAnnotation]] | None:
    text_path = resources.annotation_dir / f"{clip_id}.txt"
    vec_path = resources.motion_dataset_dir / FEATURES_DIR / f"{clip_id}.npy"
    if not text_path.is_file() or not vec_path.is_file():
        dropped["missing_files"] += 1
        return None
    motion = np.load(vec_path).astype(np.float32)
    if not np.isfinite(motion).all():
        dropped["not_finite"] += 1
        return None
    return motion, parse_text_file(text_path)


def piece_sample(
    sample_id: str,
    piece: np.ndarray,
    annotations: list[TextAnnotation],
    downsample: int,
    dropped: dict[str, int],
) -> EvaluationSample | None:
    if not GUO_MIN_FRAMES <= piece.shape[0] < GUO_MAX_FRAMES:
        dropped["out_of_length"] += 1
        return None
    captioned = [item for item in annotations if item.tokens]
    if not captioned:
        dropped["no_caption"] += 1
        return None
    token_len = piece.shape[0] // downsample
    return EvaluationSample(
        sample_id=sample_id,
        reference=piece[: token_len * downsample],
        token_len=token_len,
        captions=[item.caption for item in captioned],
        caption_tokens=[item.tokens for item in captioned],
    )


def read_evaluation_samples(
    resources: GuoEvaluationResources,
    split: Split | str,
    max_clips: int | None,
    clip_list: tuple[str, ...] | None,
) -> tuple[list[EvaluationSample], dict[str, int]]:
    downsample = resources.tokenizer_downsample_factor
    dropped = {"missing_files": 0, "not_finite": 0, "out_of_length": 0, "no_caption": 0}
    samples: list[EvaluationSample] = []
    for clip_id in listed_ids(resources, split, clip_list)[:max_clips]:
        loaded = load_annotated_motion(resources, clip_id, dropped)
        if loaded is None:
            continue
        for sample_id, piece, piece_annotations in guo_pieces(clip_id, *loaded):
            sample = piece_sample(sample_id, piece, piece_annotations, downsample, dropped)
            if sample is not None:
                samples.append(sample)
    return samples, dropped
