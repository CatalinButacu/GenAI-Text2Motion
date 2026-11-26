from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np

from text2motion.evaluation.contracts import EvaluationSample, GenerationEvaluationSet
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.evaluation.samples import deterministic_generation_seed
from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.contracts import Split


def token_length_batches(
    samples: list[EvaluationSample], batch_size: int
) -> Iterator[tuple[int, list[int]]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for position, sample in enumerate(samples):
        buckets[sample.token_len].append(position)

    for token_len, positions in buckets.items():
        for start in range(0, len(positions), max(1, batch_size)):
            yield token_len, positions[start : start + max(1, batch_size)]


def generate_sample_motions(
    resources: GuoEvaluationResources,
    generator: TextToMotionGenerator,
    samples: list[EvaluationSample],
    chosen: list[int],
    sampling: SamplingConfig,
    batch_size: int,
    run: int,
) -> list[np.ndarray | None]:
    generated: list[np.ndarray | None] = [None] * len(samples)
    for token_len, group in token_length_batches(samples, batch_size):
        batch = generator.generate_batch(
            [samples[i].captions[chosen[i]] for i in group],
            token_len,
            sampling,
            seeds=[deterministic_generation_seed(samples[i].sample_id, run) for i in group],
        )
        if batch is None:
            continue
        for position, item in zip(group, batch, strict=True):
            generated[position] = np.asarray(
                resources.denormalize(item.motion.features.cpu().numpy())
            )
    return generated


def collect_scorable_set(
    split: Split | str,
    samples: list[EvaluationSample],
    chosen: list[int],
    generated: list[np.ndarray | None],
    dropped: dict[str, int],
) -> GenerationEvaluationSet:
    scorable = [index for index, motion in enumerate(generated) if motion is not None]
    dropped["empty_generation"] = len(samples) - len(scorable)
    if not scorable:
        raise RuntimeError(f"no scorable samples in split {split!r}; dropped {dropped}")

    return GenerationEvaluationSet(
        sample_ids=[samples[i].sample_id for i in scorable],
        reference=[samples[i].reference for i in scorable],
        generated=[motion for motion in generated if motion is not None],
        token_lists=[[samples[i].caption_tokens[chosen[i]]] for i in scorable],
        captions=[samples[i].captions[chosen[i]] for i in scorable],
        requested=len(samples),
        dropped=dropped,
    )
