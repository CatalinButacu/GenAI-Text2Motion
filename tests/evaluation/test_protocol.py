import zlib
from typing import cast

import numpy as np
import pytest

from text2motion.evaluation.contracts import (
    GenerationEvaluationProtocol,
    GenerationEvaluationSet,
)
from text2motion.evaluation.evaluator import (
    HumanMl3dGenerationEvaluator,
    deterministic_generation_seed,
)
from text2motion.evaluation.metrics import replication_summary
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.generation.pipeline import TextToMotionGenerator

CLIPS = 40
DROPPED = {"missing_files": 0, "too_short": 0, "no_caption": 0, "empty_generation": 0}


def test_run_zero_keeps_the_old_seed_and_each_run_gets_a_new_seed():
    old_seed = zlib.crc32(b"000021")

    assert deterministic_generation_seed("000021") == old_seed
    assert len({deterministic_generation_seed("000021", run) for run in range(20)}) == 20


def test_the_interval_is_the_guo_half_width_over_runs():
    mean, half_width = replication_summary([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert half_width == pytest.approx(1.96 * np.std([1.0, 2.0, 3.0]) / np.sqrt(3))


def __evaluator(monkeypatch, captions_of_run) -> tuple[HumanMl3dGenerationEvaluator, list[int]]:
    evaluator = HumanMl3dGenerationEvaluator(cast(GuoEvaluationResources, None), max_frames=196)
    reference = [np.random.default_rng(index).normal(size=(1, 8)) for index in range(CLIPS)]
    calls: list[int] = []

    def generate(generator, split, max_clips, sampling, batch_size=1, run=0):
        calls.append(run)
        rng = np.random.default_rng(100 + run)
        return GenerationEvaluationSet(
            reference=reference,
            generated=[rng.normal(size=(1, 8)) for _ in range(CLIPS)],
            token_lists=[[["word"]] for _ in range(CLIPS)],
            captions=captions_of_run(run),
            requested=CLIPS,
            dropped=dict(DROPPED),
        )

    def caption_embeddings(split):
        return np.zeros((CLIPS, 8)), [np.array([index]) for index in range(CLIPS)]

    monkeypatch.setattr(evaluator, "generate_evaluation_set", generate)
    monkeypatch.setattr(evaluator, "compute_motion_embeddings", np.concatenate)
    monkeypatch.setattr(evaluator, "caption_embeddings", caption_embeddings)
    return evaluator, calls


def test_each_generation_run_samples_all_clips_again(monkeypatch):
    evaluator, calls = __evaluator(monkeypatch, lambda run: [f"c{i}" for i in range(CLIPS)])
    protocol = GenerationEvaluationProtocol(reps=3, bootstrap=0, mm_clips=0)

    report = evaluator.run_generation_evaluation_protocol(
        cast(TextToMotionGenerator, None), protocol
    )

    assert calls == [0, 1, 2]
    assert report.runs == 3
    assert report.fid_ci95 > 0


def test_a_run_with_a_different_clip_set_stops_the_evaluation(monkeypatch):
    evaluator, _ = __evaluator(monkeypatch, lambda run: [f"c{i}-{run}" for i in range(CLIPS)])
    protocol = GenerationEvaluationProtocol(reps=2, bootstrap=0, mm_clips=0)

    with pytest.raises(RuntimeError, match="different clip set"):
        evaluator.run_generation_evaluation_protocol(cast(TextToMotionGenerator, None), protocol)
