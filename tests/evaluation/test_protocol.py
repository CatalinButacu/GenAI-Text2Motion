import zlib
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from text2motion.evaluation.contracts import (
    EvaluationSample,
    GenerationEvaluationProtocol,
    GenerationEvaluationSet,
)
from text2motion.evaluation.evaluator import (
    HumanMl3dGenerationEvaluator,
    chosen_caption_index,
    deterministic_generation_seed,
)
from text2motion.evaluation.metrics import replication_summary
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.storage import FEATURES_DIR

CLIPS = 40
DROPPED = {"missing_files": 0, "not_finite": 0, "out_of_length": 0, "no_caption": 0}


def __write_clip(root, clip_id: str, frames: int, lines: list[str]) -> None:
    np.save(root / FEATURES_DIR / f"{clip_id}.npy", np.zeros((frames, 263), np.float32))
    (root / "texts" / f"{clip_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_sample_set_follows_the_guo_dataset(tmp_path):
    (tmp_path / FEATURES_DIR).mkdir()
    (tmp_path / "texts").mkdir()
    __write_clip(
        tmp_path, "A", 120, ["whole#a/DET#0.0#0.0", "part#b/DET#1.0#4.0", "tiny#c/DET#0.5#1.0"]
    )
    __write_clip(tmp_path, "B", 250, ["long#d/DET#0.0#0.0"])
    __write_clip(tmp_path, "C", 100, ["only part#e/DET#0.5#3.0"])
    resources = SimpleNamespace(
        clip_ids=lambda split: ["A", "B", "C"],
        annotation_dir=tmp_path / "texts",
        motion_dataset_dir=tmp_path,
        tokenizer_downsample_factor=4,
    )
    evaluator = HumanMl3dGenerationEvaluator(cast(GuoEvaluationResources, resources))

    samples, dropped = evaluator.evaluation_samples("test", None)

    assert [sample.sample_id for sample in samples] == ["A", "A#0", "C#0"]
    assert [sample.reference.shape[0] for sample in samples] == [120, 60, 48]
    assert [sample.captions for sample in samples] == [["whole"], ["part"], ["only part"]]
    assert dropped["out_of_length"] == 2


def test_the_caption_of_a_sample_is_drawn_again_in_each_run():
    sample = EvaluationSample("A", np.zeros((40, 263)), 10, ["x", "y", "z"], [["x"], ["y"], ["z"]])

    assert len({chosen_caption_index(sample, run) for run in range(20)}) > 1
    assert chosen_caption_index(sample, 3) == chosen_caption_index(sample, 3)


def test_run_zero_keeps_the_old_seed_and_each_run_gets_a_new_seed():
    old_seed = zlib.crc32(b"000021")

    assert deterministic_generation_seed("000021") == old_seed
    assert len({deterministic_generation_seed("000021", run) for run in range(20)}) == 20


def test_the_interval_is_the_guo_half_width_over_runs():
    mean, half_width = replication_summary([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert half_width == pytest.approx(1.96 * np.std([1.0, 2.0, 3.0]) / np.sqrt(3))


def __evaluator(monkeypatch, ids_of_run) -> tuple[HumanMl3dGenerationEvaluator, list[int]]:
    evaluator = HumanMl3dGenerationEvaluator(cast(GuoEvaluationResources, None))
    reference = [np.random.default_rng(index).normal(size=(1, 8)) for index in range(CLIPS)]
    calls: list[int] = []

    def generate(generator, split, max_clips, sampling, batch_size=1, run=0):
        calls.append(run)
        rng = np.random.default_rng(100 + run)
        return GenerationEvaluationSet(
            sample_ids=ids_of_run(run),
            reference=reference,
            generated=[rng.normal(size=(1, 8)) for _ in range(CLIPS)],
            token_lists=[[["word"]] for _ in range(CLIPS)],
            captions=[f"caption {index}" for index in range(CLIPS)],
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


def test_a_run_with_a_different_sample_set_stops_the_evaluation(monkeypatch):
    evaluator, _ = __evaluator(monkeypatch, lambda run: [f"c{i}-{run}" for i in range(CLIPS)])
    protocol = GenerationEvaluationProtocol(reps=2, bootstrap=0, mm_clips=0)

    with pytest.raises(RuntimeError, match="different sample set"):
        evaluator.run_generation_evaluation_protocol(cast(TextToMotionGenerator, None), protocol)
