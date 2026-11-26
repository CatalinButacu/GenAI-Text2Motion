import zlib
from types import MappingProxyType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from text2motion.evaluation.contracts import GenerationEvaluationProtocol
from text2motion.evaluation.evaluator import HumanMl3dGenerationEvaluator
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.storage import FEATURES_DIR

CLIPS = 40
WIDTH = 8
EXPECTED_REPORT = MappingProxyType(
    {
        "clips": 40,
        "runs": 3,
        "fid": 0.43419807749429995,
        "diversity": 2.733330567677816,
        "multimodality": 2.447979688644409,
        "matching_distance": 5.376385529836019,
        "fid_ci95": 0.0536887828004403,
        "diversity_ci95": 0.17747757110782586,
        "matching_distance_ci95": 0.12613881790303402,
        "fid_bootstrap_lo": 0.5473625685553746,
        "fid_bootstrap_hi": 0.9613131309642786,
        "r_top1": 0.052083333333333336,
        "r_top2": 0.052083333333333336,
        "r_top3": 0.0625,
        "r_top1_ci95": 0.016670138527274406,
        "r_top2_ci95": 0.016670138527274406,
        "r_top3_ci95": 0.028873526898450692,
    }
)
EXPECTED_SELECTION = MappingProxyType(
    {
        "clips": 40,
        "requested_clips": 42,
        "dropped": 2,
        "fid": 0.40467152123596045,
        "r_top1": 0.0625,
        "r_top3": 0.0625,
        "mm_dist": 5.4762420654296875,
        "diversity": 2.7706098556518555,
    }
)


class FixedGenerator:
    def eval(self):
        return self

    def generate_batch(self, captions, token_len, sampling, seeds=None):
        if token_len == 25:
            return None

        seeds = seeds or [zlib.crc32(f"{c}{i}".encode()) for i, c in enumerate(captions)]
        items = []

        for seed in seeds:
            values = np.random.default_rng(seed).normal(size=(token_len * 4, 263))
            features = torch.from_numpy(values.astype(np.float32))
            items.append(SimpleNamespace(motion=SimpleNamespace(features=features)))

        return items


def embed_motion(padded, lengths):
    rows = [row[:length, :WIDTH] for row, length in zip(padded, lengths, strict=True)]
    return torch.stack([torch.cat([row[0, :4], row.std(0)[4:]]) for row in rows])


def embed_text(words, tags, lengths):
    return words.sum(1) + tags.sum(1)


def text_inputs(tokens):
    rng = np.random.default_rng(zlib.crc32(" ".join(tokens).encode()))
    shape = (len(tokens), WIDTH)
    return rng.normal(size=shape).astype(np.float32), rng.normal(size=shape).astype(np.float32)


def fixed_evaluator(root) -> HumanMl3dGenerationEvaluator:
    (root / FEATURES_DIR).mkdir()
    (root / "texts").mkdir()
    ids = [f"{index:03d}" for index in range(CLIPS + 2)]

    for index, clip_id in enumerate(ids):
        frames = 100 if index >= CLIPS else 48 + 4 * (index % 3)
        motion = np.random.default_rng(index).normal(size=(frames, 263)).astype(np.float32)
        np.save(root / FEATURES_DIR / f"{clip_id}.npy", motion)
        lines = [f"one {index}#a{index}/DET b/NOUN#0.0#0.0", f"two {index}#c{index}/DET#0.0#0.0"]
        (root / "texts" / f"{clip_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    resources = SimpleNamespace(
        clip_ids=lambda split: ids,
        annotation_dir=root / "texts",
        motion_dataset_dir=root,
        tokenizer_downsample_factor=4,
        motion_embedder=embed_motion,
        text_embedder=embed_text,
        guo_motion_mean=np.zeros(263, np.float32),
        guo_motion_std=np.ones(263, np.float32),
        device="cpu",
        denormalize=lambda features: features,
        build_guo_text_inputs=text_inputs,
    )

    return HumanMl3dGenerationEvaluator(cast(GuoEvaluationResources, resources))


def test_the_protocol_report_keeps_the_recorded_values(tmp_path):
    protocol = GenerationEvaluationProtocol(
        reps=3, bootstrap=5, mm_clips=3, mm_repeats=4, batch_size=7, sampling=SamplingConfig()
    )

    report = fixed_evaluator(tmp_path).run_generation_evaluation_protocol(
        cast(TextToMotionGenerator, FixedGenerator()), protocol
    )

    assert report.as_dict() == pytest.approx(dict(EXPECTED_REPORT), rel=1e-6)


def test_the_selection_metrics_keep_the_recorded_values(tmp_path):
    metrics = fixed_evaluator(tmp_path).evaluate_checkpoint_selection_metrics(
        cast(TextToMotionGenerator, FixedGenerator()), batch_size=7
    )

    assert metrics == pytest.approx(dict(EXPECTED_SELECTION), rel=1e-6)
