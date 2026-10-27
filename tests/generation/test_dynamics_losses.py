from typing import cast

import pytest
import torch

from text2motion.generation.contracts import LossWeights
from text2motion.generation.losses import __dynamics_terms
from text2motion.motion.representation import DIM, SLICES
from text2motion.tokenization.nn_model import TokenizerNetwork


def __standing_motion(frames: int = 12) -> torch.Tensor:
    features = torch.zeros(1, frames, DIM)
    features[..., 3] = 0.9
    features[..., SLICES["foot"]] = 1.0
    return features


def test_default_weights_keep_the_existing_recipe():
    weights = LossWeights()
    assert not weights.dynamics_enabled
    assert "foot_skate" not in weights.active_terms()
    assert "accel" not in weights.active_terms()
    enabled = LossWeights(foot_skate=0.5, accel=0.2)
    assert enabled.needs_positions
    assert {"foot_skate", "accel"} <= set(enabled.active_terms())


def test_a_body_sliding_over_planted_feet_is_penalized_and_a_still_one_is_not():
    mean, std = torch.zeros(DIM), torch.ones(DIM)
    gt = __standing_motion()
    still = __dynamics_terms(gt.clone(), gt, None, mean, std)
    assert still["foot_skate"].item() == pytest.approx(0.0, abs=1e-6)
    sliding = gt.clone()
    sliding[..., 1] = 0.05
    slid = __dynamics_terms(sliding, gt, None, mean, std)
    assert slid["foot_skate"].item() == pytest.approx(0.05, rel=1e-3)


def test_a_one_frame_jitter_raises_the_acceleration_term():
    mean, std = torch.zeros(DIM), torch.ones(DIM)
    gt = __standing_motion()
    assert __dynamics_terms(gt.clone(), gt, None, mean, std)["accel"].item() == pytest.approx(
        0.0, abs=1e-7
    )
    jittery = gt.clone()
    jittery[0, 6, 3] += 0.1
    expected = (0.1 + 0.2 + 0.1) / (10 * 22 * 3)
    assert __dynamics_terms(jittery, gt, None, mean, std)["accel"].item() == pytest.approx(
        expected, rel=1e-3
    )


def test_padding_frames_do_not_count():
    mean, std = torch.zeros(DIM), torch.ones(DIM)
    gt = __standing_motion()
    recon = gt.clone()
    recon[0, 9:, 1] = 0.5
    terms = __dynamics_terms(recon, gt, torch.tensor([8]), mean, std)
    assert terms["foot_skate"].item() == pytest.approx(0.0, abs=1e-6)


def test_dynamics_losses_refuse_to_run_without_normalization_stats():
    from text2motion.generation.losses import generator_loss

    with pytest.raises(ValueError, match="mean and std"):
        generator_loss(
            torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, 1, dtype=torch.long),
            torch.zeros(1, 8, DIM),
            cast(TokenizerNetwork, _FakeTokenizer()),
            LossWeights(foot_skate=1.0),
            4,
        )


class _FakeTokenizer:
    def __init__(self) -> None:
        unit = type(
            "Unit",
            (),
            {"codebook_size": 4, "indices_to_codes": lambda self, idx: torch.zeros(len(idx), 2)},
        )()
        self.quantizer = type(
            "Quantizer",
            (),
            {"quantizers": [unit], "combine_code_vectors": lambda self, codes: codes[0]},
        )()
        self.post_q = lambda codes: codes
        self.decoder = lambda codes: torch.zeros(codes.size(0), codes.size(1) * 4, DIM)

    def quantizer_stack(self):
        return self.quantizer


def test_dynamics_weights_load_from_flat_train_keys_and_nested_blocks_are_refused(tmp_path):
    from text2motion.app.config import load_config

    good = tmp_path / "good.yaml"
    good.write_text("train:\n  w_foot_skate: 10.0\n  w_accel: 18.0\n", encoding="utf-8")
    weights = load_config(good).train.loss_weights
    assert weights.foot_skate == 10.0 and weights.accel == 18.0
    assert weights.root == LossWeights().root
    nested = tmp_path / "nested.yaml"
    nested.write_text("train:\n  loss_weights:\n    foot_skate: 10.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="flat train keys"):
        load_config(nested)
