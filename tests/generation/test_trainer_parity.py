import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.fixture import build_fixture_dataset
from text2motion.dataset.training.batching import collate_motion_clips
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.generation.contracts import Backbone, GeneratorConfig, GeneratorTrainingConfig
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.requests import GeneratorPretrainingRequest
from text2motion.generation.training import pretrain_epoch as epoch_module
from text2motion.generation.training.pretrain import pretrain_generator
from text2motion.generation.training.resume import GeneratorCheckpoint
from text2motion.generation.training.trainer import GeneratorTrainer
from text2motion.motion.contracts import Split
from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.corpus import PACK_FORMAT, PACK_IDENTITY_KEY, PACK_VERSION
from text2motion.tokenization.network.finite_scalar import FiniteScalarQuantizationTokenizer
from text2motion.tokenization.tokenizer_trainer import TokenizerTrainer

PARITY_FILE = Path(__file__).with_name("trainer_parity.json")
SEED = 2026
STEPS = 3
TOLERANCE = 1e-6
TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone=Backbone.MAMBA,
    d_model=64,
    n_layers=2,
    d_text=32,
    num_codebooks=2,
    codebook_size=16,
    max_seq_len=16,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
)


def fixture_batches(root: Path) -> list:
    build_fixture_dataset(root)
    dataset = CaptionedMotionDataset(
        root, MotionDataConfig(mirror_augment=False), Split.TRAIN, seed=SEED
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_motion_clips)
    return list(loader)


def tokenizer_record(batches: list) -> dict:
    torch.manual_seed(SEED)
    trainer = TokenizerTrainer(
        FiniteScalarQuantizationTokenizer(TOK), lr=1e-3, weight_decay=0.0, ema_decay=0.9
    )
    losses = [
        float(trainer.train_step(batch.features, batch.lengths)["total"])
        for batch in batches[:STEPS]
    ]
    state = trainer.resume_state(0, float("inf"), None)
    return {
        "losses": losses,
        "checkpoint_keys": sorted(state),
        "network_keys": sorted(state["tokenizer"]),
    }


def generator_record(batches: list) -> dict:
    torch.manual_seed(SEED)
    trainer = GeneratorTrainer(
        MotionTokenGenerator(GEN),
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(),
        downsample=TOK.downsample,
    )
    text = torch.randn(2, GEN.d_text)
    losses = [
        float(trainer.train_step(batch.features, text, batch.lengths)["total"])
        for batch in batches[:STEPS]
    ]
    state = GeneratorCheckpoint(trainer).resume_state(0, float("inf"), 0, None)
    return {
        "losses": losses,
        "checkpoint_keys": sorted(state),
        "network_keys": sorted(state["generator"]),
    }


def pretrain_record(root: Path, monkeypatch) -> dict:
    arrays: dict[str, Any] = {
        f"clip{index}__0": np.random.default_rng(SEED + index).integers(0, 16, (8, 2))
        for index in range(STEPS)
    }
    identity = json.dumps({"format": PACK_FORMAT, "version": PACK_VERSION})
    pack = root / "pack.npz"
    arrays[PACK_IDENTITY_KEY] = np.array(identity)
    np.savez(pack, **arrays)
    losses: list[float] = []
    original = epoch_module.token_ce_loss

    def recording(*args, **kwargs):
        loss = original(*args, **kwargs)
        losses.append(loss.item())
        return loss

    monkeypatch.setattr(epoch_module, "token_ce_loss", recording)
    torch.manual_seed(SEED)
    out_path = root / "pretrained.pt"
    pretrain_generator(
        cast(GeneratorModelSpec, SimpleNamespace(config=GEN)),
        MotionTokenGenerator(GEN),
        GeneratorPretrainingRequest(token_pack=pack, epochs=1, batch_size=1, val_fraction=0.0),
        GeneratorTrainingConfig(),
        "cpu",
        out_path,
        seed=SEED,
    )
    resume = torch.load(out_path.with_name("pretrained_last.pt"), weights_only=True)
    return {
        "losses": losses,
        "checkpoint_keys": sorted(resume),
        "network_keys": sorted(torch.load(out_path, weights_only=True)),
    }


def measure(root: Path, monkeypatch) -> dict:
    batches = fixture_batches(root / "fixture")
    (root / "pretrain").mkdir()
    return {
        "tokenizer_trainer": tokenizer_record(batches),
        "generator_trainer": generator_record(batches),
        "pretrain": pretrain_record(root / "pretrain", monkeypatch),
    }


def test_the_trainers_match_the_recorded_parity_values(tmp_path, monkeypatch):
    recorded = json.loads(PARITY_FILE.read_text(encoding="utf-8"))
    measured = measure(tmp_path, monkeypatch)

    assert set(measured) == set(recorded)
    for name, record in recorded.items():
        assert len(measured[name]["losses"]) == STEPS
        assert measured[name]["checkpoint_keys"] == record["checkpoint_keys"]
        assert measured[name]["network_keys"] == record["network_keys"]
        for got, expected in zip(measured[name]["losses"], record["losses"], strict=True):
            assert abs(got - expected) < TOLERANCE
