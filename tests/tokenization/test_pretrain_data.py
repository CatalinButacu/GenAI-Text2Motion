import json
import pickle
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from text2motion.app.config import GeneratorTrainingConfig
from text2motion.generation.contracts import GeneratorPretrainingRequest
from text2motion.generation.nn_model import GeneratorModelSpec
from text2motion.generation.pretrain import (
    PretrainingBudget,
    TokenPack,
    build_token_loaders,
    collate_tokens,
    pretrain_generator,
    split_keys_by_clip,
)
from text2motion.tokenization.corpus import PACK_FORMAT, PACK_IDENTITY_KEY, PACK_VERSION


def make_pack(tmp_path, clips=("clipA", "clipB", "clipC", "clipD"), windows=3, identity=False):
    path = tmp_path / "pack.npz"
    arrays = {}
    for c_index, clip in enumerate(clips):
        for w in range(windows):
            arrays[f"{clip}__{w}"] = np.full((4 + w, 2), c_index, dtype=np.int64)
    if identity:
        record = {"format": PACK_FORMAT, "version": PACK_VERSION}
        arrays[PACK_IDENTITY_KEY] = np.array(json.dumps(record))
    np.savez(path, **arrays)
    return path


def test_pretrain_refuses_a_token_pack_without_an_identity_record(tmp_path):
    request = GeneratorPretrainingRequest(token_pack=make_pack(tmp_path), epochs=1, batch_size=2)

    with pytest.raises(ValueError, match="without an identity record"):
        build_token_loaders(request, seed=2026)


def test_the_identity_record_is_not_a_training_window(tmp_path):
    request = GeneratorPretrainingRequest(
        token_pack=make_pack(tmp_path, identity=True), epochs=1, batch_size=2, val_fraction=0.0
    )

    loader, _, _ = build_token_loaders(request, seed=2026)

    assert PACK_IDENTITY_KEY not in cast(TokenPack, loader.dataset).keys
    assert len(cast(TokenPack, loader.dataset)) == 12


def test_pretrain_resume_without_the_resume_file_fails(tmp_path):
    request = GeneratorPretrainingRequest(
        token_pack=make_pack(tmp_path, identity=True), epochs=1, batch_size=2, resume=True
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pretrain_generator(
            cast(GeneratorModelSpec, None),
            torch.nn.Linear(2, 2),
            request,
            GeneratorTrainingConfig(),
            "cpu",
            tmp_path / "pretrained.pt",
        )


def test_windows_of_one_clip_never_straddle_the_split(tmp_path):
    pack = make_pack(tmp_path)
    keys = list(np.load(pack).keys())

    train, val = split_keys_by_clip(keys, 0.5, seed=2026)

    assert train and val
    train_clips = {k.rsplit("__", 1)[0] for k in train}
    val_clips = {k.rsplit("__", 1)[0] for k in val}
    assert not (train_clips & val_clips)
    assert sorted(train + val) == sorted(keys)


def test_split_is_deterministic_for_a_seed(tmp_path):
    keys = list(np.load(make_pack(tmp_path)).keys())

    assert split_keys_by_clip(keys, 0.25, 2026) == split_keys_by_clip(keys, 0.25, 2026)
    assert split_keys_by_clip(keys, 0.25, 7) != split_keys_by_clip(keys, 0.25, 2026)


def test_token_pack_is_picklable_so_dataloader_workers_are_safe(tmp_path):
    pack = TokenPack(str(make_pack(tmp_path)))
    first = pack[0]

    revived = pickle.loads(pickle.dumps(pack))

    assert len(revived) == len(pack)
    assert np.array_equal(revived[0].numpy(), first.numpy())


def test_empty_pack_fails_loud(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path)
    with pytest.raises(RuntimeError, match="empty token pack"):
        TokenPack(str(path))


def test_collate_pads_and_reports_true_lengths(tmp_path):
    pack = TokenPack(str(make_pack(tmp_path)))
    batch = [pack[0], pack[1], pack[2]]

    padded, lengths = collate_tokens(batch)

    assert padded.shape[0] == 3
    assert padded.shape[1] == max(int(b.shape[0]) for b in batch)
    assert lengths.tolist() == [int(b.shape[0]) for b in batch]
    for row, length in enumerate(lengths.tolist()):
        assert padded[row, length:].sum() == 0


def test_pretraining_budget_matches_different_physical_batches():
    transformer = PretrainingBudget.build(8_303, physical_batch=64, gradient_accumulation=1)
    mamba = PretrainingBudget.build(8_303, physical_batch=8, gradient_accumulation=8)

    transformer.assert_equivalent(mamba)
    assert transformer.optimizer_steps == 129
    assert transformer.microbatches == 129
    assert mamba.microbatches == 1_032
    assert transformer.consumed_examples == mamba.consumed_examples == 8_256


@pytest.mark.parametrize("batch_size,grad_accum", [(0, 1), (1, 0), (-1, 1)])
def test_pretraining_budget_rejects_invalid_accumulation(batch_size, grad_accum):
    with pytest.raises(ValueError, match="must be positive"):
        PretrainingBudget.build(100, batch_size, grad_accum)


class _FailingGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens, condition):
        raise RuntimeError("boom")


def test_a_failed_pretraining_step_writes_a_dump_with_the_tokens(tmp_path):
    request = GeneratorPretrainingRequest(
        token_pack=make_pack(tmp_path, identity=True), epochs=1, batch_size=2, val_fraction=0.0
    )
    architecture = SimpleNamespace(config=SimpleNamespace(text_prefix_len=1, d_text=2))
    dump_path = tmp_path / "failure_dump.pt"

    with pytest.raises(RuntimeError, match="boom"):
        pretrain_generator(
            cast(GeneratorModelSpec, architecture),
            _FailingGenerator(),
            request,
            GeneratorTrainingConfig(),
            "cpu",
            tmp_path / "pretrained.pt",
            failure_dump_path=dump_path,
        )

    dump = torch.load(dump_path, weights_only=True)
    assert dump["epoch"] == 0
    assert dump["epoch_step"] == 0
    assert dump["inputs"]["tokens"].shape[0] == 2
    assert "generator" in dump
