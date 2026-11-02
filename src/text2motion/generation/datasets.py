from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.generation.contracts import GeneratorPretrainingRequest
from text2motion.tokenization.corpus import pack_clip_keys, read_pack_identity


############################################################
#       PretrainingBudget
############################################################
@dataclass(frozen=True)
class PretrainingBudget:
    available_examples: int
    physical_batch: int
    gradient_accumulation: int
    optimizer_steps: int
    microbatches: int
    consumed_examples: int

    @property
    def effective_batch(self) -> int:
        return self.physical_batch * self.gradient_accumulation

    @property
    def dropped_examples(self) -> int:
        return self.available_examples - self.consumed_examples

    @classmethod
    def build(
        cls, available_examples: int, physical_batch: int, gradient_accumulation: int
    ) -> PretrainingBudget:
        if available_examples < 1:
            raise ValueError("available_examples must be positive")
        if physical_batch < 1 or gradient_accumulation < 1:
            raise ValueError("physical_batch and gradient_accumulation must be positive")
        effective_batch = physical_batch * gradient_accumulation
        optimizer_steps = available_examples // effective_batch
        if optimizer_steps < 1:
            raise ValueError(
                f"not enough examples ({available_examples}) for effective batch {effective_batch}"
            )
        microbatches = optimizer_steps * gradient_accumulation
        return cls(
            available_examples=available_examples,
            physical_batch=physical_batch,
            gradient_accumulation=gradient_accumulation,
            optimizer_steps=optimizer_steps,
            microbatches=microbatches,
            consumed_examples=optimizer_steps * effective_batch,
        )

    def assert_equivalent(self, other: PretrainingBudget) -> None:
        fields = ("effective_batch", "optimizer_steps", "consumed_examples")
        mismatches = [name for name in fields if getattr(self, name) != getattr(other, name)]
        if mismatches:
            raise RuntimeError(f"pretraining budgets differ in {', '.join(mismatches)}")


def split_keys_by_clip(keys: list[str], val_fraction: float, seed: int) -> tuple[list, list]:
    stems = sorted({key.rsplit("__", 1)[0] for key in keys})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(stems))
    n_val = int(round(len(stems) * val_fraction))
    val_stems = {stems[i] for i in shuffled[:n_val]}
    train_keys = [key for key in keys if key.rsplit("__", 1)[0] not in val_stems]
    val_keys = [key for key in keys if key.rsplit("__", 1)[0] in val_stems]
    return train_keys, val_keys


############################################################
#       TokenPack
############################################################
class TokenPack(Dataset):
    def __init__(self, path: str, keys: list[str] | None = None) -> None:
        self._path = path
        if keys is None:
            with np.load(path) as pack:
                keys = pack_clip_keys(pack)
        self.keys = list(keys)
        if not self.keys:
            raise RuntimeError(f"empty token pack: {path}")
        self._z: np.lib.npyio.NpzFile | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_z": None}

    def __getitem__(self, i: int) -> torch.Tensor:
        archive = self._z
        if archive is None:
            archive = np.load(self._path)
            self._z = archive
        return torch.from_numpy(archive[self.keys[i]].astype(np.int64))


def collate_tokens(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    codebooks = batch[0].shape[1]
    length_max = max(item.shape[0] for item in batch)
    out = torch.zeros(len(batch), length_max, codebooks, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        out[i, : item.shape[0]] = item
        lengths[i] = item.shape[0]
    return out, lengths


def build_token_loaders(
    request: GeneratorPretrainingRequest, seed: int
) -> tuple[DataLoader, DataLoader | None, PretrainingBudget]:
    read_pack_identity(request.token_pack)
    with np.load(request.token_pack) as pack:
        all_keys = pack_clip_keys(pack)
    train_keys, val_keys = split_keys_by_clip(all_keys, request.val_fraction, seed)
    budget = PretrainingBudget.build(len(train_keys), request.batch_size, request.grad_accum)

    loader = DataLoader(
        TokenPack(str(request.token_pack), train_keys),
        batch_size=request.batch_size,
        shuffle=True,
        collate_fn=collate_tokens,
        drop_last=True,
        num_workers=request.num_workers,
    )
    val_loader = None
    if val_keys:
        val_loader = DataLoader(
            TokenPack(str(request.token_pack), val_keys),
            batch_size=request.batch_size,
            shuffle=False,
            collate_fn=collate_tokens,
            num_workers=request.num_workers,
        )
    return loader, val_loader, budget
