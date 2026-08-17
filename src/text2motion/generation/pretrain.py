from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.generation.model import GeneratorArchitecture, token_ce_loss
from text2motion.generation.trainer import Amp, TrainingConfig, adamw_groups

MetricSink = Callable[[dict[str, object]], None]


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


@dataclass(frozen=True)
class PretrainingRequest:
    token_pack: Path
    epochs: int = 30
    batch_size: int = 64
    grad_accum: int = 1
    num_workers: int = 0
    val_fraction: float = 0.05
    out_path: Path | None = None
    resume: bool = False


def split_keys_by_clip(keys: list[str], val_fraction: float, seed: int) -> tuple[list, list]:
    stems = sorted({key.rsplit("__", 1)[0] for key in keys})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(stems))
    n_val = int(round(len(stems) * val_fraction))
    val_stems = {stems[i] for i in shuffled[:n_val]}
    train_keys = [key for key in keys if key.rsplit("__", 1)[0] not in val_stems]
    val_keys = [key for key in keys if key.rsplit("__", 1)[0] in val_stems]
    return train_keys, val_keys


class TokenPack(Dataset):
    def __init__(self, path: str, keys: list[str] | None = None) -> None:
        self._path = path
        if keys is None:
            with np.load(path) as pack:
                keys = list(pack.keys())
        self.keys = list(keys)
        if not self.keys:
            raise RuntimeError(f"empty token pack: {path}")
        self._z: np.lib.npyio.NpzFile | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_z": None}

    def __getitem__(self, i: int) -> torch.Tensor:
        if self._z is None:
            self._z = np.load(self._path)
        return torch.from_numpy(self._z[self.keys[i]].astype(np.int64))


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
    request: PretrainingRequest, seed: int
) -> tuple[DataLoader, DataLoader | None, PretrainingBudget]:
    with np.load(request.token_pack) as pack:
        all_keys = list(pack.keys())
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


def pretrain_generator(
    architecture: GeneratorArchitecture,
    generator,
    request: PretrainingRequest,
    train: TrainingConfig,
    device: str,
    out_path: Path,
    on_metrics: MetricSink = lambda metrics: None,
    seed: int = 2026,
) -> None:
    loader, val_loader, budget = build_token_loaders(request, seed)
    print(
        f"pretrain split by clip: {len(loader.dataset)} train windows, "
        f"{0 if val_loader is None else len(val_loader.dataset)} val windows"
    )

    groups = adamw_groups(generator, train.lr, train.decay_groups)
    opt = torch.optim.AdamW(groups, lr=train.lr, weight_decay=train.weight_decay)
    total_steps = request.epochs * budget.optimizer_steps
    warmup = train.warmup_steps
    floor = train.lr_min_ratio

    def lr_multiplier(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out_path.with_name(out_path.stem + "_last.pt")

    amp_on = train.amp == Amp.BF16 and device == "cuda"
    print(
        f"segments {len(loader.dataset)}  examples/epoch {budget.consumed_examples}  "
        f"microbatches/epoch {budget.microbatches}  "
        f"optimizer_steps/epoch {budget.optimizer_steps}  "
        f"effective_batch {budget.effective_batch}  device {device}  amp {train.amp}"
    )

    step = 0
    start_epoch = 0
    if request.resume and resume_path.is_file():
        state = torch.load(resume_path, map_location=device)
        generator.load_state_dict(state["generator"])
        opt.load_state_dict(state["optimizer"])
        step, start_epoch = state["step"], state["epoch"] + 1
        print(f"resumed pretrain from {resume_path} at epoch {start_epoch} (step {step})")

    def null_condition(batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            architecture.config.text_prefix_len,
            architecture.config.d_text,
            device=device,
        )

    for epoch in range(start_epoch, request.epochs):
        generator.train()
        running = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)

        for microbatch_index, (tokens, lengths) in enumerate(loader):
            if microbatch_index >= budget.microbatches:
                break
            tokens, lengths = tokens.to(device), lengths.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                raw_loss = token_ce_loss(
                    generator(tokens, null_condition(tokens.size(0))), tokens, lengths
                )
                loss = raw_loss / request.grad_accum
            loss.backward()
            if (microbatch_index + 1) % request.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), train.grad_clip)
                for group in opt.param_groups:
                    group["lr"] = train.lr * lr_multiplier(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            running += raw_loss.item()
            seen += 1

        if step != (epoch + 1) * budget.optimizer_steps:
            raise RuntimeError(
                "pretraining optimizer-step budget drifted: "
                f"got {step}, expected {(epoch + 1) * budget.optimizer_steps}"
            )

        record = {"epoch": epoch + 1, "ce": running / seen}
        line = f"pretrain ep {epoch + 1:3d}  ce {running / seen:.4f}"

        if val_loader is not None:
            generator.eval()
            val_total = 0.0
            val_seen = 0
            with torch.no_grad():
                for tokens, lengths in val_loader:
                    tokens, lengths = tokens.to(device), lengths.to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                        val_total += float(
                            token_ce_loss(
                                generator(tokens, null_condition(tokens.size(0))), tokens, lengths
                            )
                        )
                    val_seen += 1
            record["val_ce"] = val_total / val_seen
            record["gap"] = record["val_ce"] - record["ce"]
            line += f"  val_ce {record['val_ce']:.4f}  gap {record['gap']:+.4f}"

        on_metrics(record)
        print(line)
        torch.save(generator.state_dict(), out_path)
        torch.save(
            {
                "generator": generator.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "epoch": epoch,
            },
            resume_path,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path.with_suffix(out_path.suffix + ".done").write_text(
        f"epochs {request.epochs}", encoding="utf-8"
    )
    print(f"saved pretrained init -> {out_path}")
