# python scripts/training/tokenizer_preflight.py --out runs/outputs/gates/tokenizer_rvq_c1b_preflight.json -- --config configs/tokenizer/rvq_l8_1024.yaml --tokenizer rvq --epochs 500 --window 64 --batch_size 128 --ema_decay 0.99
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.cli import parse_request
from text2motion.app.cli_requests import TrainTokenizerRequest
from text2motion.app.training import create_tokenizer_trainer, with_tokenizer_kind
from text2motion.motion.contracts import Split
from text2motion.motion.storage import capture_rng_state, restore_rng_state
from text2motion.tokenization.tokenizer_trainer import TokenizerTrainer

GATE_STEPS = 300
GATE_RATIO = 0.4
TIMED_STEPS = 20

type TrainerFactory = Callable[[], TokenizerTrainer]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def overfit_gate(trainer: TokenizerTrainer, batch: torch.Tensor, steps: int) -> dict:
    first = float(trainer.train_step(batch)["recon"])
    last = first
    for _ in range(steps - 1):
        last = float(trainer.train_step(batch)["recon"])
    return {
        "first_recon": first,
        "last_recon": last,
        "ratio": last / first,
        "passed": last <= GATE_RATIO * first,
    }


def ema_coverage(trainer: TokenizerTrainer) -> dict:
    floating = sorted(
        name
        for name, buffer in trainer.tokenizer_network.named_buffers()
        if buffer.is_floating_point()
    )
    missing = sorted(set(floating) - set(trainer.parameter_ema.shadow))
    return {"floating_buffers": floating, "missing_from_ema": missing, "passed": not missing}


def max_difference(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    gaps = [float((first[name].float() - second[name].float()).abs().max()) for name in first]
    return max(gaps, default=0.0)


@torch.no_grad()
def export_coherence(
    trainer: TokenizerTrainer, batch: torch.Tensor, make_trainer: TrainerFactory
) -> dict:
    network = trainer.tokenizer_network
    with trainer.ema_weights() as module:
        module.eval()
        exported = {name: value.detach().clone() for name, value in module.state_dict().items()}
        expected = module(batch)[0]
        module.train()
    shadow = trainer.parameter_ema.shadow
    stored = {name: exported[name] for name in shadow if name in exported}
    shadow_gap = max_difference(stored, {name: shadow[name] for name in stored})
    fresh = make_trainer().tokenizer_network
    fresh.load_state_dict(exported)
    fresh.eval()
    output_gap = float((fresh(batch)[0] - expected).abs().max())
    live_differs = (
        max_difference(stored, {name: network.state_dict()[name] for name in stored}) > 0.0
    )
    return {
        "shadow_vs_export_max_diff": shadow_gap,
        "reload_output_max_diff": output_gap,
        "export_differs_from_live_weights": live_differs,
        "passed": shadow_gap == 0.0 and output_gap == 0.0,
    }


def resume_round_trip(
    trainer: TokenizerTrainer, batch: torch.Tensor, make_trainer: TrainerFactory
) -> dict:
    buffer = io.BytesIO()
    torch.save(trainer.resume_state(0, float("inf"), None), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location=batch.device, weights_only=True)
    twin = make_trainer()
    twin.load_resume_state(state, legacy=False)
    random_state = capture_rng_state()
    trainer.train_step(batch)
    restore_rng_state(random_state)
    twin.train_step(batch)
    weights_gap = max_difference(
        trainer.tokenizer_network.state_dict(), twin.tokenizer_network.state_dict()
    )
    shadow_gap = max_difference(trainer.parameter_ema.shadow, twin.parameter_ema.shadow)
    return {
        "next_step_weights_max_diff": weights_gap,
        "next_step_ema_max_diff": shadow_gap,
        "passed": weights_gap == 0.0 and shadow_gap == 0.0,
    }


def step_timing(
    trainer: TokenizerTrainer, batch: torch.Tensor, steps_per_epoch: int, epochs: int
) -> dict:
    synchronize(batch.device)
    started = time.perf_counter()
    for _ in range(TIMED_STEPS):
        trainer.train_step(batch)
    synchronize(batch.device)
    seconds = (time.perf_counter() - started) / TIMED_STEPS
    peak = (
        torch.cuda.max_memory_allocated(batch.device) / 1e9 if batch.device.type == "cuda" else 0.0
    )
    return {
        "seconds_per_step": seconds,
        "steps_per_epoch": steps_per_epoch,
        "train_minutes_per_epoch": seconds * steps_per_epoch / 60,
        "train_hours_planned": seconds * steps_per_epoch * epochs / 3600,
        "excludes": "data loading and validation evaluations",
        "peak_gpu_gb": peak,
    }


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate_steps", type=int, default=GATE_STEPS)
    own, rest = parser.parse_known_args()
    return own, [argument for argument in rest if argument != "--"]


def main() -> None:
    own, request_arguments = parse_arguments()
    request = parse_request("train-tokenizer", request_arguments)
    if not isinstance(request, TrainTokenizerRequest):
        raise TypeError("the train-tokenizer request did not parse")
    app = with_tokenizer_kind(ApplicationBootstrap.from_config(request.config), request.tokenizer)
    repository = app.create_motion_repository()
    loader = repository.window_batch_loader(Split.TRAIN, request.window, request.batch_size)
    batch = next(iter(loader)).features.to(app.device)

    def make_trainer() -> TokenizerTrainer:
        return create_tokenizer_trainer(app, request)

    trainer = make_trainer()
    report = {
        "tokenizer": request.tokenizer.value,
        "config": str(request.config),
        "batch_shape": list(batch.shape),
        "overfit_gate": overfit_gate(trainer, batch, own.gate_steps),
        "ema_coverage": ema_coverage(trainer),
        "export_coherence": export_coherence(trainer, batch, make_trainer),
        "resume_round_trip": resume_round_trip(trainer, batch, make_trainer),
        "timing": step_timing(trainer, batch, len(loader), request.epochs),
    }
    checks = ("overfit_gate", "ema_coverage", "export_coherence", "resume_round_trip")
    report["passed"] = all(report[name]["passed"] for name in checks)
    own.out.parent.mkdir(parents=True, exist_ok=True)
    own.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
