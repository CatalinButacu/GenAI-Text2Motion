# .\.venv\Scripts\python.exe scripts/benchmarks/kernel_probe.py
from __future__ import annotations

import argparse
import logging
import statistics
import time

import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.run_log import log_metrics, start_run
from text2motion.generation.contracts import Backbone

logger = logging.getLogger(__name__)


def training_step_cost(model, tokens, text, repeats: int) -> tuple[float, float]:
    model.train()
    durations: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(repeats + 3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        model(tokens, text).float().pow(2).mean().backward()
        torch.cuda.synchronize()
        if index >= 3:
            durations.append((time.perf_counter() - started) * 1000)
    peak_megabytes = torch.cuda.max_memory_allocated() / 2**20
    return statistics.median(durations), peak_megabytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=49)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    app = ApplicationBootstrap.from_config(args.config)
    if app.device != "cuda":
        raise RuntimeError("the kernel probe needs a CUDA device")
    run_dir = start_run("kernel_probe", app.config, app.config.paths.logs_dir, vars(args))
    codebooks = app.config.tokenizer.num_quantizers
    vocabulary = 1024
    plain = app.resolve_generator_spec(Backbone.MAMBA, vocabulary, use_kernel=False).create_model(
        app.device
    )
    kernel = app.resolve_generator_spec(Backbone.MAMBA, vocabulary, use_kernel=True).create_model(
        app.device
    )
    kernel.load_state_dict(plain.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(0)
    tokens = torch.randint(
        0, vocabulary, (args.batch_size, args.tokens, codebooks), generator=generator
    ).to(app.device)
    text = torch.randn(args.batch_size, app.config.generator.d_text, generator=generator).to(
        app.device
    )

    with torch.no_grad():
        plain.eval()
        kernel.eval()
        difference = (plain(tokens, text) - kernel(tokens, text)).abs().max().item()
    plain_ms, plain_megabytes = training_step_cost(plain, tokens, text, args.repeats)
    kernel_ms, kernel_megabytes = training_step_cost(kernel, tokens, text, args.repeats)
    report = {
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "max_logit_difference": difference,
        "plain_step_ms": plain_ms,
        "kernel_step_ms": kernel_ms,
        "speedup": plain_ms / kernel_ms,
        "plain_peak_mb": plain_megabytes,
        "kernel_peak_mb": kernel_megabytes,
    }
    log_metrics(run_dir, report)
    logger.info(f"largest difference between the two outputs: {difference:.2e}")
    logger.info(
        f"training step (forward + backward), batch {args.batch_size} x {args.tokens} tokens: "
        f"plain {plain_ms:.1f} ms, kernel {kernel_ms:.1f} ms, speed-up {plain_ms / kernel_ms:.2f}x"
    )
    logger.info(
        f"peak GPU memory: plain {plain_megabytes:.0f} MB, kernel {kernel_megabytes:.0f} MB"
    )


if __name__ == "__main__":
    main()
