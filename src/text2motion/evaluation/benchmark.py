from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from text2motion.evaluation.contracts import StreamingBenchmarkProtocol
from text2motion.generation.model import (
    Backbone,
    GeneratorConfig,
    GeneratorModelSpec,
    MotionTokenGenerator,
)


def streaming_state_size_bytes(state) -> int:
    total = 0
    for layer_state in state:
        if layer_state is None:
            continue
        for tensor in layer_state:
            total += tensor.numel() * tensor.element_size()
    return total


@torch.no_grad()
def _advance(generator: MotionTokenGenerator, tokens: torch.Tensor, state):
    h, state = generator.backbone.step(generator.embed_motion_tokens(tokens), state)
    nxt = generator.predict_token_logits(h).argmax(-1)
    if generator.end_id is not None:
        nxt = nxt.clamp(max=generator.cfg.codebook_size - 1)
    return nxt, state


@torch.no_grad()
def warm_up_streaming_generator(generator: MotionTokenGenerator, device: str, steps: int, batch: int = 1) -> None:
    if steps <= 0:
        return
    cfg = generator.cfg
    state = generator.backbone.init_state(batch, torch.device(device))
    text = torch.randn(batch, cfg.text_prefix_len, cfg.d_text, device=device)
    prefix = generator.project_text_prefix(text)
    for position in range(prefix.size(1)):
        _, state = generator.backbone.step(prefix[:, position], state)
    tokens = torch.zeros(batch, cfg.num_codebooks, dtype=torch.long, device=device)
    for _ in range(steps):
        tokens, state = _advance(generator, tokens, state)
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_streaming_horizons(
    generator: MotionTokenGenerator,
    horizons: list[int],
    device: str,
    warmup_steps: int = StreamingBenchmarkProtocol().warmup,
) -> list[dict[str, float]]:
    cfg = generator.cfg
    cuda = device == "cuda"
    warm_up_streaming_generator(generator, device, warmup_steps)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    state = generator.backbone.init_state(1, torch.device(device))
    text = torch.randn(1, cfg.text_prefix_len, cfg.d_text, device=device)
    prefix = generator.project_text_prefix(text)
    for position in range(prefix.size(1)):
        h, state = generator.backbone.step(prefix[:, position], state)

    rows: list[dict[str, float]] = []
    step_ms: list[float] = []
    tokens = torch.zeros(1, cfg.num_codebooks, dtype=torch.long, device=device)
    for step in range(1, max(horizons) + 1):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens, state = _advance(generator, tokens, state)
        if cuda:
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000)

        if step in horizons:
            segment = sorted(step_ms[-min(len(step_ms), 64) :])
            rows.append(
                {
                    "horizon": step,
                    "ms_per_step": statistics.median(segment),
                    "ms_p10": segment[int(0.10 * (len(segment) - 1))],
                    "ms_p90": segment[int(0.90 * (len(segment) - 1))],
                    "state_bytes": streaming_state_size_bytes(state),
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                }
            )
    return rows


@torch.no_grad()
def benchmark_concurrent_streams(
    generator: MotionTokenGenerator,
    batch_sizes: list[int],
    horizon: int,
    device: str,
    warmup_steps: int = 16,
) -> list[dict[str, float]]:
    cfg = generator.cfg
    cuda = device == "cuda"
    rows: list[dict[str, float]] = []

    for batch in batch_sizes:
        warm_up_streaming_generator(generator, device, warmup_steps, batch=batch)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        state = generator.backbone.init_state(batch, torch.device(device))
        text = torch.randn(batch, cfg.text_prefix_len, cfg.d_text, device=device)
        prefix = generator.project_text_prefix(text)
        for position in range(prefix.size(1)):
            _, state = generator.backbone.step(prefix[:, position], state)

        tokens = torch.zeros(batch, cfg.num_codebooks, dtype=torch.long, device=device)
        step_ms: list[float] = []
        for _ in range(horizon):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokens, state = _advance(generator, tokens, state)
            if cuda:
                torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1000)

        median_ms = statistics.median(step_ms[-min(len(step_ms), 64) :])
        rows.append(
            {
                "streams": batch,
                "horizon": horizon,
                "ms_per_step": median_ms,
                "state_bytes": streaming_state_size_bytes(state),
                "state_bytes_per_stream": streaming_state_size_bytes(state) / batch,
                "cuda_peak_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                "streams_per_second": batch / (median_ms / 1000.0),
            }
        )
        del state
        if cuda:
            torch.cuda.empty_cache()
    return rows


def run_streaming_state_latency_benchmark(
    request: StreamingBenchmarkProtocol,
    generator: GeneratorConfig,
    codebook_size: int,
    num_codebooks: int,
    seed: int,
    device: str,
) -> dict:
    results: dict[str, list[dict[str, float]]] = {}
    concurrency: dict[str, list[dict[str, float]]] = {}
    horizons = sorted(request.horizons)

    for backbone in Backbone:
        bench_seq_len = max(horizons) + generator.text_prefix_len + request.headroom
        architecture = GeneratorModelSpec.resolve(
            generator,
            backbone,
            codebook_size,
            num_codebooks,
            max_seq_len=bench_seq_len,
        )
        torch.manual_seed(seed)
        module = architecture.create_model(device).eval()
        results[backbone.value] = benchmark_streaming_horizons(
            module, horizons, device, warmup_steps=request.warmup
        )
        params = sum(p.numel() for p in module.parameters())
        print(
            f"\n{backbone.value} ({params:,} params, "
            f"{architecture.config.n_layers} layers, device {device})"
        )
        print(f"{'horizon':>8} {'ms/step':>9} {'state':>12} {'cuda peak':>12}")
        for row in results[backbone.value]:
            print(
                f"{row['horizon']:>8} {row['ms_per_step']:>9.2f} "
                f"{row['state_bytes'] / 1e6:>10.2f}MB {row['cuda_peak_bytes'] / 1e6:>10.1f}MB"
            )

        if request.streams:
            rows = benchmark_concurrent_streams(
                module, sorted(request.streams), request.streams_horizon, device
            )
            concurrency[backbone.value] = rows
            print(f"  concurrency @ horizon {request.streams_horizon}")
            print(f"  {'streams':>8} {'ms/step':>9} {'state':>12} {'per-stream':>12} {'peak':>10}")
            for row in rows:
                print(
                    f"  {row['streams']:>8} {row['ms_per_step']:>9.2f} "
                    f"{row['state_bytes'] / 1e6:>10.2f}MB "
                    f"{row['state_bytes_per_stream'] / 1e6:>10.3f}MB "
                    f"{row['cuda_peak_bytes'] / 1e6:>8.1f}MB"
                )

        del module
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {"label": request.label, "results": results}
    if concurrency:
        payload["concurrency"] = concurrency

    out_path = Path(request.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return payload
