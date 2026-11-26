from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

import torch

from text2motion.evaluation.contracts import StreamingBenchmarkProtocol
from text2motion.evaluation.reports import log_concurrency_table, log_horizon_table
from text2motion.generation.nn_generator import (
    Backbone,
    GeneratorConfig,
    GeneratorModelSpec,
    MotionTokenGenerator,
)

logger = logging.getLogger(__name__)

MODEL_ONLY_SCOPE = (
    "model-only architecture benchmark: random weights, token embedding + backbone + argmax "
    "per step; no trained checkpoint, CLIP text encoder, sampling or guidance, tokenizer "
    "decoding, transport, body fit or display (see scripts/benchmarks/service_latency.py)"
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
def advance(
    generator: MotionTokenGenerator, tokens: torch.Tensor, state, text_vector: torch.Tensor
):
    embedded = generator.add_per_step_text(generator.embed_motion_tokens(tokens), text_vector)
    h, state = generator.backbone.step(embedded, state)
    nxt = generator.predict_token_logits(h).argmax(-1)
    if generator.end_id is not None:
        nxt = nxt.clamp(max=generator.cfg.codebook_size - 1)
    return nxt, state


@torch.no_grad()
def prime_streaming_state(generator: MotionTokenGenerator, device: str, batch: int):
    cfg = generator.cfg
    state = generator.backbone.init_state(batch, torch.device(device))
    text = torch.randn(batch, cfg.text_prefix_len, cfg.d_text, device=device)
    prefix = generator.project_text_prefix(text)
    text_vector = generator.per_step_text_vector(text, prefix)
    for position in range(prefix.size(1)):
        _, state = generator.backbone.step(prefix[:, position], state)
    tokens = torch.zeros(batch, cfg.num_codebooks, dtype=torch.long, device=device)
    return tokens, state, text_vector


@torch.no_grad()
def warm_up_streaming_generator(
    generator: MotionTokenGenerator, device: str, steps: int, batch: int = 1
) -> None:
    if steps <= 0:
        return
    tokens, state, text_vector = prime_streaming_state(generator, device, batch)
    for _ in range(steps):
        tokens, state = advance(generator, tokens, state, text_vector)
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def timed_advance(
    generator: MotionTokenGenerator,
    tokens: torch.Tensor,
    state,
    text_vector: torch.Tensor,
    cuda: bool,
):
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens, state = advance(generator, tokens, state, text_vector)
    if cuda:
        torch.cuda.synchronize()
    return tokens, state, (time.perf_counter() - t0) * 1000


def horizon_row(step: int, step_ms: list[float], state, cuda: bool) -> dict[str, float]:
    segment = sorted(step_ms[-min(len(step_ms), 64) :])
    return {
        "horizon": step,
        "ms_per_step": statistics.median(segment),
        "ms_p10": segment[int(0.10 * (len(segment) - 1))],
        "ms_p90": segment[int(0.90 * (len(segment) - 1))],
        "state_bytes": streaming_state_size_bytes(state),
        "cuda_peak_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
    }


@torch.no_grad()
def benchmark_streaming_horizons(
    generator: MotionTokenGenerator,
    horizons: list[int],
    device: str,
    warmup_steps: int = StreamingBenchmarkProtocol().warmup,
) -> list[dict[str, float]]:
    cuda = device == "cuda"
    warm_up_streaming_generator(generator, device, warmup_steps)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    tokens, state, text_vector = prime_streaming_state(generator, device, 1)
    rows: list[dict[str, float]] = []
    step_ms: list[float] = []
    for step in range(1, max(horizons) + 1):
        tokens, state, elapsed = timed_advance(generator, tokens, state, text_vector, cuda)
        step_ms.append(elapsed)
        if step in horizons:
            rows.append(horizon_row(step, step_ms, state, cuda))
    return rows


@torch.no_grad()
def measure_stream_batch(
    generator: MotionTokenGenerator, batch: int, horizon: int, device: str
) -> dict[str, float]:
    cuda = device == "cuda"
    tokens, state, text_vector = prime_streaming_state(generator, device, batch)
    step_ms: list[float] = []
    for _ in range(horizon):
        tokens, state, elapsed = timed_advance(generator, tokens, state, text_vector, cuda)
        step_ms.append(elapsed)

    median_ms = statistics.median(step_ms[-min(len(step_ms), 64) :])
    return {
        "streams": batch,
        "horizon": horizon,
        "ms_per_step": median_ms,
        "state_bytes": streaming_state_size_bytes(state),
        "state_bytes_per_stream": streaming_state_size_bytes(state) / batch,
        "cuda_peak_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
        "streams_per_second": batch / (median_ms / 1000.0),
    }


@torch.no_grad()
def benchmark_concurrent_streams(
    generator: MotionTokenGenerator,
    batch_sizes: list[int],
    horizon: int,
    device: str,
    warmup_steps: int = 16,
) -> list[dict[str, float]]:
    cuda = device == "cuda"
    rows: list[dict[str, float]] = []

    for batch in batch_sizes:
        warm_up_streaming_generator(generator, device, warmup_steps, batch=batch)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        rows.append(measure_stream_batch(generator, batch, horizon, device))
        if cuda:
            torch.cuda.empty_cache()
    return rows


def run_backbone_benchmark(
    request: StreamingBenchmarkProtocol, architecture: GeneratorModelSpec, seed: int, device: str
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    torch.manual_seed(seed)
    module = architecture.create_model(device).eval()
    results = benchmark_streaming_horizons(
        module, sorted(request.horizons), device, warmup_steps=request.warmup
    )
    params = sum(p.numel() for p in module.parameters())
    logger.info(
        f"\n{architecture.backbone.value} ({params:,} params, "
        f"{architecture.config.n_layers} layers, device {device})"
    )
    log_horizon_table(results)

    rows: list[dict[str, float]] = []
    if request.streams:
        rows = benchmark_concurrent_streams(
            module, sorted(request.streams), request.streams_horizon, device
        )
        log_concurrency_table(request.streams_horizon, rows)

    del module
    if device == "cuda":
        torch.cuda.empty_cache()
    return results, rows


def write_benchmark_payload(
    request: StreamingBenchmarkProtocol,
    results: dict[str, list[dict[str, float]]],
    concurrency: dict[str, list[dict[str, float]]],
) -> dict:
    payload = {"label": request.label, "scope": MODEL_ONLY_SCOPE, "results": results}
    if concurrency:
        payload["concurrency"] = concurrency

    out_path = Path(request.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"\nwrote {out_path}")
    return payload


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
    logger.info(f"scope: {MODEL_ONLY_SCOPE}")

    for backbone in Backbone:
        architecture = GeneratorModelSpec.resolve(
            generator,
            backbone,
            codebook_size,
            num_codebooks,
            max_seq_len=max(request.horizons) + generator.text_prefix_len + request.headroom,
        )
        results[backbone.value], rows = run_backbone_benchmark(request, architecture, seed, device)
        if rows:
            concurrency[backbone.value] = rows

    return write_benchmark_payload(request, results, concurrency)
