"""Constant-memory benchmark: streaming SSM vs parallel-scan equivalent.

This script produces the load-bearing dissertation figure for the claim:

    "Mamba's recurrent hidden state gives O(1) per-step memory regardless
     of clip length; a Transformer KV cache would grow O(T)."

We measure peak VRAM and per-step latency for two paths on the SAME model:

  parallel_forward(T): the offline scan that allocates intermediate
                      (B, T, d_model) tensors for every layer.

  streaming_step():    stream_begin + stream_step T times. Only carries
                      the (B, d_inner, d_state) hidden state across
                      steps; intermediates are freed after each call.

We sweep T at {50, 250, 1000, 5000} latent steps -- 50 is one HumanML3D
clip; 5000 is ~17 minutes of motion, well past what offline inference
can hold in 4 GB of VRAM.

Run on CPU as well -- on a 3.5 GB local box the offline path will OOM
past T=1000 while streaming chugs along. That OOM-vs-no-OOM contrast IS
the figure.

Output JSON: runs/streaming_benchmark/results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from src.modules.motion.nn_models import TextToMotionSSM


class BenchCfg:
    """Headline config (configs/motion_ssm.yaml) but smaller batch=1 and
    unidirectional=True so streaming is supported. We measure architecture
    cost, not training cost, so this is fair vs a parallel-mode comparison.
    """

    def __init__(self) -> None:
        self.d_model = 384
        self.d_state = 64
        self.n_layers = 6
        self.motion_dim = 168
        self.text_embed_dim = 256
        self.max_motion_length = 20000  # support up to 5000 latent steps
        self.max_text_length = 64
        self.vocab_size = 1000
        self.bidirectional = False  # streaming requires this
        self.use_film = True
        self.use_sbert = False  # avoid SBERT download in benchmark
        self.sbert_model = "all-MiniLM-L6-v2"
        self.freeze_sbert = True
        self.gradient_checkpointing = False
        self.arch = "independent"
        self.rvq_latent_dim = 128
        self.rvq_n_codebooks = 6
        self.rvq_codebook_size = 512
        self.rvq_down_t = 4


def reset_peak_mem(device: torch.device) -> None:
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    # CPU: psutil RSS is the closest analogue
    import psutil  # type: ignore

    return psutil.Process().memory_info().rss / 1e6


def bench_parallel(model: TextToMotionSSM, latent_len: int, device: torch.device) -> dict:
    """Run the offline parallel forward once at the given latent length.

    Returns peak memory and wall-clock latency. Raises if OOM.
    """
    reset_peak_mem(device)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    motion_length = latent_len * model.config.rvq_down_t
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            _ = model(tokens, motion_length=motion_length)
        elapsed = time.perf_counter() - t0

        return {
            "ok": True,
            "elapsed_s": elapsed,
            "peak_mem_mb": peak_mem_mb(device),
            "throughput_steps_per_s": latent_len / elapsed,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
        return {"ok": False, "error": str(exc)[:200]}


def bench_streaming(model: TextToMotionSSM, latent_len: int, device: torch.device) -> dict:
    """Run T stream_step calls, measure aggregate cost + per-step memory.

    The peak-memory number is the high-watermark across all steps; it
    should stay essentially flat as latent_len grows (the O(1) claim).
    """
    reset_peak_mem(device)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    per_step_peak: list[float] = []

    try:
        with torch.no_grad():
            state = model.stream_begin(tokens)

            for _ in range(latent_len):
                reset_peak_mem(device)
                _, _, state = model.stream_step(state)
                per_step_peak.append(peak_mem_mb(device))
        elapsed = time.perf_counter() - t0

        return {
            "ok": True,
            "elapsed_s": elapsed,
            "peak_mem_mb": max(per_step_peak),
            "median_step_mem_mb": sorted(per_step_peak)[len(per_step_peak) // 2],
            "throughput_steps_per_s": latent_len / elapsed,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
        return {"ok": False, "error": str(exc)[:200]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--ts", type=int, nargs="+", default=[50, 250, 1000, 5000],
        help="latent-step counts to benchmark",
    )
    p.add_argument("--output", default="runs/streaming_benchmark/results.json")
    args = p.parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    print(f"Device: {device}")
    cfg = BenchCfg()
    model = TextToMotionSSM(cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d_model={cfg.d_model} n_layers={cfg.n_layers} params={n_params / 1e6:.1f}M")
    print(f"Sweeping latent_len in {args.ts}")
    print()
    print(
        f"{'T':>6}  {'parallel_mem':>14}  {'parallel_s':>11}  "
        f"{'stream_mem':>12}  {'stream_s':>10}  {'mem_ratio':>10}"
    )
    print("-" * 80)
    results = []

    for T in args.ts:
        par = bench_parallel(model, T, device)
        stream = bench_streaming(model, T, device)
        row: dict = {"latent_len": T, "parallel": par, "streaming": stream}

        if par.get("ok") and stream.get("ok"):
            ratio = par["peak_mem_mb"] / stream["peak_mem_mb"]
            row["mem_ratio_parallel_over_stream"] = ratio
            print(
                f"{T:>6}  {par['peak_mem_mb']:>11.1f} MB  "
                f"{par['elapsed_s']:>9.2f}s  "
                f"{stream['peak_mem_mb']:>9.1f} MB  "
                f"{stream['elapsed_s']:>8.2f}s  "
                f"{ratio:>9.2f}x"
            )
        else:
            par_str = (
                f"{par.get('peak_mem_mb', 0):>11.1f} MB"
                if par.get("ok") else "        OOM "
            )
            stream_str = (
                f"{stream.get('peak_mem_mb', 0):>9.1f} MB"
                if stream.get("ok") else "      OOM "
            )
            print(f"{T:>6}  {par_str}  {'-':>10}  {stream_str}  {'-':>9}  {'-':>9}")
        results.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": device.type, "rows": results}, indent=2))
    print()
    print(f"Wrote {out}")
    # Headline claim: streaming peak memory should stay roughly constant.
    stream_mems = [
        r["streaming"]["peak_mem_mb"] for r in results if r["streaming"].get("ok")
    ]

    if len(stream_mems) >= 2:
        spread = max(stream_mems) - min(stream_mems)
        ref = min(stream_mems)
        rel = spread / max(ref, 1e-6)
        print()
        print(
            f"Streaming memory spread across T: {spread:.1f} MB "
            f"({rel * 100:.1f}% of min={ref:.1f} MB)"
        )

        if rel < 0.5:
            print("VERDICT: streaming memory ~constant across T -- O(1) claim supported.")
        else:
            print("WARNING: streaming memory grew >50% across T -- investigate before publishing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
