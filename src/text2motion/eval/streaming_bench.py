from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.model.generator import MotionGenerator
from text2motion.shared.config import load_config


def state_bytes(state) -> int:
    total = 0
    for layer_state in state:
        if layer_state is None:
            continue
        for tensor in layer_state:
            total += tensor.numel() * tensor.element_size()
    return total


@torch.no_grad()
def bench_backbone(
    generator: MotionGenerator, horizons: list[int], device: str
) -> list[dict[str, float]]:
    cfg = generator.cfg
    cuda = device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    state = generator.backbone.init_state(1, torch.device(device))
    text = torch.randn(1, cfg.text_prefix_len, cfg.d_text, device=device)
    prefix = generator.text_prefix(text)
    for position in range(prefix.size(1)):
        h, state = generator.backbone.step(prefix[:, position], state)

    rows: list[dict[str, float]] = []
    step_ms: list[float] = []
    tokens = torch.zeros(1, cfg.num_codebooks, dtype=torch.long, device=device)
    for step in range(1, max(horizons) + 1):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        h, state = generator.backbone.step(generator.embed_tokens(tokens), state)
        tokens = generator.logits(h).argmax(-1)  # greedy: benchmark measures the step, not quality
        if generator.end_id is not None:
            tokens = tokens.clamp(max=cfg.codebook_size - 1)  # END has no embedding-decode meaning
        if cuda:
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000)

        if step in horizons:
            segment = step_ms[-min(len(step_ms), 64) :]  # median of the trailing window
            rows.append(
                {
                    "horizon": step,
                    "ms_per_step": statistics.median(segment),
                    "state_bytes": state_bytes(state),
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming latency/state-size benchmark.")
    parser.add_argument("--config", default="configs/generator/final100m.yaml")
    parser.add_argument("--horizons", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/streaming_bench.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, list[dict[str, float]]] = {}

    for backbone in ["transformer", "mamba"]:
        n_layers = cfg.generator.mamba_n_layers if backbone == "mamba" else cfg.generator.n_layers
        bench_seq_len = max(args.horizons) + cfg.generator.text_prefix_len + 8
        gen_cfg = replace(
            cfg.generator, backbone=backbone, n_layers=n_layers, max_seq_len=bench_seq_len
        )
        torch.manual_seed(cfg.seed)
        generator = MotionGenerator(gen_cfg).to(device).eval()
        results[backbone] = bench_backbone(generator, sorted(args.horizons), device)
        params = sum(p.numel() for p in generator.parameters())
        print(f"\n{backbone} ({params:,} params, {n_layers} layers, device {device})")
        print(f"{'horizon':>8} {'ms/step':>9} {'state':>12} {'cuda peak':>12}")
        for row in results[backbone]:
            print(
                f"{row['horizon']:>8} {row['ms_per_step']:>9.2f} "
                f"{row['state_bytes'] / 1e6:>10.2f}MB {row['cuda_peak_bytes'] / 1e6:>10.1f}MB"
            )
        del generator
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": args.config, "results": results}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
