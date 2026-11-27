# .\.venv\Scripts\python.exe scripts/qa/steering_jump.py --out docs/notes/steering_joint_jump.csv
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from text2motion.generation.contracts import Backbone, GeneratorConfig
from text2motion.generation.decoder import StreamingMotionDecoder
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.steering import SteeringMode, steer_stream
from text2motion.motion.recovery import recover_skeleton
from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.network.finite_scalar import FiniteScalarQuantizationTokenizer

TOKENIZER = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GENERATOR = GeneratorConfig(
    backbone=Backbone.MAMBA,
    d_model=32,
    n_layers=2,
    d_text=16,
    num_codebooks=2,
    codebook_size=16,
    max_seq_len=64,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
)
MODES = (SteeringMode.REANCHOR, SteeringMode.CROSSFADE)
CM_PER_M = 100.0


def peak_step_cm(joints: np.ndarray, start: int, end: int) -> float:
    step = np.linalg.norm(np.diff(joints[start - 1 : end + 1], axis=0), axis=-1)
    return float(step.max() * CM_PER_M)


def steered_joints(generator, decoder, old, new, args, mode, seed) -> np.ndarray:
    torch.manual_seed(seed)
    chunks = [
        chunk.squeeze(0)
        for _, chunk in steer_stream(
            generator,
            decoder,
            old,
            new,
            args.steps,
            args.switch,
            mode,
            blend_frames=args.blend,
        )
    ]
    return recover_skeleton(torch.cat(chunks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--switch", type=int, default=12)
    parser.add_argument("--blend", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="docs/notes/steering_joint_jump.csv")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    generator = MotionTokenGenerator(GENERATOR).eval()
    decoder = StreamingMotionDecoder(
        FiniteScalarQuantizationTokenizer(TOKENIZER),
        TOKENIZER.downsample,
        torch.zeros(263),
        torch.ones(263),
        chunk_tokens=4,
    )
    texts = torch.randn(args.pairs, 2, 1, GENERATOR.d_text)
    start = args.switch * decoder.downsample
    end = start + args.blend
    rows = []
    for pair in range(args.pairs):
        row: dict[str, float] = {"pair": float(pair)}
        for mode in MODES:
            seed = args.seed + pair
            old, new = texts[pair, 0], texts[pair, 1]
            switched = steered_joints(generator, decoder, old, new, args, mode, seed)
            reference = steered_joints(generator, decoder, old, old, args, mode, seed)
            jump = peak_step_cm(switched, start, end)
            base = peak_step_cm(reference, start, end)
            row[f"{mode.value}_jump_cm"] = round(jump, 4)
            row[f"{mode.value}_reference_cm"] = round(base, 4)
            row[f"{mode.value}_excess_cm"] = round(jump - base, 4)
        rows.append(row)
    columns = [key for key in rows[0] if key != "pair"]
    summary = []
    for label, reducer in (
        ("mean", np.mean),
        ("p95", lambda values: np.percentile(values, 95)),
    ):
        summary.append(
            {"pair": label}
            | {key: round(float(reducer([r[key] for r in rows])), 4) for key in columns}
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", *columns])
        writer.writeheader()
        writer.writerows(rows + summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
