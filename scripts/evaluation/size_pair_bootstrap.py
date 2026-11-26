# python scripts/evaluation/size_pair_bootstrap.py --backbone mamba --small 256 --large 320 --resamples 1000
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from text2motion.app.runtime.context import ApplicationBootstrap
from text2motion.evaluation.metrics import fid, paired_fid_delta
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.motion.contracts import Split

SEED = 2026
TOKENIZER = Path("runs/checkpoints/tokenizer/fsq_g8_v1024.pt")
SAMPLING = SamplingConfig(temperature=1.0, cfg_scale=6.0)


def width_config(width: int) -> str:
    return f"configs/generator/size_sweep/width_{width}.yaml"


def generated_embeddings(
    width: int, backbone: Backbone, pattern: str, evaluator
) -> tuple[list, np.ndarray, np.ndarray]:
    app = ApplicationBootstrap.from_config(width_config(width), seed=SEED)
    tokenizer = app.models.load_tokenizer(TOKENIZER)
    checkpoint = app.checkpoint_directory("generator") / pattern.format(
        backbone=backbone.value, width=width
    )
    generator = app.models.load_text_to_motion_generator(
        backbone, checkpoint, tokenizer, tokenizer_checkpoint=TOKENIZER
    )
    evaluation = evaluator.generate_evaluation_set(
        generator, Split.VALIDATION, None, SAMPLING, batch_size=32
    )
    reference = evaluator.compute_motion_embeddings(evaluation.reference)
    generated = evaluator.compute_motion_embeddings(evaluation.generated)
    del generator
    torch.cuda.empty_cache()
    return list(evaluation.captions), reference, generated


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=Backbone, choices=list(Backbone), required=True)
    parser.add_argument("--small", type=int, default=256)
    parser.add_argument("--large", type=int, default=320)
    parser.add_argument("--checkpoint_pattern", default="generator_{backbone}_w{width}_sweep.pt")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/outputs/qa"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    base = ApplicationBootstrap.from_config(width_config(arguments.small), seed=SEED)
    evaluator = base.create_humanml3d_generation_evaluator()
    small = generated_embeddings(
        arguments.small, arguments.backbone, arguments.checkpoint_pattern, evaluator
    )
    large = generated_embeddings(
        arguments.large, arguments.backbone, arguments.checkpoint_pattern, evaluator
    )
    if small[0] != large[0] or not np.array_equal(small[1], large[1]):
        raise RuntimeError("the two sizes were not scored on the same clips in the same order")
    reference = small[1]
    delta = paired_fid_delta(
        reference, small[2], large[2], resamples=arguments.resamples, seed=SEED
    )
    report = {
        "backbone": arguments.backbone.value,
        "split": Split.VALIDATION.value,
        "clips": len(reference),
        "small_width": arguments.small,
        "large_width": arguments.large,
        "small_fid": float(fid(reference, small[2])),
        "large_fid": float(fid(reference, large[2])),
        "fid_small_minus_large": delta,
    }
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    out = arguments.out_dir / f"size_pair_bootstrap_{arguments.backbone.value}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
