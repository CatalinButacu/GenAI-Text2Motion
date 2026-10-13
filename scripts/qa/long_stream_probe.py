# .\.venv\Scripts\python.exe scripts/qa/long_stream_probe.py --backbone mamba --ckpt artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.run_log import log_metrics, start_run
from text2motion.evaluation.metrics import (
    compute_guo_text_embeddings,
    motion_statistics,
    r_precision,
)
from text2motion.generation.contracts import Backbone, StudioSamplingConfig
from text2motion.generation.nn_model import seeded_sampling_uniforms
from text2motion.motion.representation import recover_skeleton

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryPolicy:
    name: str
    anchor_every: int
    anchor_context: int


POLICIES = (
    MemoryPolicy("full_state", 0, 0),
    MemoryPolicy("reset_32_keep_16", 32, 16),
    MemoryPolicy("reset_8_keep_16", 8, 16),
    MemoryPolicy("reset_4_keep_16", 4, 16),
    MemoryPolicy("queue_keep_16", 1, 16),
    MemoryPolicy("queue_keep_47", 1, 47),
)
STATISTIC_KEYS = (
    "foot_skate_m_per_s",
    "mean_acceleration_cm_per_frame2",
    "mean_jerk_cm_per_frame3",
    "root_teleports",
)


def test_prompts(resources, count: int) -> list[tuple[str, list[str]]]:
    clip_ids = resources.clip_ids("test")
    random.Random(0).shuffle(clip_ids)
    prompts: list[tuple[str, list[str]]] = []
    for clip_id in clip_ids:
        text_path = resources.annotation_dir / f"{clip_id}.txt"
        if not text_path.exists():
            continue
        fields = text_path.read_text(encoding="utf-8").splitlines()[0].split("#")
        prompts.append((fields[0].strip(), fields[1].split(" ")))
        if len(prompts) == count:
            return prompts
    raise ValueError(f"only {len(prompts)} captioned test clips found, {count} requested")


def policy_fits(policy: MemoryPolicy, positions: int | None, prefix: int, tokens: int) -> bool:
    if positions is None:
        return True
    available = positions - prefix
    if policy.anchor_every == 0:
        return tokens <= available
    return policy.anchor_context + policy.anchor_every <= available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument("--backbone", required=True, type=Backbone)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument(
        "--tokenizer_ckpt",
        type=Path,
        default=Path("artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"),
    )
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment_tokens", type=int, default=49)
    args = parser.parse_args()

    app = ApplicationBootstrap.from_config(args.config)
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    pipeline = app.load_text_to_motion_generator(
        args.backbone, args.ckpt, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
    ).eval()
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    run_dir = start_run(f"long_stream_{args.backbone.value}", app.config, app.config.paths.logs_dir)

    prompts = test_prompts(resources, args.prompts)
    captions = [caption for caption, _ in prompts]
    text_inputs = [resources.build_guo_text_inputs(tokens) for _, tokens in prompts]
    text_embeddings = compute_guo_text_embeddings(
        resources.text_embedder, text_inputs, resources.device
    )
    tokens_total = args.segments * args.segment_tokens
    token_generator = pipeline.token_generator
    sampling = StudioSamplingConfig()
    segment_frames = args.segment_tokens * pipeline.downsample
    report: dict[str, list[dict[str, float]]] = {}

    with torch.no_grad():
        text_condition = pipeline.text_encoder(captions)
        prefix_length = token_generator.project_text_prefix(text_condition).size(1)
        uniforms = seeded_sampling_uniforms(
            range(len(captions)), tokens_total, token_generator.cfg.num_codebooks, app.device
        )
        for policy in POLICIES:
            positions = token_generator.backbone.max_positions
            if positions is not None and not isinstance(positions, int):
                raise TypeError(f"max_positions must be an int or None, got {type(positions)}")
            if not policy_fits(policy, positions, prefix_length, tokens_total):
                logger.info(f"{policy.name}: skipped, {positions} learned positions are too few")
                continue
            started = time.perf_counter()
            steps = list(
                token_generator.stream_token_indices(
                    text_condition,
                    tokens_total,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    cfg_scale=sampling.cfg_scale,
                    uniforms=uniforms,
                    anchor_every=policy.anchor_every,
                    anchor_context=policy.anchor_context,
                )
            )
            milliseconds_per_token = (time.perf_counter() - started) * 1000 / tokens_total
            indices = torch.stack(steps, dim=1)
            decoded = tokenizer.tokenizer_model.decode(indices).cpu().numpy()
            rows: list[dict[str, float]] = []
            for segment in range(args.segments):
                window = slice(segment * segment_frames, (segment + 1) * segment_frames)
                features = [np.asarray(resources.denormalize(clip[window])) for clip in decoded]
                statistics = [motion_statistics(recover_skeleton(feature)) for feature in features]
                motion_embeddings = evaluator.compute_motion_embeddings(features)
                top_hits = r_precision(text_embeddings, motion_embeddings, pool_size=len(captions))
                own_distance = np.linalg.norm(motion_embeddings - text_embeddings, axis=-1)
                row = {
                    "segment": segment + 1,
                    "seconds_from": segment * segment_frames / 20,
                    "text_distance": float(own_distance.mean()),
                    "r_top1": float(top_hits[0]),
                    "r_top3": float(top_hits[2]),
                    "ms_per_token": milliseconds_per_token,
                }
                for key in STATISTIC_KEYS:
                    row[key] = float(np.mean([statistic[key] for statistic in statistics]))
                rows.append(row)
                log_metrics(run_dir, {"policy": policy.name, **row})
                logger.info(
                    f"{policy.name:18s} segment {segment + 1}  text distance {row['text_distance']:.3f}  "
                    f"R@1 {row['r_top1']:.3f}  R@3 {row['r_top3']:.3f}  "
                    f"foot slide {row['foot_skate_m_per_s']:.3f}  "
                    f"accel {row['mean_acceleration_cm_per_frame2']:.3f}  "
                    f"{milliseconds_per_token:.1f} ms/token"
                )
            report[policy.name] = rows

    out_path = app.config.paths.outputs_dir / "qa" / f"long_stream_{args.backbone.value}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "backbone": args.backbone.value,
        "checkpoint": str(args.ckpt),
        "prompts": len(captions),
        "segment_tokens": args.segment_tokens,
        "policies": report,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
