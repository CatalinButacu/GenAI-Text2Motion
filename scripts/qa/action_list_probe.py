# .\.venv\Scripts\python.exe scripts/qa/action_list_probe.py --backbone mamba --ckpt artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.run_log import log_metrics, start_run
from text2motion.evaluation.metrics import (
    compute_guo_text_embeddings,
    motion_statistics,
    own_caption_ranks,
    r_precision,
)
from text2motion.generation.contracts import Backbone, StudioSamplingConfig
from text2motion.generation.nn_model import seeded_sampling_uniforms
from text2motion.motion.representation import recover_skeleton

logger = logging.getLogger(__name__)
SPLIT_SEED = 2026
ACTIONS = 3
JOINER = ", then "
BOUNDARY_HALF_FRAMES = 10
METHODS = ("joined_caption", "action_list")


def captioned_test_clips(resources, count: int) -> list[tuple[str, str, list[str]]]:
    clip_ids = sorted(resources.clip_ids("test"))
    random.Random(SPLIT_SEED).shuffle(clip_ids)
    clips: list[tuple[str, str, list[str]]] = []
    for clip_id in clip_ids:
        text_path = resources.annotation_dir / f"{clip_id}.txt"
        if not text_path.exists():
            continue
        fields = text_path.read_text(encoding="utf-8").splitlines()[0].split("#")
        clips.append((clip_id, fields[0].strip(), fields[1].split(" ")))
        if len(clips) == count:
            return clips
    raise ValueError(f"only {len(clips)} captioned test clips found, {count} requested")


def mean_statistic(joint_windows: list[np.ndarray], key: str) -> float:
    return float(np.mean([motion_statistics(window)[key] for window in joint_windows]))


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
    parser.add_argument("--sequences", type=int, default=96)
    parser.add_argument("--action_tokens", type=int, default=48)
    parser.add_argument("--anchor_every", type=int, default=8)
    parser.add_argument("--anchor_context", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    if args.action_tokens % args.anchor_every:
        raise ValueError(
            "action_tokens must be a multiple of anchor_every, so a reset is on each boundary"
        )

    app = ApplicationBootstrap.from_config(args.config)
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    pipeline = app.load_text_to_motion_generator(
        args.backbone, args.ckpt, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
    ).eval()
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    run_dir = start_run(
        f"action_list_{args.backbone.value}", app.config, app.config.paths.logs_dir, vars(args)
    )

    clips = captioned_test_clips(resources, args.sequences * ACTIONS)
    sequences = [clips[start : start + ACTIONS] for start in range(0, len(clips), ACTIONS)]
    action_text_embeddings = [
        compute_guo_text_embeddings(
            resources.text_embedder,
            [resources.build_guo_text_inputs(sequence[action][2]) for sequence in sequences],
            resources.device,
        )
        for action in range(ACTIONS)
    ]
    token_generator = pipeline.token_generator
    sampling = StudioSamplingConfig()
    tokens_total = ACTIONS * args.action_tokens
    action_frames = args.action_tokens * pipeline.downsample
    features: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}

    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            group = sequences[start : start + args.batch_size]
            joined = pipeline.text_encoder(
                [JOINER.join(caption for _, caption, _ in sequence) for sequence in group]
            )
            for_action = [
                pipeline.text_encoder([sequence[action][1] for sequence in group])
                for action in range(ACTIONS)
            ]
            uniforms = seeded_sampling_uniforms(
                range(start, start + len(group)),
                tokens_total,
                token_generator.cfg.num_codebooks,
                app.device,
            )
            streams = {
                "joined_caption": (joined, None),
                "action_list": (
                    for_action[0],
                    lambda index, texts=for_action: texts[
                        min(index // args.action_tokens, ACTIONS - 1)
                    ],
                ),
            }
            for method, (first_text, text_at) in streams.items():
                steps = list(
                    token_generator.stream_token_indices(
                        first_text,
                        tokens_total,
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        cfg_scale=sampling.cfg_scale,
                        uniforms=uniforms,
                        anchor_every=args.anchor_every,
                        anchor_context=args.anchor_context,
                        text_at=text_at,
                    )
                )
                decoded = tokenizer.tokenizer_model.decode(torch.stack(steps, dim=1)).cpu().numpy()
                features[method].extend(np.asarray(resources.denormalize(clip)) for clip in decoded)
            logger.info(f"generated sequences {start + 1} to {start + len(group)}")

    report: dict[str, dict] = {}
    hits: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    distances: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    for method in METHODS:
        joints = [recover_skeleton(feature) for feature in features[method]]
        rows = []
        for action in range(ACTIONS):
            window = slice(action * action_frames, (action + 1) * action_frames)
            motion_embeddings = evaluator.compute_motion_embeddings(
                [feature[window] for feature in features[method]]
            )
            text_embeddings = action_text_embeddings[action]
            top_hits = r_precision(text_embeddings, motion_embeddings)
            distance = np.linalg.norm(motion_embeddings - text_embeddings, axis=-1)
            ranks = own_caption_ranks(text_embeddings, motion_embeddings, np.arange(len(distance)))
            hits[method].append(ranks == 0)
            distances[method].append(distance)
            row = {
                "action": action + 1,
                "text_distance": float(distance.mean()),
                "r_top1": float(top_hits[0]),
                "r_top3": float(top_hits[2]),
                "foot_skate_m_per_s": mean_statistic(
                    [clip[window] for clip in joints], "foot_skate_m_per_s"
                ),
            }
            rows.append(row)
            log_metrics(run_dir, {"method": method, **row})
            logger.info(
                f"{method:15s} action {action + 1}  text distance {row['text_distance']:.3f}  "
                f"R@1 {row['r_top1']:.3f}  R@3 {row['r_top3']:.3f}  "
                f"foot slide {row['foot_skate_m_per_s']:.3f}"
            )
        boundary_windows = [
            clip[boundary - BOUNDARY_HALF_FRAMES : boundary + BOUNDARY_HALF_FRAMES]
            for clip in joints
            for boundary in range(action_frames, ACTIONS * action_frames, action_frames)
        ]
        middle_windows = [
            clip[middle - BOUNDARY_HALF_FRAMES : middle + BOUNDARY_HALF_FRAMES]
            for clip in joints
            for middle in range(action_frames // 2, ACTIONS * action_frames, action_frames)
        ]
        smoothness = {
            f"{place}_{key}": mean_statistic(windows, key)
            for place, windows in (("boundary", boundary_windows), ("middle", middle_windows))
            for key in (
                "mean_acceleration_cm_per_frame2",
                "mean_jerk_cm_per_frame3",
                "root_teleports",
            )
        }
        log_metrics(run_dir, {"method": method, **smoothness})
        logger.info(f"{method:15s} {smoothness}")
        report[method] = {"actions": rows, "smoothness": smoothness}

    paired = []
    for action in range(ACTIONS):
        joined_hit = hits["joined_caption"][action]
        list_hit = hits["action_list"][action]
        interval = stats.bootstrap(
            (list_hit.astype(float),),
            np.mean,
            method="percentile",
            n_resamples=2000,
            rng=np.random.default_rng(0),
        ).confidence_interval
        only_list = int((list_hit & ~joined_hit).sum())
        only_joined = int((joined_hit & ~list_hit).sum())
        discordant = only_list + only_joined
        list_distance = distances["action_list"][action]
        joined_distance = distances["joined_caption"][action]
        row = {
            "action": action + 1,
            "action_list_r_top1_ci_low": float(interval.low),
            "action_list_r_top1_ci_high": float(interval.high),
            "only_action_list_correct": only_list,
            "only_joined_caption_correct": only_joined,
            "mcnemar_exact_p": float(stats.binomtest(only_list, discordant, 0.5).pvalue)
            if discordant
            else 1.0,
            "wilcoxon_distance_p": float(
                np.asarray(stats.wilcoxon(list_distance, joined_distance))[1]
            ),
        }
        paired.append(row)
        log_metrics(run_dir, {"method": "paired", **row})
        logger.info(f"paired          {row}")

    out_path = (
        app.config.paths.outputs_dir
        / "qa"
        / f"action_list_{args.backbone.value}_keep{args.anchor_context}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "backbone": args.backbone.value,
        "checkpoint": str(args.ckpt),
        "split_seed": SPLIT_SEED,
        "sequences": [[clip_id for clip_id, _, _ in sequence] for sequence in sequences],
        "action_tokens": args.action_tokens,
        "anchor_every": args.anchor_every,
        "anchor_context": args.anchor_context,
        "methods": report,
        "paired": paired,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
