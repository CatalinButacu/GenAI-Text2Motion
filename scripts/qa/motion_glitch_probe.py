from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.generation.contracts import Backbone
from text2motion.motion.representation import FPS, StreamingSkeletonRecovery, recover_skeleton
from text2motion.streaming.decoder import StreamingMotionDecoder

PROMPTS = (
    "a person runs forward",
    "a person walks in a circle",
    "a person jumps up and down",
    "a person dances energetically",
    "a person waves with the right hand",
)
FEET = (7, 8, 10, 11)
TELEPORT_METERS_PER_FRAME = 0.25


def streamed_joints(pipeline, decoder, text_emb, steps, cfg_scale, anchor_every, anchor_context):
    token_source = pipeline.token_generator.stream_token_indices(
        text_emb,
        steps,
        cfg_scale=cfg_scale,
        anchor_every=anchor_every,
        anchor_context=anchor_context,
    )
    recovery = StreamingSkeletonRecovery()
    chunks = []
    features = []
    for chunk in decoder.stream_tokens(token_source):
        frame_features = chunk.squeeze(0).cpu().numpy()
        features.append(frame_features)
        chunks.append(recovery(frame_features))
    return np.concatenate(chunks), np.concatenate(features)


def glitch_report(joints: np.ndarray, chunk_frames: int, anchor_frames: int, limit: float) -> dict:
    velocity = np.diff(joints, axis=0)
    acceleration = np.diff(velocity, axis=0)
    acceleration_peak = np.linalg.norm(acceleration, axis=-1).max(axis=-1)
    spike_frames = np.nonzero(acceleration_peak > limit)[0] + 1
    root_step = np.linalg.norm(velocity[:, 0, [0, 2]], axis=-1)
    teleports = np.nonzero(root_step > TELEPORT_METERS_PER_FRAME)[0] + 1
    lowest_foot = joints[:, FEET, 1].min(axis=-1)
    floor = np.percentile(lowest_foot, 5)
    grounded = joints[:, FEET, 1] < floor + 0.05
    foot_speed = np.linalg.norm(velocity[:, FEET][..., [0, 2]], axis=-1)
    skating = float((foot_speed * grounded[1:]).sum() / max(1, grounded[1:].sum()) * FPS)
    on_chunk_seam = int(np.isin(spike_frames % chunk_frames, [0]).sum())
    on_anchor = int(np.isin(spike_frames % anchor_frames, [0, 1, 2]).sum()) if anchor_frames else 0
    return {
        "frames": int(len(joints)),
        "nan_frames": int(np.isnan(joints).any(axis=(1, 2)).sum()),
        "acceleration_spikes": int(len(spike_frames)),
        "spikes_on_chunk_seams": on_chunk_seam,
        "expected_on_chunk_seams_if_random": round(len(spike_frames) / chunk_frames, 2),
        "spikes_near_anchor_rebuilds": on_anchor,
        "expected_near_anchor_if_random": round(len(spike_frames) * 3 / anchor_frames, 2)
        if anchor_frames
        else 0,
        "root_teleports": int(len(teleports)),
        "max_root_step_m": round(float(root_step.max()), 3),
        "foot_skate_m_per_s": round(skating, 3),
        "spike_frames": spike_frames[:40].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument("--backbone", default="mamba", type=Backbone)
    parser.add_argument(
        "--ckpt", default="artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt"
    )
    parser.add_argument(
        "--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
    )
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--spike_m_per_frame2", type=float, default=0.08)
    parser.add_argument("--out", default="artifacts/outputs/qa/motion_glitch_probe.json")
    args = parser.parse_args()

    app = ApplicationBootstrap.from_config(args.config)
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    pipeline = app.load_text_to_motion_generator(
        args.backbone, args.ckpt, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
    ).eval()
    scaler = app.create_motion_repository().load_scaler()
    mean = torch.from_numpy(scaler.mean).to(app.device)
    std = torch.from_numpy(scaler.std).to(app.device)
    decoder = StreamingMotionDecoder(
        tokenizer.network, downsample=tokenizer.downsample, mean=mean, std=std
    )
    chunk_frames = decoder.chunk_tokens * decoder.downsample
    service = app.config.service
    modes = [("plain", 0), ("anchored", service.anchor_every_tokens or 32)]
    print(
        f"decoder chunk {chunk_frames} frames, left {decoder.left_context} / lookahead {decoder.lookahead} tokens; "
        f"anchor every {modes[1][1]} tokens over {service.anchor_context_tokens} tokens of context"
    )
    report = {}
    with torch.no_grad():
        for mode, anchor_every in modes:
            position_limit = pipeline.token_generator.backbone.max_positions
            if anchor_every == 0 and position_limit:
                steps = min(args.steps, position_limit - 1)
            else:
                steps = args.steps
            anchor_frames = anchor_every * decoder.downsample
            report[mode] = {}
            for prompt in PROMPTS:
                torch.manual_seed(0)
                text_emb = pipeline.text_encoder([prompt])
                joints, features = streamed_joints(
                    pipeline,
                    decoder,
                    text_emb,
                    steps,
                    args.cfg_scale,
                    anchor_every,
                    service.anchor_context_tokens,
                )
                whole = recover_skeleton(features)
                streaming_error = float(np.abs(whole - joints).max())
                row = glitch_report(joints, chunk_frames, anchor_frames, args.spike_m_per_frame2)
                row["streamed_vs_whole_recovery_max_m"] = round(streaming_error, 4)
                report[mode][prompt] = row
                print(
                    f"{mode:<9} {prompt:<36} spikes {row['acceleration_spikes']:>3} "
                    f"(seams {row['spikes_on_chunk_seams']}/{row['expected_on_chunk_seams_if_random']}, "
                    f"anchor {row['spikes_near_anchor_rebuilds']}/{row['expected_near_anchor_if_random']}) "
                    f"teleports {row['root_teleports']} max step {row['max_root_step_m']} m "
                    f"skate {row['foot_skate_m_per_s']} m/s nan {row['nan_frames']} "
                    f"stream-vs-whole {row['streamed_vs_whole_recovery_max_m']} m"
                )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
