from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.schema import load_dataclass
from text2motion.evaluation.plausibility import motion_statistics
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.motion.representation import recover_skeleton
from text2motion.studio.avatar import build_smplx_body_model, solve_smplx_pose_from_joints
from text2motion.studio.config import StudioConfig


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


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
    parser.add_argument("--clips", type=int, default=64)
    parser.add_argument("--mesh_clips", type=int, default=12)
    parser.add_argument("--out", default="artifacts/outputs/qa/motion_naturalness.json")
    args = parser.parse_args()

    app = ApplicationBootstrap.from_config(args.config)
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    pipeline = app.load_text_to_motion_generator(
        args.backbone, args.ckpt, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
    ).eval()
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    clip_ids = resources.clip_ids("test")
    random.Random(0).shuffle(clip_ids)
    vectors_dir = resources.motion_dataset_dir / "new_joint_vecs"
    sampling = SamplingConfig(temperature=1.0, top_p=0.9, cfg_scale=6.0)
    downsample = pipeline.downsample
    stages: dict[str, list[dict[str, float]]] = {
        "ground_truth": [],
        "tokenizer_roundtrip": [],
        "generated": [],
        "generated_mesh": [],
    }
    fit_config = replace(
        load_dataclass(StudioConfig, "configs/studio.yml").fit,
        model_dir="data/smplx_models",
        gender="neutral",
    )
    used = 0
    with torch.no_grad():
        for clip_id in clip_ids:
            if used >= args.clips:
                break
            vector_path = vectors_dir / f"{clip_id}.npy"
            text_path = resources.annotation_dir / f"{clip_id}.txt"
            if not vector_path.exists() or not text_path.exists():
                continue
            raw = np.load(vector_path).astype(np.float32)
            frames = (min(len(raw), 196) // downsample) * downsample
            if frames < 40:
                continue
            raw = raw[:frames]
            caption = text_path.read_text(encoding="utf-8").splitlines()[0].split("#")[0].strip()
            stages["ground_truth"].append(motion_statistics(recover_skeleton(raw)))
            normalized = torch.from_numpy(resources.normalize(raw)).unsqueeze(0).to(app.device)
            network = tokenizer.tokenizer_model
            indices = network.encode(normalized)
            roundtrip = network.decode(indices)[0].cpu().numpy()
            stages["tokenizer_roundtrip"].append(
                motion_statistics(recover_skeleton(resources.denormalize(roundtrip)))
            )
            generated = pipeline.generate_batch([caption], frames // downsample, sampling)[0]
            joints = recover_skeleton(
                resources.denormalize(generated.motion.features.cpu().numpy())
            )
            stages["generated"].append(motion_statistics(joints))
            if used < args.mesh_clips:
                model = build_smplx_body_model(fit_config, len(joints), "cpu")
                result = solve_smplx_pose_from_joints(model, joints.astype(np.float32), "cpu")
                mesh = (
                    model(
                        global_orient=torch.tensor(result.global_orient),
                        body_pose=torch.tensor(result.body_pose),
                        transl=torch.tensor(result.transl),
                        betas=torch.zeros(len(joints), model.num_betas),
                    )
                    .joints[:, :22]
                    .numpy()
                )
                stages["generated_mesh"].append(motion_statistics(mesh))
            used += 1
    report = {stage: summarize(rows) for stage, rows in stages.items() if rows}
    keys = list(report["ground_truth"])
    print(f"{used} test clips (<= 196 frames), same captions and lengths for generation")
    print(f"{'stage':<22}" + "".join(f"{key[:24]:>26}" for key in keys))
    for stage, values in report.items():
        print(f"{stage:<22}" + "".join(f"{values[key]:>26.3f}" for key in keys))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
