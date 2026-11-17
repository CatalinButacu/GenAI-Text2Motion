# python scripts/render/render_clip.py --prompt "a person walks forward" --out artifacts/outputs/figures/demo_clip.png
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.cli_requests import ModelServiceRequest
from text2motion.app.serving import load_inference_pipeline
from text2motion.generation.contracts import Backbone
from text2motion.motion.representation import DIM, FPS, kinematic_bones, recover_skeleton
from text2motion.motion.storage import sha256_file

REVIEW_FRAMES = tuple(range(176, 289))


def review_tpose(model) -> np.ndarray:
    with torch.no_grad():
        zeros = {
            "global_orient": torch.zeros((1, 3)),
            "transl": torch.zeros((1, 3)),
            "betas": torch.zeros((1, 10)),
            "body_pose": torch.zeros((1, 63)),
        }
        rest = model(**zeros).joints[:, :22].numpy().copy()
        for shoulder, wrist in ((16, 20), (17, 21)):
            delta = rest[0, wrist] - rest[0, shoulder]
            zeros["body_pose"][0, (shoulder - 1) * 3 + 2] = float(
                np.arctan2(-delta[1], abs(delta[0])) * np.sign(delta[0])
            )
        rest = model(**zeros).joints[:, :22].numpy().copy()
    rest[..., 1] -= rest[..., 1].min()
    if np.max(np.abs(rest[0, (20, 21), 1] - rest[0, (16, 17), 1])) > 0.01:
        raise ValueError("neutral SMPL-X calibration did not produce horizontal T-pose arms")
    if rest[0, 15, 1] <= rest[0, (7, 8, 10, 11), 1].max():
        raise ValueError("neutral SMPL-X T-pose failed Y-up calibration")
    return rest


def review_frame(renderer, case: dict, frame: int):
    from PIL import Image, ImageDraw, ImageFont

    renderer.scene.current_frame_id = frame
    image = renderer.get_frame()
    result = Image.new("RGB", (image.width, image.height + 150), "white")
    result.paste(image, (0, 150))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=16)
    lines = [
        f"{case['case']} | {case['backbone']} | t={frame / FPS:.2f}s | A LEFT / B RIGHT",
        *textwrap.wrap("OLD: " + case["old_caption"], width=90),
        *textwrap.wrap("NEW: " + case["new_caption"], width=90),
    ]
    if len(lines) > 7:
        raise ValueError("review captions exceed the explicit image header budget")
    draw.multiline_text((8, 8), "\n".join(lines), fill="black", font=font, spacing=2)
    return result


def render_review(a: argparse.Namespace) -> None:
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.scene.camera import ViewerCamera
    from aitviewer.scene.scene import Scene
    from PIL import Image

    from text2motion.studio.config import SmplxFitConfig
    from text2motion.studio.fitting import build_smplx_body_model
    from text2motion.studio.scene import create_skeleton_sequence_node

    started = time.perf_counter()
    manifest = json.loads(a.review.read_text(encoding="utf-8"))
    if a.review_out.exists():
        raise FileExistsError("review rendering requires a fresh output directory")
    a.review_out.mkdir(parents=True)
    renderer = HeadlessRenderer(size=(800, 480), samples=0)
    renderer.auto_set_floor = False
    renderer.auto_set_camera_target = False
    camera = ViewerCamera(orthographic=1.2, width=800, height=480)
    renderer.set_temp_camera(camera)
    ready, outputs = {}, []
    try:
        if renderer.scene is None:
            raise RuntimeError("headless viewer did not initialize a scene")
        scene: Scene = renderer.scene
        scene.floor.enabled = True
        model = build_smplx_body_model(SmplxFitConfig(model_dir=a.model_dir), 1, "cpu")
        rest = review_tpose(model)
        node = create_skeleton_sequence_node(rest, radius=0.025)
        node.color = (0.05, 0.15, 0.65, 1.0)
        scene.add(node)
        camera.position = np.array([0.0, 1.5, 4.0])
        camera.target = np.array([0.0, 1.0, 0.0])
        camera.update_matrices(800, 480)
        renderer.save_frame(str(a.review_out / "readiness_tpose.png"))
        scene.remove(node)
        del model
        source = json.loads(Path(manifest["source"]).read_text(encoding="utf-8"))
        sample = source["cases"]["mamba"]["0"]["16"]["report"]["pairs"][0]["left"]
        known_path = (
            Path("data") / "HumanML3D_official" / "new_joint_vecs" / f"{sample['clip_id']}.npy"
        )
        known = recover_skeleton(np.load(known_path, allow_pickle=False))
        node = create_skeleton_sequence_node(known, radius=0.025)
        node.color = (0.05, 0.15, 0.65, 1.0)
        scene.add(node)
        scene.current_frame_id = 0
        renderer.save_frame(str(a.review_out / "readiness_known_clip.png"))
        scene.remove(node)
        ready = {
            "renderer": renderer.ctx.info["GL_RENDERER"],
            "known_clip": str(known_path),
            "known_clip_sha256": sha256_file(known_path),
            "tpose_y_up_passed": True,
        }
        print(f"REVIEW readiness passed: {ready['renderer']}", flush=True)
        for case in manifest["cases"]:
            nodes, clips = [], []
            for label in ("A", "B"):
                path = Path(case["panels"][label])
                if sha256_file(path) != case["panel_sha256"][label]:
                    raise ValueError(f"{case['case']}/{label}: archived joints changed")
                clip = np.load(path, allow_pickle=False)
                if clip.shape != (384, 22, 3) or not np.isfinite(clip).all():
                    raise ValueError("review panels require finite384-frame canonical joints")
                clips.append(clip.copy())
            review = np.stack(clips)[:, REVIEW_FRAMES]
            lo = review.min((0, 1, 2))
            hi = review.max((0, 1, 2))
            spacing = max(float(hi[0] - lo[0]), 1.5) + 1.0
            for clip, shift in zip(clips, (-spacing / 2, spacing / 2), strict=True):
                clip[..., 0] += shift
                node = create_skeleton_sequence_node(clip, radius=0.025)
                node.color = (0.05, 0.15, 0.65, 1.0)
                nodes.append(node)
                scene.add(node)
            merged = np.stack(clips)[:, REVIEW_FRAMES]
            lo, hi = merged.min((0, 1, 2)), merged.max((0, 1, 2))
            center = (lo + hi) / 2
            camera.target = center
            camera.position = center + np.array([1.5, 1.0, 8.0])
            camera.update_matrices(800, 480)
            points = merged.reshape(-1, 3)
            projected = np.column_stack((points, np.ones(len(points)))) @ camera.get_view_matrix().T
            camera.ortho_size = (
                max(
                    float(np.abs(projected[:, 1]).max()),
                    float(np.abs(projected[:, 0]).max()) * 480 / 800,
                )
                * 1.1
            )
            camera.update_matrices(800, 480)
            frames = []
            for frame in REVIEW_FRAMES:
                image = review_frame(renderer, case, frame)
                frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
            gif = a.review_out / f"{case['case']}.gif"
            frames[0].save(
                gif,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // FPS,
                loop=0,
                optimize=False,
            )
            montage = Image.new("RGB", (1200, 630), "white")
            stills = [
                frames[REVIEW_FRAMES.index(frame)].convert("RGB").resize((400, 315))
                for frame in (176, 191, 192, 208, 240, 288)
            ]
            for index, image in enumerate(stills):
                montage.paste(image, ((index % 3) * 400, (index // 3) * 315))
            png = a.review_out / f"{case['case']}.png"
            montage.save(png)
            outputs.append(
                {
                    "case": case["case"],
                    "gif": str(gif),
                    "png": str(png),
                    "gif_sha256": sha256_file(gif),
                    "png_sha256": sha256_file(png),
                }
            )
            for node in nodes:
                scene.remove(node)
            print(f"REVIEW {len(outputs)}/{len(manifest['cases'])}: {case['case']}", flush=True)
    finally:
        renderer.close()
    report = {
        "status": "complete",
        "source": str(a.review),
        "source_sha256": sha256_file(a.review),
        "readiness": ready,
        "outputs": outputs,
        "fps": FPS,
        "frames_per_case": len(REVIEW_FRAMES),
        "frame_start": REVIEW_FRAMES[0],
        "frame_stop": REVIEW_FRAMES[-1] + 1,
        "wall_seconds": time.perf_counter() - started,
        "limitations": "Aitviewer recovered-joint review, not fitted mesh or human ratings. "
        "GIF is an explicit Pillow output; no ffmpeg or dependency changes. "
        "Only static panel offsets for display; saved motion is unchanged. "
        "Fixed common orthographic camera per pair;113 frames around the switch, not a full-sequence review.",
    }
    (a.review_out / "render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run(a: argparse.Namespace) -> None:
    if a.review is not None:
        render_review(a)
        return
    app = ApplicationBootstrap.from_config(a.config)
    cfg = app.config
    steps = a.steps if a.steps is not None else cfg.data.max_motion_len // cfg.tokenizer.downsample
    request = ModelServiceRequest(
        config=Path(a.config),
        ckpt=Path(a.ckpt),
        tokenizer_ckpt=Path(a.tokenizer_ckpt),
        backbone=Backbone(a.backbone),
    )
    pipeline = load_inference_pipeline(app, request)
    chunks = []
    with torch.no_grad():
        emb = pipeline.generator.text_encoder([a.prompt]).to(pipeline.device)
        it = pipeline.generator.token_generator.stream_token_indices(
            emb,
            steps,
            temperature=a.temperature,
            top_p=a.top_p,
            cfg_scale=a.cfg_scale,
            stop_at_end=not a.fixed_length,
        )
        for ch in pipeline.decoder.stream_tokens(it):
            chunks.append(ch.squeeze(0).cpu().numpy())
    feat = np.concatenate(chunks, 0) if chunks else np.zeros((0, DIM), "float32")
    joints = recover_skeleton(feat)
    np.save(a.npy, joints)
    print(f"frames {joints.shape[0]}  -> saved {a.npy}")
    if joints.shape[0] == 0:
        print("NO FRAMES generated")
        return

    bones = kinematic_bones()
    frame_count = joints.shape[0]
    idx = np.linspace(0, frame_count - 1, min(a.frames, frame_count)).astype(int)

    lo, hi = joints.min((0, 1)), joints.max((0, 1))
    fig = plt.figure(figsize=(2.6 * len(idx), 3.2))
    for k, fi in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), k + 1, projection="3d")
        if not isinstance(ax, Axes3D):
            raise TypeError("the 3d projection did not give 3D axes")
        p = joints[fi]
        for b in bones:
            ax.plot(
                [p[b[0], 0], p[b[1], 0]],
                [p[b[0], 2], p[b[1], 2]],
                [p[b[0], 1], p[b[1], 1]],
                color="tab:blue",
                lw=2,
            )
        ax.plot(
            p[:, 0], p[:, 2], p[:, 1], linestyle="", marker="o", markersize=2.5, color="tab:red"
        )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[2], hi[2])
        ax.set_zlim(lo[1], hi[1])
        ax.set_title(f"t={fi}", fontsize=9)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-75)
    fig.suptitle(
        f'{a.backbone} pilot, cfg {a.cfg_scale}: "{a.prompt}"  ({frame_count} frames @{FPS}fps)',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a streamed generated clip to a skeleton PNG strip."
    )
    p.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--ckpt", default="artifacts/checkpoints/generator/generator_transformer.pt")
    p.add_argument("--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--review", type=Path)
    p.add_argument(
        "--review_out", type=Path, default=Path("artifacts") / "outputs" / "qa" / "review_media"
    )
    p.add_argument("--model_dir", default="data\\smplx_models")
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fixed_length", action="store_true")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--npy", default="artifacts/outputs/demo_joints.npy")
    p.add_argument("--out", default="artifacts/outputs/figures/demo_clip.png")
    run(p.parse_args())


if __name__ == "__main__":
    main()
