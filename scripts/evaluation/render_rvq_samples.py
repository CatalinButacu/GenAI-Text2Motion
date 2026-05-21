"""Render visual reconstructions for a few sample clips per RVQ checkpoint.

Picks best / median / worst clips by per-clip reconstruction MSE, decodes them
through the codec, denormalizes both original and reconstruction, and writes
two MP4s per pick (original + reconstructed) using the existing aitviewer
SMPL-X headless renderer.

Note: the renderer uses OpenGL on the GPU. While another training job is using
the GPU, this may compete for VRAM. Pass --force-now if you accept the risk;
otherwise wait until training completes.

Usage:
    python scripts/evaluation/render_rvq_samples.py \
        --checkpoint checkpoints/rvq_tokenizer/<run_id>/best_model.pt \
        --n-samples 2 --picks best,median,worst
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys
import tempfile

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluation.eval_rvq import (
    action_label,
    amass_subset,
    build_test_dataset,
    load_checkpoint,
)
from src.data.motion_normalize import MotionStats, denormalize
from src.modules.render.smplx_render import (
    configure_renderer,
    get_renderer,
    reset_scene,
    smplx_params2_sequence,
)

log = logging.getLogger(__name__)

def per_clip_mse(model, dataset, device, batch_size: int) -> tuple[np.ndarray, list[int]]:
    """MSE per clip (mask-aware) + frame counts. Used to rank clips."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mses: list[float] = []
    lengths: list[int] = []

    with torch.no_grad():

        for batch in loader:
            motion = batch["motion"].to(device)
            mask = batch["motion_mask"].to(device)
            recon, _, _ = model(motion)
            diff_sq = (recon - motion) ** 2
            mask_exp = mask.unsqueeze(-1)
            num = (diff_sq * mask_exp).sum(dim=(1, 2))
            den = (mask_exp.sum(dim=1).squeeze(-1) * motion.shape[2]).clamp(min=1.0)
            clip_mse = (num / den).cpu().numpy().tolist()
            mses.extend(clip_mse)

            for length in batch["length"]:
                lengths.append(int(length))

    return np.array(mses), lengths

def pick_indices(mses: np.ndarray, picks: list[str], n_per: int) -> dict[str, list[int]]:
    sorted_idx = np.argsort(mses)
    n = len(mses)
    out: dict[str, list[int]] = {}

    if "best" in picks:
        out["best"] = sorted_idx[:n_per].tolist()

    if "median" in picks:
        midpoint = n // 2
        half = n_per // 2
        start = max(0, midpoint - half)
        out["median"] = sorted_idx[start:start + n_per].tolist()

    if "worst" in picks:
        out["worst"] = sorted_idx[-n_per:][::-1].tolist()

    return out

def render_smplx_to_gif(smplx_params: np.ndarray, output_path: str, fps: int = 30,
                     width: int = 720, height: int = 480) -> None:
    """Render a (T, 168) SMPL-X sequence to an animated GIF.

    aitviewer's save_video with output_path=None and frame_dir=<tmp> writes
    a PNG per frame without invoking FFmpeg. We then stitch into a GIF with
    imageio (Pillow backend, no external deps).
    """
    seq = smplx_params2_sequence(smplx_params)
    renderer = get_renderer(width=width, height=height)
    reset_scene(renderer)
    configure_renderer(renderer, seq, fps)

    frame_dir = tempfile.mkdtemp(prefix="rvq_render_")

    try:
        renderer.save_video(frame_dir=frame_dir, video_dir=None, output_fps=fps)
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "**", "frame_*.png"),
                                       recursive=True))

        if not frame_paths:
            raise RuntimeError(f"no frames written to {frame_dir}")
        frames = [imageio.imread(p) for p in frame_paths]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        imageio.mimsave(output_path, frames, duration=1.0 / fps, loop=0)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

def render_clip(model, dataset, idx: int, stats: MotionStats, device,
               fps: int, out_orig: str, out_recon: str) -> int:
    item = dataset[idx]
    motion_norm = item["motion"].unsqueeze(0).to(device)  # (1, T_pad, 168) z-normalized
    real_len = int(item["length"])

    with torch.no_grad():
        recon_norm, _, _ = model(motion_norm)

    orig_norm_np = motion_norm[0, :real_len].cpu().numpy()
    recon_norm_np = recon_norm[0, :real_len].cpu().numpy()
    orig_smplx = denormalize(orig_norm_np, stats)
    recon_smplx = denormalize(recon_norm_np, stats)

    log.info("[render] idx=%d  T=%d  -> %s", idx, real_len, os.path.basename(out_orig))
    render_smplx_to_gif(orig_smplx, out_orig, fps=fps)
    log.info("[render] idx=%d  T=%d  -> %s", idx, real_len, os.path.basename(out_recon))
    render_smplx_to_gif(recon_smplx, out_recon, fps=fps)

    return real_len

def safe_id(s: str) -> str:
    bad = '/\\:*?"<>|'

    for ch in bad:
        s = s.replace(ch, "_")

    return s[:60]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--n-samples", type=int, default=2, dest="n_samples",
                        help="Number of clips per pick category")
    parser.add_argument("--picks", default="best,median,worst",
                        help="Comma-separated list from {best, median, worst}")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="stats_path",
                        help="Stats path (must match what the dataset was z-normalized with)")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), "eval", "render")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("[render] outputs -> %s", args.output_dir)

    if not os.path.exists(args.stats_path):
        log.error("[render] stats not found: %s", args.stats_path)

        return 1
    s = np.load(args.stats_path)
    stats = MotionStats(mean=s["mean"], std=s["std"])
    log.info("[render] stats loaded mean|.|=%.4f std|.|=%.4f",
             float(np.abs(stats.mean).mean()), float(np.abs(stats.std).mean()))

    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    dataset, src = build_test_dataset(cfg, args.stats_path)
    log.info("[render] test set: %d samples (source=%s)", len(dataset), src)

    log.info("[render] computing per-clip MSE")
    mses, lengths = per_clip_mse(model, dataset, device, args.batch_size)
    log.info("[render]   p10=%.4f  p50=%.4f  p90=%.4f",
             float(np.percentile(mses, 10)), float(np.percentile(mses, 50)),
             float(np.percentile(mses, 90)))

    requested = [p.strip() for p in args.picks.split(",") if p.strip()]
    picks_by_kind = pick_indices(mses, requested, args.n_samples)

    report_lines: list[str] = []
    report_lines.append(f"# RVQ render report — {src}")
    report_lines.append(f"\n**Checkpoint:** `{args.checkpoint}`")
    report_lines.append(f"**Stats:** `{args.stats_path}`")
    report_lines.append(f"**N per pick:** {args.n_samples}\n")
    report_lines.append(
        "| pick | clip idx | MSE | T frames | label / source | orig.mp4 | recon.mp4 |"
    )
    report_lines.append("|---|---|---|---|---|---|---|")

    for kind, indices in picks_by_kind.items():

        for i in indices:
            sample = dataset[i]
            sid = sample.get("texts", "") or "clip"

            if src == "amass":
                raw_id = dataset.samples[i].get("sample_id", f"clip_{i}")  # type: ignore[attr-defined]
                label = amass_subset(raw_id)
                file_tag = safe_id(raw_id)
            else:
                label = action_label(sid if isinstance(sid, str) else "")
                file_tag = safe_id(f"{label}_{i:04d}")

            orig_path = os.path.join(args.output_dir, f"{kind}_{i:04d}_{file_tag}_orig.gif")
            recon_path = os.path.join(args.output_dir, f"{kind}_{i:04d}_{file_tag}_recon.gif")

            try:
                T = render_clip(model, dataset, i, stats, device, args.fps,
                               orig_path, recon_path)
            except (RuntimeError, OSError) as exc:
                log.error("[render] FAILED idx=%d kind=%s: %s", i, kind, exc)
                continue
            report_lines.append(
                f"| {kind} | {i} | {float(mses[i]):.4f} | {T} | {label} | "
                f"{os.path.basename(orig_path)} | {os.path.basename(recon_path)} |"
            )

    with open(os.path.join(args.output_dir, "report_render.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log.info("[render] DONE — see %s/report_render.md", args.output_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
