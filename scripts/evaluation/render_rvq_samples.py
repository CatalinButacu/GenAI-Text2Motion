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
import logging
import os
import sys

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.eval_rvq import (
    actionLabel,
    amassSubset,
    buildTestDataset,
    loadCheckpoint,
)
from src.data.motion_normalize import MotionStats, denormalize
from src.modules.render.smplx_render import (
    configureRenderer,
    getRenderer,
    resetScene,
    smplxParams2Sequence,
)

log = logging.getLogger(__name__)


def perClipMse(model, dataset, device, batchSize: int) -> tuple[np.ndarray, list[int]]:
    """MSE per clip (mask-aware) + frame counts. Used to rank clips."""
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)
    mses: list[float] = []
    lengths: list[int] = []

    with torch.no_grad():

        for batch in loader:
            motion = batch["motion"].to(device)
            mask = batch["motion_mask"].to(device)
            recon, _, _ = model(motion)
            diffSq = (recon - motion) ** 2
            maskExp = mask.unsqueeze(-1)
            num = (diffSq * maskExp).sum(dim=(1, 2))
            den = (maskExp.sum(dim=1).squeeze(-1) * motion.shape[2]).clamp(min=1.0)
            clipMse = (num / den).cpu().numpy().tolist()
            mses.extend(clipMse)

            for length in batch["length"]:
                lengths.append(int(length))

    return np.array(mses), lengths


def pickIndices(mses: np.ndarray, picks: list[str], nPer: int) -> dict[str, list[int]]:
    sortedIdx = np.argsort(mses)
    n = len(mses)
    out: dict[str, list[int]] = {}

    if "best" in picks:
        out["best"] = sortedIdx[:nPer].tolist()

    if "median" in picks:
        midpoint = n // 2
        half = nPer // 2
        start = max(0, midpoint - half)
        out["median"] = sortedIdx[start:start + nPer].tolist()

    if "worst" in picks:
        out["worst"] = sortedIdx[-nPer:][::-1].tolist()

    return out


def renderSmplxToGif(smplxParams: np.ndarray, outputPath: str, fps: int = 30,
                     width: int = 720, height: int = 480) -> None:
    """Render a (T, 168) SMPL-X sequence to an animated GIF.

    aitviewer's save_video with output_path=None and frame_dir=<tmp> writes
    a PNG per frame without invoking FFmpeg. We then stitch into a GIF with
    imageio (Pillow backend, no external deps).
    """
    import glob
    import shutil
    import tempfile

    seq = smplxParams2Sequence(smplxParams)
    renderer = getRenderer(width=width, height=height)
    resetScene(renderer)
    configureRenderer(renderer, seq, fps)

    frameDir = tempfile.mkdtemp(prefix="rvq_render_")

    try:
        renderer.save_video(frame_dir=frameDir, video_dir=None, output_fps=fps)
        framePaths = sorted(glob.glob(os.path.join(frameDir, "**", "frame_*.png"),
                                       recursive=True))

        if not framePaths:
            raise RuntimeError(f"no frames written to {frameDir}")
        frames = [imageio.imread(p) for p in framePaths]
        os.makedirs(os.path.dirname(outputPath) or ".", exist_ok=True)
        imageio.mimsave(outputPath, frames, duration=1.0 / fps, loop=0)
    finally:
        shutil.rmtree(frameDir, ignore_errors=True)


def renderClip(model, dataset, idx: int, stats: MotionStats, device,
               fps: int, outOrig: str, outRecon: str) -> int:
    item = dataset[idx]
    motionNorm = item["motion"].unsqueeze(0).to(device)  # (1, T_pad, 168) z-normalized
    realLen = int(item["length"])

    with torch.no_grad():
        reconNorm, _, _ = model(motionNorm)

    origNormNp = motionNorm[0, :realLen].cpu().numpy()
    reconNormNp = reconNorm[0, :realLen].cpu().numpy()
    origSmplx = denormalize(origNormNp, stats)
    reconSmplx = denormalize(reconNormNp, stats)

    log.info("[render] idx=%d  T=%d  -> %s", idx, realLen, os.path.basename(outOrig))
    renderSmplxToGif(origSmplx, outOrig, fps=fps)
    log.info("[render] idx=%d  T=%d  -> %s", idx, realLen, os.path.basename(outRecon))
    renderSmplxToGif(reconSmplx, outRecon, fps=fps)

    return realLen


def safeId(s: str) -> str:
    bad = '/\\:*?"<>|'

    for ch in bad:
        s = s.replace(ch, "_")

    return s[:60]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, dest="batchSize")
    parser.add_argument("--n-samples", type=int, default=2, dest="nSamples",
                        help="Number of clips per pick category")
    parser.add_argument("--picks", default="best,median,worst",
                        help="Comma-separated list from {best, median, worst}")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="statsPath",
                        help="Stats path (must match what the dataset was z-normalized with)")
    parser.add_argument("--output-dir", default=None, dest="outputDir")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.outputDir is None:
        args.outputDir = os.path.join(os.path.dirname(args.checkpoint), "eval", "render")
    os.makedirs(args.outputDir, exist_ok=True)
    log.info("[render] outputs -> %s", args.outputDir)

    if not os.path.exists(args.statsPath):
        log.error("[render] stats not found: %s", args.statsPath)

        return 1
    s = np.load(args.statsPath)
    stats = MotionStats(mean=s["mean"], std=s["std"])
    log.info("[render] stats loaded mean|.|=%.4f std|.|=%.4f",
             float(np.abs(stats.mean).mean()), float(np.abs(stats.std).mean()))

    device = torch.device(args.device)
    model, cfg = loadCheckpoint(args.checkpoint, device)
    dataset, src = buildTestDataset(cfg, args.statsPath)
    log.info("[render] test set: %d samples (source=%s)", len(dataset), src)

    log.info("[render] computing per-clip MSE")
    mses, lengths = perClipMse(model, dataset, device, args.batchSize)
    log.info("[render]   p10=%.4f  p50=%.4f  p90=%.4f",
             float(np.percentile(mses, 10)), float(np.percentile(mses, 50)),
             float(np.percentile(mses, 90)))

    requested = [p.strip() for p in args.picks.split(",") if p.strip()]
    picksByKind = pickIndices(mses, requested, args.nSamples)

    reportLines: list[str] = []
    reportLines.append(f"# RVQ render report — {src}")
    reportLines.append(f"\n**Checkpoint:** `{args.checkpoint}`")
    reportLines.append(f"**Stats:** `{args.statsPath}`")
    reportLines.append(f"**N per pick:** {args.nSamples}\n")
    reportLines.append(
        "| pick | clip idx | MSE | T frames | label / source | orig.mp4 | recon.mp4 |"
    )
    reportLines.append("|---|---|---|---|---|---|---|")

    for kind, indices in picksByKind.items():

        for i in indices:
            sample = dataset[i]
            sid = sample.get("texts", "") or "clip"

            if src == "amass":
                rawId = dataset.samples[i].get("sample_id", f"clip_{i}")  # type: ignore[attr-defined]
                label = amassSubset(rawId)
                fileTag = safeId(rawId)
            else:
                label = actionLabel(sid if isinstance(sid, str) else "")
                fileTag = safeId(f"{label}_{i:04d}")

            origPath = os.path.join(args.outputDir, f"{kind}_{i:04d}_{fileTag}_orig.gif")
            reconPath = os.path.join(args.outputDir, f"{kind}_{i:04d}_{fileTag}_recon.gif")

            try:
                T = renderClip(model, dataset, i, stats, device, args.fps,
                               origPath, reconPath)
            except (RuntimeError, OSError) as exc:
                log.error("[render] FAILED idx=%d kind=%s: %s", i, kind, exc)
                continue
            reportLines.append(
                f"| {kind} | {i} | {float(mses[i]):.4f} | {T} | {label} | "
                f"{os.path.basename(origPath)} | {os.path.basename(reconPath)} |"
            )

    with open(os.path.join(args.outputDir, "report_render.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(reportLines))
    log.info("[render] DONE — see %s/report_render.md", args.outputDir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
