"""Surface RVQ codebook prototypes for human labeling (option A from methodology).

The RVQ tokenizer is unsupervised — it learns motion atoms with no text labels.
This script renders short SMPL-X GIFs for the most-used entries of a codebook,
so a human can watch each prototype and assign a verb-phrase label. Together,
the labels form the atomic motion vocabulary discovered by the tokenizer.

Compound actions ("walk and then sit") emerge naturally as sequences of these
atomic tokens — see scripts/evaluation/vocab_compose.py + CodebookVocab for
inference-time decomposition.

Usage:
    python scripts/evaluation/codebook_label.py \\
        --checkpoint checkpoints/rvq_tokenizer/<run_id>/best_model.pt \\
        --codebook 0 --top-k 64 --n-prototypes 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluation.eval_rvq import buildTestDataset, loadCheckpoint
from scripts.evaluation.render_rvq_samples import renderSmplxToGif
from src.data.motion_normalize import MotionStats, denormalize

log = logging.getLogger(__name__)

@torch.no_grad()
def encodeAll(model, dataset, device, batchSize: int):
    """Encode the whole dataset once. Cache indices, motion, latent mask, texts."""
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)
    indicesList: list[torch.Tensor] = []
    motionList: list[torch.Tensor] = []
    maskLatList: list[torch.Tensor] = []
    texts: list[str] = []
    downT = model.downT

    for batch in loader:
        motion = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)
        x = motion.transpose(1, 2)
        z = model.encoder(x).transpose(1, 2)
        _, indices, _ = model.rvq(z)
        B, Tpad = mask.shape
        Tlat = indices.shape[1]
        latMask = torch.zeros(B, Tlat, dtype=torch.bool, device=mask.device)

        for tl in range(Tlat):
            start = tl * downT
            end = min(start + downT, Tpad)
            latMask[:, tl] = mask[:, start:end].any(dim=1)
        indicesList.append(indices.cpu())
        motionList.append(motion.cpu())
        maskLatList.append(latMask.cpu())
        rawTexts = batch.get("texts", [""] * B)

        for t in rawTexts:
            texts.append(t if isinstance(t, str) else "")

    return (
        torch.cat(indicesList, dim=0),
        torch.cat(motionList, dim=0),
        torch.cat(maskLatList, dim=0),
        texts,
    )

def topEntriesByFrequency(indices: torch.Tensor, mask: torch.Tensor,
                          codebook: int, topK: int) -> list[tuple[int, int]]:
    flatIdx = indices[..., codebook]
    flatVals = flatIdx[mask].numpy()
    bins = np.bincount(flatVals)
    order = np.argsort(bins)[::-1]

    return [(int(e), int(bins[e])) for e in order[:topK] if bins[e] > 0]

def pickPrototypes(indices: torch.Tensor, mask: torch.Tensor, codebook: int,
                   entry: int, nProto: int) -> list[tuple[int, int]]:
    # One exemplar per distinct clip — gives diversity in body shape/context
    # without needing a second forward pass to compute exact distances.
    flatIdx = indices[..., codebook]
    matches = ((flatIdx == entry) & mask).nonzero(as_tuple=False)

    if matches.numel() == 0:
        return []
    seen: set[int] = set()
    out: list[tuple[int, int]] = []

    for nt in matches:
        clipIdx = int(nt[0])
        frameLat = int(nt[1])

        if clipIdx in seen:
            continue
        seen.add(clipIdx)
        out.append((clipIdx, frameLat))

        if len(out) >= nProto:
            break

    return out

def renderPrototype(motion: torch.Tensor, frameLat: int, downT: int,
                    windowFrames: int, stats: MotionStats, outPath: str,
                    fps: int) -> int:
    center = frameLat * downT + downT // 2
    half = windowFrames // 2
    Tpad = motion.shape[0]
    start = max(0, center - half)
    end = min(Tpad, center + half)
    clip = motion[start:end].cpu().numpy()
    smplx = denormalize(clip, stats)
    renderSmplxToGif(smplx, outPath, fps=fps)

    return end - start

def topTextSnippets(indices: torch.Tensor, mask: torch.Tensor, codebook: int,
                    entry: int, texts: list[str], maxSnippets: int = 5) -> list[str]:
    flatIdx = indices[..., codebook]
    matches = ((flatIdx == entry) & mask).nonzero(as_tuple=False)
    seen: set[int] = set()
    out: list[str] = []

    for nt in matches:
        clipIdx = int(nt[0])

        if clipIdx in seen or clipIdx >= len(texts):
            continue
        seen.add(clipIdx)
        snippet = texts[clipIdx].strip().replace("\n", " ")[:80]

        if snippet:
            out.append(snippet)

        if len(out) >= maxSnippets:
            break

    return out

def writeReadme(outDir: str, runId: str, codebook: int, nEntries: int) -> None:
    text = f"""# Codebook labeling — `{runId}` codebook {codebook}

This directory holds the prototypes for the {nEntries} most-frequent entries of
RVQ codebook {codebook}.

## How to label

Open `labels.csv` in a spreadsheet. For each row, watch the prototype GIFs
listed in `prototype_paths` and write a short verb-phrase in the `label`
column. Examples: `walk_forward`, `arms_above_head`, `sit_down`, `idle`.

## Conventions

- One verb per label (no compounds — those emerge as token sequences).
- Lowercase, snake_case.
- Use `idle` for ambiguous static poses.
- Use `transition` if it's clearly a between-state movement.
- Use `noise` if the prototype looks broken/garbled (these entries get
  dropped from the vocab).

## What happens after

Run `python scripts/evaluation/vocab_compose.py --labels labels.csv ...`
to test compound-text decomposition (e.g. "walk then sit" -> token sequence).
"""
    with open(os.path.join(outDir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, dest="batchSize")
    parser.add_argument("--codebook", type=int, default=0,
                        help="Which RVQ layer to label (0 = coarsest, recommended)")
    parser.add_argument("--top-k", type=int, default=64, dest="topK",
                        help="Label only the K most-used entries (rest are rare/dead)")
    parser.add_argument("--n-prototypes", type=int, default=3, dest="nPrototypes")
    parser.add_argument("--window-frames", type=int, default=32, dest="windowFrames",
                        help="Length of each prototype clip in motion frames (~1s @30fps)")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="statsPath")
    parser.add_argument("--output-dir", default=None, dest="outputDir")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    runId = os.path.basename(os.path.dirname(args.checkpoint))

    if args.outputDir is None:
        args.outputDir = os.path.join(os.path.dirname(args.checkpoint),
                                       "eval", "codebook_labels")
    os.makedirs(args.outputDir, exist_ok=True)
    log.info("[label] outputs -> %s", args.outputDir)

    if not os.path.exists(args.statsPath):
        log.error("[label] stats not found: %s", args.statsPath)

        return 1
    s = np.load(args.statsPath)
    stats = MotionStats(mean=s["mean"], std=s["std"])

    device = torch.device(args.device)
    model, cfg = loadCheckpoint(args.checkpoint, device)
    dataset, src = buildTestDataset(cfg, args.statsPath)
    log.info("[label] test set: %d samples (source=%s)", len(dataset), src)

    log.info("[label] encoding test set ...")
    indices, motion, latMask, texts = encodeAll(model, dataset, device, args.batchSize)
    log.info("[label] encoded N=%d  Tlat=%d  K=%d",
             indices.shape[0], indices.shape[1], indices.shape[2])

    topEntries = topEntriesByFrequency(indices, latMask, args.codebook, args.topK)
    log.info("[label] codebook %d: %d entries to label (covers %d/%d steps)",
             args.codebook, len(topEntries),
             sum(c for _, c in topEntries),
             int(latMask.sum().item()))

    csvPath = os.path.join(args.outputDir, "labels.csv")
    rows: list[dict] = []

    with open(csvPath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "codebook_idx", "entry_idx", "n_assigned", "prototype_paths",
            "top_texts", "label", "notes",
        ])
        writer.writeheader()

        for rank, (entry, count) in enumerate(topEntries):
            entryDir = os.path.join(args.outputDir, f"cb{args.codebook}_e{entry:04d}")
            os.makedirs(entryDir, exist_ok=True)
            picks = pickPrototypes(indices, latMask, args.codebook,
                                    entry, args.nPrototypes)
            paths: list[str] = []

            for k, (clipIdx, frameLat) in enumerate(picks):
                outPath = os.path.join(entryDir, f"proto_{k}.gif")

                try:
                    Tactual = renderPrototype(
                        motion[clipIdx], frameLat, model.downT, args.windowFrames,
                        stats, outPath, args.fps,
                    )
                except (RuntimeError, OSError) as exc:
                    log.warning("[label] cb%d e%d proto%d: render failed: %s",
                                args.codebook, entry, k, exc)
                    continue
                rel = os.path.relpath(outPath, args.outputDir)
                paths.append(rel)
                log.info("[label] %3d/%d cb%d e%d  proto%d  T=%d  -> %s",
                         rank + 1, len(topEntries), args.codebook, entry, k, Tactual, rel)

            snippets = topTextSnippets(indices, latMask, args.codebook, entry, texts)
            row = {
                "codebook_idx": args.codebook,
                "entry_idx": entry,
                "n_assigned": count,
                "prototype_paths": ";".join(paths),
                "top_texts": " | ".join(snippets),
                "label": "",
                "notes": "",
            }
            writer.writerow(row)
            rows.append(row)

    writeReadme(args.outputDir, runId, args.codebook, len(topEntries))
    log.info("[label] DONE  csv=%s  prototypes=%d  see README.md for instructions",
             csvPath, len(rows))

    return 0

if __name__ == "__main__":
    sys.exit(main())
