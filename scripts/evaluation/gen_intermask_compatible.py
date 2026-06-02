"""Generate Inter-X test-split motions in InterMask-compatible (T, 56, 6) format.

Loads our SSM checkpoint, iterates the official Inter-X test split (1708 clips),
generates a two-actor motion per text prompt, converts to Inter-X canonical
joint layout, and saves one .npy file per clip ID.

Output layout: <output_dir>/<clip_id>.npy
Each file: float32 array of shape (T, 56, 6).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.intermask_compat import pairToInterxFormat
from src.modules.motion.ssm_model import SSMMotionModel

log = logging.getLogger(__name__)

MAX_MOTION_LENGTH = 200  # SSM training cap


def readFirstCaption(textPath: str) -> str:
    with open(textPath, encoding="utf-8") as f:
        first = f.readline().strip()
    # format: caption#pos_tagged#start#end
    return first.split("#", 1)[0]


def readTestSplit(splitPath: str) -> list[str]:
    with open(splitPath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def loadGtLengths(h5Path: str, clipIds: list[str]) -> dict[str, int]:
    """Get per-clip GT frame counts so we match Inter-X eval's length protocol."""
    out: dict[str, int] = {}
    with h5py.File(h5Path, "r") as h:

        for cid in clipIds:
            if cid in h:
                out[cid] = int(h[cid].shape[0])

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssm-checkpoint", type=str,
                        default="checkpoints/motion_ssm/best_model.pt", dest="ssmCheckpoint")
    parser.add_argument("--rvq-checkpoint", type=str,
                        default="checkpoints/rvq_tokenizer/best_model.pt", dest="rvqCheckpoint")
    parser.add_argument("--test-split", type=str,
                        default="data/inter-x/splits/test.txt", dest="testSplit")
    parser.add_argument("--text-dir", type=str,
                        default="data/inter-x/processed/texts_processed", dest="textDir")
    parser.add_argument("--gt-h5", type=str,
                        default="data/inter-x/inter-x-001.h5", dest="gtH5",
                        help="GT motion h5 to read per-clip frame counts.")
    parser.add_argument("--output-dir", type=str, required=True, dest="outputDir")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0, dest="topP")
    parser.add_argument("--cfg-scale", type=float, default=1.0, dest="cfgScale")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N clips (smoke testing).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    os.makedirs(args.outputDir, exist_ok=True)
    log.info("loading SSM + RVQ...")
    model = SSMMotionModel(checkpointPath=args.ssmCheckpoint,
                           rvqCheckpointPath=args.rvqCheckpoint)

    clipIds = readTestSplit(args.testSplit)
    log.info("test split: %d clips", len(clipIds))

    if args.limit is not None:
        clipIds = clipIds[:args.limit]
        log.info("limited to first %d clips", len(clipIds))
    gtLens = loadGtLengths(args.gtH5, clipIds)
    log.info("loaded GT lengths for %d/%d clips", len(gtLens), len(clipIds))

    nSaved = 0
    nSkipped = 0
    t0 = time.time()

    for i, cid in enumerate(clipIds):
        textPath = os.path.join(args.textDir, cid + ".txt")

        if not os.path.exists(textPath):
            nSkipped += 1
            continue
        caption = readFirstCaption(textPath)
        # Match Inter-X eval protocol: generate at GT length, capped at 200.
        gtT = gtLens.get(cid, 100)
        numFrames = min(gtT, MAX_MOTION_LENGTH)

        try:
            p1, p2 = model.generateBothActors(
                caption, numFrames=numFrames,
                temperature=args.temperature, topP=args.topP, cfgScale=args.cfgScale,
            )
        except Exception as exc:
            log.warning("generation failed for %s: %s", cid, exc)
            nSkipped += 1
            continue

        if p2 is None:
            # Single-actor model: replicate P1 as P2 for format consistency.
            p2 = p1.copy()
        motion263 = pairToInterxFormat(p1, p2)
        np.save(os.path.join(args.outputDir, cid + ".npy"), motion263)
        nSaved += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(clipIds) - i - 1) / max(rate, 1e-6)
            log.info("[%4d/%d] saved=%d skipped=%d  rate=%.2f/s  eta=%.1f min",
                     i + 1, len(clipIds), nSaved, nSkipped, rate, eta / 60)
    log.info("DONE: saved %d, skipped %d, total wall %.1f min",
             nSaved, nSkipped, (time.time() - t0) / 60)
    log.info("output dir: %s", args.outputDir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
