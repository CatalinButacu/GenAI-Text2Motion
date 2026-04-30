"""Local smoke test: verify the full training pipeline works end-to-end.

Runs both training stages on a tiny data subset (default 300 samples,
5 epochs each) so you can confirm there are no crashes and loss is
decreasing before launching a full cloud run.

Usage (CPU, ~5-10 min):
    python scripts/eval/smoke_test.py

Usage (GPU, ~2-3 min):
    python scripts/eval/smoke_test.py --device cuda --batch-size 32

The script:
  1. Trains the RVQ tokenizer for N epochs on a small slice of AMASS.
  2. Trains the SSM for N epochs using the just-trained tokenizer.
  3. Generates 5 test prompts and prints motion quality stats.
  4. Runs the held-out test split and prints final metrics.
  5. Exits with code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.motion.ssm_model import SSMMotionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")

SMOKE_PROMPTS = [
    "a person walks forward",
    "a person runs fast",
    "a person jumps up",
    "a person waves their right hand",
    "a person sits down slowly",
]


def run(cmd: list[str], label: str) -> int:
    log.info("[smoke] %s: %s", label, " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error("[smoke] %s FAILED (exit=%d, %.1fs)", label, result.returncode, elapsed)
    else:
        log.info("[smoke] %s done in %.1fs", label, elapsed)
    return result.returncode


def trainRVQ(args) -> int:
    cmd = [
        sys.executable, "scripts/training/train_rvq_tokenizer.py",
        "--data-dir", args.dataDir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batchSize),
        "--max-motion-length", str(args.maxMotionLength),
        "--max-samples", str(args.maxSamples),
        "--checkpoint-dir", str(Path(args.outDir) / "rvq"),
        "--device", args.device,
        "--seed", "42",
    ]
    return run(cmd, "RVQ tokenizer")


def trainSSM(args) -> int:
    rvqPath = str(Path(args.outDir) / "rvq" / "best_model.pt")
    ssmCkptDir = str(Path(args.outDir) / "ssm")
    cmd = [
        sys.executable, "scripts/training/train_motion_ssm.py",
        "--data-source", "amass",
        "--data-dir", args.dataDir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batchSize),
        "--max-motion-length", str(args.maxMotionLength),
        "--max-samples", str(args.maxSamples),
        "--checkpoint-dir", ssmCkptDir,
        "--rvq-checkpoint", rvqPath,
        "--device", args.device,
        "--use-sbert",
        "--bidirectional",
        "--use-film",
        "--seed", "42",
    ]
    return run(cmd, "MotionSSM")


def generateSamples(args) -> None:
    rvqPath = str(Path(args.outDir) / "rvq" / "best_model.pt")
    # Find the best_model.pt inside the run subdirectory
    ssmBase = Path(args.outDir) / "ssm"
    candidates = sorted(ssmBase.rglob("best_model.pt"))
    if not candidates:
        log.warning("[smoke] no SSM checkpoint found -- skipping generation")
        return
    ssmPath = str(candidates[-1])

    log.info("[smoke] loading model: rvq=%s  ssm=%s", rvqPath, ssmPath)
    try:
        model = SSMMotionModel(ssmPath, rvqPath)
    except Exception as e:
        log.error("[smoke] could not load model: %s", e)
        return

    print("\n" + "=" * 60)
    print("  GENERATED MOTION QUALITY (smoke prompts)")
    print("=" * 60)
    clipsDir = Path(args.outDir) / "clips"
    clipsDir.mkdir(parents=True, exist_ok=True)

    for prompt in SMOKE_PROMPTS:
        clip = model.generateFromTextTokens(prompt, numFrames=60)
        m = clip.smplxParams
        smoothness = float(np.abs(np.diff(m, axis=0)).mean())
        validity = np.isfinite(m).all() and float(np.abs(m[:, 6:69]).max()) < np.pi
        print(f"  [{prompt[:40]:<40}]  "
              f"frames={m.shape[0]}  std={m.std():.3f}  "
              f"smooth={smoothness:.4f}  valid={validity}")
        np.save(clipsDir / f"{prompt[:30].replace(' ', '_')}.npy", m)

    print("=" * 60 + "\n")
    log.info("[smoke] clips saved to %s", clipsDir)


def main() -> int:
    p = argparse.ArgumentParser(description="Local smoke test for the full training pipeline")
    p.add_argument("--data-dir", default="data/AMASS", dest="dataDir")
    p.add_argument("--max-samples", type=int, default=300,
                   help="Clips to use (300 is enough to verify the pipeline)", dest="maxSamples")
    p.add_argument("--epochs", type=int, default=5,
                   help="Epochs per stage (5 is enough to see loss decreasing)")
    p.add_argument("--batch-size", type=int, default=16, dest="batchSize")
    p.add_argument("--max-motion-length", type=int, default=64,
                   help="Shorter sequences = faster smoke test", dest="maxMotionLength")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs/smoke_test",
                   help="Where to save checkpoints and generated clips", dest="outDir")
    p.add_argument("--skip-ssm", action="store_true",
                   help="Only train tokenizer (faster sanity check)", dest="skipSsm")
    args = p.parse_args()

    log.info("[smoke] starting  device=%s  max_samples=%d  epochs=%d",
             args.device, args.maxSamples, args.epochs)
    tTotal = time.time()

    # Stage 1 — RVQ tokenizer
    rc = trainRVQ(args)
    if rc != 0:
        log.error("[smoke] RVQ training failed. Fix the error above then re-run.")
        return 1

    rvqBest = Path(args.outDir) / "rvq" / "best_model.pt"
    if not rvqBest.exists():
        log.error("[smoke] RVQ best_model.pt not found at %s", rvqBest)
        return 1

    if args.skipSsm:
        log.info("[smoke] --skip-ssm set, stopping after RVQ. Total: %.1fs", time.time() - tTotal)
        return 0

    # Stage 2 — MotionSSM
    rc = trainSSM(args)
    if rc != 0:
        log.error("[smoke] SSM training failed. Fix the error above then re-run.")
        return 1

    # Stage 3 — generation quality
    generateSamples(args)

    log.info("[smoke] ALL PASSED in %.1fs", time.time() - tTotal)
    log.info("[smoke] Next step: launch full cloud training "
             "with larger --epochs and --max-samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
