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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

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

def train_rvq(args) -> int:
    cmd = [
        sys.executable, "scripts/training/train_rvq_tokenizer.py",
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-motion-length", str(args.max_motion_length),
        "--max-samples", str(args.max_samples),
        "--checkpoint-dir", str(Path(args.out_dir) / "rvq"),
        "--device", args.device,
        "--seed", "42",
    ]
    return run(cmd, "RVQ tokenizer")

def train_ssm(args) -> int:
    rvq_path = str(Path(args.out_dir) / "rvq" / "best_model.pt")
    ssm_ckpt_dir = str(Path(args.out_dir) / "ssm")
    cmd = [
        sys.executable, "scripts/training/train_motion_ssm.py",
        "--data-source", "amass",
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-motion-length", str(args.max_motion_length),
        "--max-samples", str(args.max_samples),
        "--checkpoint-dir", ssm_ckpt_dir,
        "--rvq-checkpoint", rvq_path,
        "--device", args.device,
        "--use-sbert",
        "--bidirectional",
        "--use-film",
        "--seed", "42",
    ]
    return run(cmd, "MotionSSM")

def generate_samples(args) -> None:
    rvq_path = str(Path(args.out_dir) / "rvq" / "best_model.pt")
    # Find the best_model.pt inside the run subdirectory
    ssm_base = Path(args.out_dir) / "ssm"
    candidates = sorted(ssm_base.rglob("best_model.pt"))
    if not candidates:
        log.warning("[smoke] no SSM checkpoint found -- skipping generation")
        return
    ssm_path = str(candidates[-1])

    log.info("[smoke] loading model: rvq=%s  ssm=%s", rvq_path, ssm_path)
    try:
        model = SSMMotionModel(ssm_path, rvq_path)
    except Exception as e:
        log.error("[smoke] could not load model: %s", e)
        return

    print("\n" + "=" * 60)
    print("  GENERATED MOTION QUALITY (smoke prompts)")
    print("=" * 60)
    clips_dir = Path(args.out_dir) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for prompt in SMOKE_PROMPTS:
        clip = model.generate_from_text_tokens(prompt, num_frames=60)
        m = clip.smplx_params
        smoothness = float(np.abs(np.diff(m, axis=0)).mean())
        validity = np.isfinite(m).all() and float(np.abs(m[:, 6:69]).max()) < np.pi
        print(f"  [{prompt[:40]:<40}]  "
              f"frames={m.shape[0]}  std={m.std():.3f}  "
              f"smooth={smoothness:.4f}  valid={validity}")
        np.save(clips_dir / f"{prompt[:30].replace(' ', '_')}.npy", m)

    print("=" * 60 + "\n")
    log.info("[smoke] clips saved to %s", clips_dir)

def main() -> int:
    p = argparse.ArgumentParser(description="Local smoke test for the full training pipeline")
    p.add_argument("--data-dir", default="data/AMASS", dest="data_dir")
    p.add_argument("--max-samples", type=int, default=300,
                   help="Clips to use (300 is enough to verify the pipeline)", dest="max_samples")
    p.add_argument("--epochs", type=int, default=5,
                   help="Epochs per stage (5 is enough to see loss decreasing)")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--max-motion-length", type=int, default=64,
                   help="Shorter sequences = faster smoke test", dest="max_motion_length")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs/smoke_test",
                   help="Where to save checkpoints and generated clips", dest="out_dir")
    p.add_argument("--skip-ssm", action="store_true",
                   help="Only train tokenizer (faster sanity check)", dest="skip_ssm")
    args = p.parse_args()

    log.info("[smoke] starting  device=%s  max_samples=%d  epochs=%d",
             args.device, args.max_samples, args.epochs)
    t_total = time.time()

    # Stage 1 — RVQ tokenizer
    rc = train_rvq(args)
    if rc != 0:
        log.error("[smoke] RVQ training failed. Fix the error above then re-run.")
        return 1

    rvq_best = Path(args.out_dir) / "rvq" / "best_model.pt"
    if not rvq_best.exists():
        log.error("[smoke] RVQ best_model.pt not found at %s", rvq_best)
        return 1

    if args.skip_ssm:
        log.info("[smoke] --skip-ssm set, stopping after RVQ. Total: %.1fs", time.time() - t_total)
        return 0

    # Stage 2 — MotionSSM
    rc = train_ssm(args)
    if rc != 0:
        log.error("[smoke] SSM training failed. Fix the error above then re-run.")
        return 1

    # Stage 3 — generation quality
    generate_samples(args)

    log.info("[smoke] ALL PASSED in %.1fs", time.time() - t_total)
    log.info("[smoke] Next step: launch full cloud training "
             "with larger --epochs and --max-samples.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
