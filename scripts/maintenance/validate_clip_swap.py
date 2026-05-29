"""Proof-of-concept comparison: SBERT vs CLIP text encoder on a tiny training run.

This is NOT a paper-ready ablation. It runs 5 epochs each on a small data
slice with otherwise-identical hyperparameters, and reports the val_ce curve
to verify the wiring is correct and that CLIP at minimum doesn't hurt.

A real ablation needs ~50-100 epochs on full HumanML3D and 3+ seeds.

Run from repo root (point at a precomputed RVQ checkpoint and AMASS root):
    python scripts/maintenance/validate_clip_swap.py \\
        --data-dir D:/Facultate/dissertation/data/AMASS \\
        --rvq-checkpoint <RVQ_CKPT_ROOT>/best_model.pt \\
        --epochs 5 --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_training(label: str, text_encoder: str, args, out_dir: Path) -> dict:
    """Invoke train_motion_ssm.py once. Returns {label, val_ce_curve, wall_time}."""
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/training/train_motion_ssm.py",
        "--data-source", "amass",
        "--data-dir", args.data_dir,
        "--rvq-checkpoint", args.rvq_checkpoint,
        "--checkpoint-dir", str(run_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-samples", str(args.max_samples),
        "--max-motion-length", "120",  # smaller for speed
        "--d-model", "256",            # smaller for speed
        "--n-layers", "2",             # smaller for speed
        "--lr", "5e-4",
        "--seed", "42",
        "--device", args.device,
        "--use-sbert",
        "--text-encoder", text_encoder,
        "--bidirectional",
        "--use-film",
    ]
    print(f"\n{'=' * 70}")
    print(f"  RUN: {label}  (text encoder: {text_encoder})")
    print(f"{'=' * 70}")
    print("  cmd:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    elapsed = time.perf_counter() - t0
    print(f"  exit_code={proc.returncode}  wall={elapsed:.1f}s")

    # Pull the val_ce curve out of training.log if present.
    val_curve = parse_val_curve(run_dir)
    return {
        "label": label,
        "text_encoder": text_encoder,
        "exit_code": proc.returncode,
        "wall_time_sec": elapsed,
        "val_ce_curve": val_curve,
    }


def parse_val_curve(run_dir: Path) -> list[float]:
    """Read run_dir/.../training.log and pull out 'val_loss=' values per epoch."""
    log_paths = list(run_dir.rglob("training.log"))
    if not log_paths:
        return []
    text = log_paths[0].read_text(encoding="utf-8", errors="ignore")
    curve: list[float] = []
    for m in re.finditer(r"val_loss=([0-9]+\.[0-9]+)", text):
        curve.append(float(m.group(1)))
    return curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="AMASS root (e.g. D:/Facultate/dissertation/data/AMASS)")
    parser.add_argument("--rvq-checkpoint", required=True,
                        help="Path to a trained RVQ tokenizer best_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/clip_validation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, encoder in [
        ("baseline_sbert", "sbert-small"),
        ("candidate_clip_b", "clip-b"),
    ]:
        results.append(run_training(label, encoder, args, out_dir))

    # Verdict
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"{'label':<22}  {'wall (s)':<10}  {'final val_ce':<14}  {'curve':<10}")
    for r in results:
        last = r["val_ce_curve"][-1] if r["val_ce_curve"] else float("nan")
        print(
            f"{r['label']:<22}  {r['wall_time_sec']:<10.1f}  "
            f"{last:<14.4f}  {r['val_ce_curve']}"
        )

    if all(r["val_ce_curve"] for r in results):
        sbert_final = results[0]["val_ce_curve"][-1]
        clip_final = results[1]["val_ce_curve"][-1]
        delta = sbert_final - clip_final
        verdict = "CLIP wins" if delta > 0.01 else (
            "SBERT wins" if delta < -0.01 else "tied"
        )
        print()
        print(f"  delta (sbert - clip) = {delta:+.4f}   -> {verdict}")
        print("  (5-epoch run, not a paper-ready ablation -- just a sanity check)")

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
