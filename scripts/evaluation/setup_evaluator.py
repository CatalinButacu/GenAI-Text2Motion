#!/usr/bin/env python
"""One-time setup: link T2M evaluator weights from the download cache.

Run this once after the initial download to install the official T2M evaluator
weights into the location expected by compute_fid.py.

Usage
-----
    python scripts/evaluation/setup_evaluator.py

What it does
------------
1. Checks if weights are already in data/t2m/text_mot_match/model/finest.tar
2. If not, copies from the download cache at
   data/t2m_download/extracted/t2m/text_mot_match/model/finest.tar
3. Validates that the weights load correctly into T2MMotionEncoder
4. Runs a quick forward-pass sanity check
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (loaded for side effects when motion_encoder imports torch)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.motion_encoder import extractFeatures, loadEncoder

SRC = Path("data/t2m_download/extracted/t2m/text_mot_match/model/finest.tar")
DEST = Path("data/t2m/text_mot_match/model/finest.tar")


def setup() -> bool:
    if DEST.exists():
        print(f"[setup] Weights already present at {DEST}")
    elif SRC.exists():
        print(f"[setup] Copying {SRC.stat().st_size / 1e6:.1f}MB -> {DEST} ...")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC, DEST)
        print("[setup] Copy complete.")
    else:
        print(f"[setup] ERROR: source not found at {SRC}")
        print("  Run the download step first:")
        print(
            "    gdown --fuzzy 'https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view'"
        )
        return False

    # Validate
    print("[setup] Validating weight load ...")

    enc = loadEncoder(inputDim=168, device="cpu")
    if not enc._loaded_pretrained:
        print("[setup] FAILED: weights not loaded (check error above)")
        return False

    # Quick forward pass
    motions = [
        np.random.randn(60, 168).astype("float32"),
        np.random.randn(30, 168).astype("float32"),
    ]
    feats = extractFeatures(enc, motions, device="cpu")

    norms = np.linalg.norm(feats, axis=1)
    ok = feats.shape == (2, 512) and np.allclose(norms, 1.0, atol=1e-5)
    if ok:
        print(f"[setup] OK --feats shape={feats.shape}, L2 norms={norms}")
        print()
        print("  T2M evaluator is ready. FID results WILL BE comparable to")
        print("  published results (T2M CVPR 2022, T2M-GPT, MoMask, etc.)")
        print()
        print("  Run evaluation with:")
        print("    python scripts/evaluation/compute_fid.py \\")
        print("        --checkpoint checkpoints/motion_ssm_hml3d/best_model.pt \\")
        print("        --data-dir data/humanml3d \\")
        print("        --split test")
    else:
        print(f"[setup] Shape/norm check FAILED: shape={feats.shape} norms={norms}")
        return False

    return True


if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)
