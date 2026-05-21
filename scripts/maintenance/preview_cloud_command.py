"""Render the exact train_motion_ssm.py command that startup.sh would launch
on EC2 for a given set of terraform vars.

Useful as a "dry run" before terraform apply: confirms every flag interpolates
correctly and the EXTRA_FLAGS string assembly is sane.
"""

from __future__ import annotations

import argparse
import shlex


def render(tier: str) -> str:
    presets = {
        "tier-A-baseline": {  # current default
            "data_source": "unified", "epochs_ssm": 80, "batch_size": 64,
            "text_encoder": "", "ar_k_head": False, "compile_model": False,
        },
        "tier-B-clip-b": {
            "data_source": "unified", "epochs_ssm": 20, "batch_size": 64,
            "text_encoder": "clip-b", "ar_k_head": False, "compile_model": False,
        },
        "tier-C-clip-ar-compile": {
            "data_source": "unified", "epochs_ssm": 30, "batch_size": 64,
            "text_encoder": "clip-b", "ar_k_head": True, "compile_model": True,
        },
    }
    p = presets[tier]

    # Mirror EXTRA_FLAGS construction from startup.sh lines 268-271
    extra: list[str] = []
    if p["text_encoder"]:
        extra += ["--text-encoder", p["text_encoder"]]
    if p["ar_k_head"]:
        extra += ["--ar-k-head"]
    if p["compile_model"]:
        extra += ["--compile"]

    cmd = [
        "python", "scripts/training/train_motion_ssm.py",
        "--data-source", p["data_source"],
        "--sources", "humanml3d",
        "--use-sbert",
        "--bidirectional",
        "--use-film",
        "--gradient-checkpointing",
        "--d-model", "384",
        "--d-state", "64",
        "--n-layers", "6",
        "--max-motion-length", "200",
        "--batch-size", str(p["batch_size"]),
        "--lr", "1e-4",
        "--epochs", str(p["epochs_ssm"]),
        "--num-workers", "4",
        "--checkpoint-dir", "checkpoints/motion_ssm",
        "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
        "--device", "cuda",
        *extra,
    ]
    return " \\\n    ".join(shlex.quote(t) for t in cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="tier-B-clip-b",
                        choices=["tier-A-baseline", "tier-B-clip-b", "tier-C-clip-ar-compile"])
    args = parser.parse_args()
    print(f"# Cloud will run (tier: {args.tier}):\n")
    print(render(args.tier))


if __name__ == "__main__":
    main()
