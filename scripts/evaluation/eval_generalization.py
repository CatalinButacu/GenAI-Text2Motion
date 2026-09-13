from __future__ import annotations

import argparse
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.motion.contracts import Split
from text2motion.tokenization.metrics import evaluate_tokenizer_reconstruction

_CHECKPOINT_CONFIGS = {
    "rvq_l4_512": "configs/tokenizer/rvq_l4_512.yaml",
    "rvq_l6_512": "configs/tokenizer/rvq_l6_512.yaml",
    "rvq_l8_512": "configs/tokenizer/rvq_l8_512.yaml",
    "rvq_l4_1024": "configs/tokenizer/rvq_l4_1024.yaml",
    "rvq_l6_1024": "configs/tokenizer/rvq_l6_1024.yaml",
    "rvq_l8_1024": "configs/tokenizer/rvq_l8_1024.yaml",
    "fsq_g4_v512": "configs/tokenizer/fsq_g4_v512.yaml",
    "fsq_g6_v512": "configs/tokenizer/tokenizer_isovocab.yaml",
    "fsq_g8_v512": "configs/tokenizer/fsq_g8_v512.yaml",
    "fsq_g4_v1024": "configs/tokenizer/fsq_g4_v1024.yaml",
    "fsq_g6_v1024": "configs/tokenizer/fsq_g6_v1024.yaml",
    "fsq_g8_v1024": "configs/tokenizer/fsq_g8_v1024.yaml",
    "fsq_g4_v1000": "configs/tokenizer/tok_g4_v1000.yaml",
    "fsq_g6_v1000": "configs/tokenizer/tok_g6_v1000.yaml",
    "fsq_g8_v1000": "configs/tokenizer/tok_g8_v1000.yaml",
}
_SPLITS = (Split.TRAIN, Split.VALIDATION, Split.TEST)


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = ApplicationBootstrap.from_config(next(iter(_CHECKPOINT_CONFIGS.values())), device=device)
    evaluator = base.create_humanml3d_generation_evaluator()
    checkpoint_dir = Path(base.paths.checkpoints_dir) / "tokenizer"

    stems = [s for s in (args.only.split(",") if args.only else _CHECKPOINT_CONFIGS) if s]
    rows: list[tuple[str, float, float, float]] = []
    print(
        f"recon-FID at {args.clips} seed-2026-shuffled clips/split (equal N -> gaps comparable)\n"
    )
    print(f"{'checkpoint':14} {'train':>8} {'val':>8} {'test':>8} {'gap(t-tr)':>10} {'|t-v|':>8}")
    for stem in stems:
        path = checkpoint_dir / f"{stem}.pt"
        if not path.is_file():
            print(f"{stem:14} (no checkpoint, skipped)")
            continue
        app = ApplicationBootstrap.from_config(_CHECKPOINT_CONFIGS[stem], device=device)
        module = app.create_tokenizer_network()
        module.load_state_dict(torch.load(path, map_location=device))
        module.to(device).eval()
        repository = app.create_motion_repository()
        fids = {
            sp: evaluate_tokenizer_reconstruction(
                module,
                repository.root,
                repository.load_scaler(),
                max_frames=app.config.data.max_motion_len,
                downsample=app.config.tokenizer.downsample,
                device=device,
                max_clips=args.clips,
                split=sp,
                shuffle_seed=2026,
                recon_fid=evaluator.calculate_reconstruction_fid,
            )["recon_fid"]
            for sp in _SPLITS
        }
        gap = fids[Split.TEST] - fids[Split.TRAIN]
        tv = abs(fids[Split.TEST] - fids[Split.VALIDATION])
        rows.append(
            (stem, fids[Split.TRAIN], fids[Split.VALIDATION], fids[Split.TEST])
        )
        print(
            f"{stem:14} {fids[Split.TRAIN]:8.4f} {fids[Split.VALIDATION]:8.4f} "
            f"{fids[Split.TEST]:8.4f} "
            f"{gap:+10.4f} {tv:8.4f}"
        )

    md = Path(base.paths.outputs_dir) / "generalization.md"
    lines = [
        f"# Tokenizer generalization audit (recon-FID, {args.clips} clips/split, seed 2026)",
        "",
        "Equal sample size across splits (FID is N-biased); the **gap = test - train** is the overfit",
        "indicator, **|test - val|** flags any test-selection inflation. Absolute values differ from the",
        "full-test headline by design.",
        "",
        "| checkpoint | train | val | test | gap (test-train) | \\|test-val\\| |",
        "|---|---|---|---|---|---|",
    ]
    for stem, tr, va, te in rows:
        lines.append(
            f"| {stem} | {tr:.4f} | {va:.4f} | {te:.4f} | {te - tr:+.4f} | {abs(te - va):.4f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {md}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tokenizer generalization gap (train/val/test recon-FID)."
    )
    p.add_argument(
        "--clips", type=int, default=1000, help="equal clip count per split (FID is N-biased)"
    )
    p.add_argument("--only", default="", help="comma-separated checkpoint stems (default: all)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
