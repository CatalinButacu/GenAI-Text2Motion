#!/usr/bin/env python
"""Train TextToMotionSSM.

Two data sources are supported:

  amass (default)
    Trains on AMASS .npz files (native 168-dim SMPL-X).
    HumanML3D text annotations are used automatically for label enrichment
    if data/humanml3d/texts/ exists.
    Command:
        python scripts/training/train_motion_ssm.py --data-dir data/AMASS

  humanml3d
        Trains on AMASS SMPL-X clips selected by HumanML3D texts/splits/index mappings.
        Requires data/humanml3d/{texts,split,index.csv} plus local AMASS .npz files.
        Provides richer, real text supervision without carrying the legacy 272-dim feature path.
    Command:
        python scripts/training/train_motion_ssm.py --data-source humanml3d

Key architecture flags:
  --use-sbert        Use frozen SentenceTransformer instead of SimpleTextEncoder.
                     Recommended for new training runs (better text understanding).
  --bidirectional    Use BiMambaLayer (forward + backward scan) instead of MambaLayer.
                     Inspired by Motion Mamba (ECCV 2024) BSM block.

Checkpoints are saved to checkpoints/motion_ssm/ (amass) or
checkpoints/motion_ssm_hml3d/ (humanml3d) and are compatible with
SSMMotionGenerator.load_checkpoint() for inference.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from src.modules.motion.config import TrainingConfig
from src.modules.motion.training import train_amass, train_humanml3d, train_unified
from src.shared.constants import (
    SSM_D_MODEL,
    SSM_D_STATE,
    SSM_N_LAYERS,
)
from src.shared.run_ctx import init_wandb, log_gpu_sanity, make_run_dir, snapshot_config

NPZ = "*.npz"

def check_prereqs_amass(d: Path) -> None:
    npz = list(d.rglob(NPZ)) if d.exists() else []
    print(f"[INFO] AMASS: {len(npz)} .npz files at {d}")
    hml = Path("data/humanml3d/texts")
    if hml.exists():
        print(f"[INFO] HumanML3D enrichment: {len(list(hml.glob('*.txt')))} annotation files")
    else:
        print("[INFO] Run 'python scripts/data/download_humanml3d.py' for text enrichment")
    if not npz:
        print("[WARNING] No .npz files found --training will use synthetic data.")

def check_prereqs_unified() -> None:
    cfg = TrainingConfig()
    dirs = {
        "AMASS": Path(cfg.data_dir),
        "ARCTIC": Path(cfg.arctic_data_dir),
    }
    for name, p in dirs.items():
        if p.exists():
            n = len(list(p.rglob(NPZ))) + len(list(p.rglob("*.pkl")))
            print(f"[INFO] {name}: {p} ({n} files)")
        else:
            print(f"[WARNING] {name} directory not found at {p} -- will be skipped")

def check_prereqs_humanml3d(d: Path) -> None:
    texts = d / "texts"
    split_dir = d / "split"
    index_csv = d / "index.csv"
    amass_dir = Path(TrainingConfig().data_dir)
    amass_npz = list(amass_dir.rglob(NPZ)) if amass_dir.exists() else []

    if not texts.exists():
        print(f"[ERROR] {texts} not found.")
        print("        Run: python scripts/data/download_humanml3d.py")
        sys.exit(1)
    if not split_dir.exists():
        print(f"[ERROR] {split_dir} not found.")
        print("        Run: python scripts/data/download_humanml3d.py")
        sys.exit(1)
    if not index_csv.exists():
        print(f"[ERROR] {index_csv} not found.")
        print("        HumanML3D SMPL-X training uses index.csv to map texts onto AMASS clips.")
        sys.exit(1)
    if not amass_npz:
        print(f"[ERROR] No AMASS .npz files found at {amass_dir}.")
        print("        HumanML3D mode now loads native SMPL-X motion from AMASS, not motion_data/.")
        sys.exit(1)

    n_t = len(list(texts.glob("*.txt")))
    print(f"[INFO] HumanML3D annotations: {n_t} text files, index={index_csv}")
    print(f"[INFO] AMASS backing store: {len(amass_npz)} .npz files at {amass_dir}")

def check_prereqs(source: str, data_dir: str) -> None:
    d = Path(data_dir)
    if source == "amass":
        check_prereqs_amass(d)
    elif source == "unified":
        check_prereqs_unified()
    else:
        check_prereqs_humanml3d(d)

def main():
    parser = argparse.ArgumentParser(
        description="Train TextToMotionSSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-source", choices=["amass", "humanml3d", "unified"],
                        default="amass", dest="data_source")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory. Default: data/AMASS or data/humanml3d annotations",
    dest="data_dir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, dest="checkpoint_dir")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'latest'")
    parser.add_argument(
        "--warm-start", action="store_true", dest="warm_start",
        help="With --resume: load model weights only, skip optimizer/scheduler/epoch state. "
             "Use to change LR or other hyperparams while keeping pretrained weights.",
    )
    parser.add_argument(
        "--compile", action="store_true", dest="compile_model",
        help=("Wrap the model in torch.compile(mode='reduce-overhead', dynamic=False). "
              "Expected 3-5x training throughput on GPU; ignored on CPU. First batch "
              "compiles for 30-90s; amortizes after ~3 epochs."),
    )
    parser.add_argument(
        "--ar-k-head", action="store_true", dest="ar_k_head",
        help=("Use the autoregressive K-codebook head (ResidualKHead) instead of "
              "the legacy independent K-classifier head (RVQMotionDecoder). "
              "Each codebook conditions on the embedded sum of prior-codebook "
              "tokens, restoring the residual structure of RVQ. Requires training "
              "from scratch -- old checkpoints have the independent head's weights."),
    )
    parser.add_argument("--d-model", type=int, default=SSM_D_MODEL, dest="d_model")
    parser.add_argument("--d-state", type=int, default=SSM_D_STATE, dest="d_state")
    parser.add_argument("--n-layers", type=int, default=SSM_N_LAYERS, dest="n_layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 = non-deterministic)")
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process, safe on Windows)",
    dest="num_workers")
    # ---- Architecture flags (Phase 2 & 3) ----
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help="Use frozen SentenceTransformer encoder (recommended for new runs)",
    dest="use_sbert")
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help=(
            "sentence-transformers model name for the text encoder. "
            "Convenience aliases via --text-encoder override this."
        ),
    dest="sbert_model")
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=None,
        choices=[None, "sbert-small", "sbert-mpnet", "clip-b", "clip-l"],
        help=(
            "Convenience alias for --sbert-model. Maps:\n"
            "  sbert-small  -> all-MiniLM-L6-v2  (384-d, 22M params, baseline)\n"
            "  sbert-mpnet  -> all-mpnet-base-v2 (768-d, 110M)\n"
            "  clip-b       -> clip-ViT-B-32     (512-d, OpenAI CLIP; best for motion verbs)\n"
            "  clip-l       -> clip-ViT-L-14     (768-d, larger CLIP)"
        ),
        dest="text_encoder",
    )
    parser.add_argument(
        "--no-freeze-sbert",
        action="store_true",
        help="Fine-tune SBERT weights instead of freezing them",
    dest="no_freeze_sbert")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use BiMambaLayer (fwd+bwd scan) instead of MambaLayer",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute SSM activations during backward (halves VRAM, +33%% time). "
        "Required for 4 GB GPUs with bidirectional or large batch.",
    dest="gradient_checkpointing")
    parser.add_argument(
        "--use-film",
        action="store_true",
        help="FiLM conditioning at every SSM layer (text scale+shift). "
        "Recommended for new runs: text is injected at every layer, not just input.",
    dest="use_film")
    parser.add_argument(
        "--max-motion-length",
        type=int,
        default=200,
        help="Maximum motion sequence length in frames (default: 200 = 6.7s at 30fps).",
    dest="max_motion_length")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size (None = all). Useful for smoke tests.",
    dest="max_samples")
    parser.add_argument(
        "--rvq-checkpoint",
        type=str,
        default="checkpoints/rvq_tokenizer/best_model.pt",
        help="Path to the frozen RVQ tokenizer checkpoint (trained first).",
    dest="rvq_checkpoint")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["amass", "arctic", "humanml3d", "interx"],
        default=None,
        help="Data sources for unified training. Defaults to [amass, arctic]. "
             "Example: --sources amass humanml3d interx",
    )
    parser.add_argument("--arctic-dir", type=str, default="data/arctic/unpack",
                        dest="arctic_dir",
                        help="ARCTIC dataset root (used when --sources includes arctic)")
    parser.add_argument("--humanml3d-dir", type=str, default="data/humanml3d",
                        dest="humanml3d_dir",
                        help="HumanML3D root (used when --sources includes humanml3d)")
    parser.add_argument("--interx-dir", type=str, default="data/inter-x",
                        dest="interx_dir",
                        help="Inter-X root (used when --sources includes interx)")
    parser.add_argument("--amass-dir", type=str, default="data/AMASS",
                        dest="amass_dir",
                        help="AMASS backing store path (used as the motion source "
                             "for HumanML3D, and for unified mode amass)")
    args = parser.parse_args()

    # Resolve defaults
    # --text-encoder alias overrides --sbert-model if both supplied.
    TEXT_ENCODER_ALIASES = {
        "sbert-small": "all-MiniLM-L6-v2",
        "sbert-mpnet": "all-mpnet-base-v2",
        "clip-b":      "clip-ViT-B-32",
        "clip-l":      "clip-ViT-L-14",
    }
    if args.text_encoder is not None:
        args.sbert_model = TEXT_ENCODER_ALIASES[args.text_encoder]
    if args.data_dir is None:
        args.data_dir = "data/humanml3d" if args.data_source == "humanml3d" else "data/AMASS"
    if args.checkpoint_dir is None:
        suffix = "_hml3d" if args.data_source == "humanml3d" else ""
        args.checkpoint_dir = f"checkpoints/motion_ssm{suffix}"

    # Per-run directory: checkpoints/<base>/<run_id>/ -- isolates artifacts, configs, logs
    run_base = Path(args.checkpoint_dir)
    run_dir, run_id = make_run_dir(str(run_base))
    args.checkpoint_dir = str(run_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "training.log", mode="a"),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(
        "data_source=%s  data_dir=%s  epochs=%d  device=%s  batch=%d",
        args.data_source,
        args.data_dir,
        args.epochs,
        args.device,
        args.batch_size,
    )

    check_prereqs(args.data_source, args.data_dir)

    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        seed=None if args.seed == -1 else args.seed,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_sbert=args.use_sbert,
        sbert_model=args.sbert_model,
        freeze_sbert=not args.no_freeze_sbert,
        bidirectional=args.bidirectional,
        gradient_checkpointing=args.gradient_checkpointing,
        use_film=args.use_film,
        max_motion_length=args.max_motion_length,
        max_samples=args.max_samples,
        rvq_checkpoint_path=args.rvq_checkpoint,
        warm_start=args.warm_start,
        compile_model=args.compile_model,
        arch="residual_k" if args.ar_k_head else "independent",
    )
    if args.sources is not None and args.data_source == "unified":
        config.unified_sources = args.sources
    # Pass per-source data dirs through so unified_factory can find them
    config.arctic_data_dir = args.arctic_dir
    config.humanml3d_dir = args.humanml3d_dir
    config.interx_dir = args.interx_dir
    # In humanml3d mode the AMASS backing store is separate from --data-dir
    # (which points at HumanML3D texts/indices). Use --amass-dir for that.
    config.amass_dir = args.amass_dir if args.data_source == "humanml3d" else args.data_dir
    log.info(
        "arch: use_sbert=%s  bidirectional=%s  use_film=%s  grad_ckpt=%s  "
        "d_model=%d  n_layers=%d  max_motion_length=%d  rvq_ckpt=%s",
        config.use_sbert,
        config.bidirectional,
        config.use_film,
        config.gradient_checkpointing,
        config.d_model,
        config.n_layers,
        config.max_motion_length,
        config.rvq_checkpoint_path,
    )

    log.info("[train_motion_ssm] run_id=%s  run_dir=%s", run_id, run_dir)
    log_gpu_sanity()
    snapshot_config(run_dir, config)
    init_wandb(
        project="motion_ssm",
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        tags=[args.data_source],
    )

    dispatch = {"humanml3d": train_humanml3d, "unified": train_unified, "amass": train_amass}
    best = dispatch[args.data_source](config)

    log.info(
        "Training complete. best_val_loss=%.4f  checkpoint: %s/best_model.pt",
        best,
        args.checkpoint_dir,
    )

if __name__ == "__main__":
    main()
