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
from src.modules.motion.training import trainAmass, trainHumanml3d, trainUnified
from src.shared.constants import (
    SSM_D_MODEL,
    SSM_D_STATE,
    SSM_N_LAYERS,
)
from src.shared.run_ctx import initWandb, logGpuSanity, makeRunDir, snapshotConfig

NPZ = "*.npz"

def checkPrereqsAmass(d: Path) -> None:
    npz = list(d.rglob(NPZ)) if d.exists() else []
    print(f"[INFO] AMASS: {len(npz)} .npz files at {d}")
    hml = Path("data/humanml3d/texts")
    if hml.exists():
        print(f"[INFO] HumanML3D enrichment: {len(list(hml.glob('*.txt')))} annotation files")
    else:
        print("[INFO] Run 'python scripts/data/download_humanml3d.py' for text enrichment")
    if not npz:
        print("[WARNING] No .npz files found --training will use synthetic data.")

def checkPrereqsUnified() -> None:
    cfg = TrainingConfig()
    dirs = {
        "AMASS": Path(cfg.dataDir),
        "ARCTIC": Path(cfg.arcticDataDir),
    }
    for name, p in dirs.items():
        if p.exists():
            n = len(list(p.rglob(NPZ))) + len(list(p.rglob("*.pkl")))
            print(f"[INFO] {name}: {p} ({n} files)")
        else:
            print(f"[WARNING] {name} directory not found at {p} -- will be skipped")

def checkPrereqsHumanml3d(d: Path) -> None:
    texts = d / "texts"
    splitDir = d / "split"
    indexCsv = d / "index.csv"
    amassDir = Path(TrainingConfig().dataDir)
    amassNpz = list(amassDir.rglob(NPZ)) if amassDir.exists() else []

    if not texts.exists():
        print(f"[ERROR] {texts} not found.")
        print("        Run: python scripts/data/download_humanml3d.py")
        sys.exit(1)
    if not splitDir.exists():
        print(f"[ERROR] {splitDir} not found.")
        print("        Run: python scripts/data/download_humanml3d.py")
        sys.exit(1)
    if not indexCsv.exists():
        print(f"[ERROR] {indexCsv} not found.")
        print("        HumanML3D SMPL-X training uses index.csv to map texts onto AMASS clips.")
        sys.exit(1)
    if not amassNpz:
        print(f"[ERROR] No AMASS .npz files found at {amassDir}.")
        print("        HumanML3D mode now loads native SMPL-X motion from AMASS, not motion_data/.")
        sys.exit(1)

    nT = len(list(texts.glob("*.txt")))
    print(f"[INFO] HumanML3D annotations: {nT} text files, index={indexCsv}")
    print(f"[INFO] AMASS backing store: {len(amassNpz)} .npz files at {amassDir}")

def checkPrereqs(source: str, dataDir: str) -> None:
    d = Path(dataDir)
    if source == "amass":
        checkPrereqsAmass(d)
    elif source == "unified":
        checkPrereqsUnified()
    else:
        checkPrereqsHumanml3d(d)

def main():
    parser = argparse.ArgumentParser(
        description="Train TextToMotionSSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-source", choices=["amass", "humanml3d", "unified"],
                        default="amass", dest="dataSource")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory. Default: data/AMASS or data/humanml3d annotations",
    dest="dataDir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32, dest="batchSize")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, dest="checkpointDir")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'latest'")
    parser.add_argument(
        "--warm-start", action="store_true", dest="warmStart",
        help="With --resume: load model weights only, skip optimizer/scheduler/epoch state. "
             "Use to change LR or other hyperparams while keeping pretrained weights.",
    )
    parser.add_argument("--d-model", type=int, default=SSM_D_MODEL, dest="dModel")
    parser.add_argument("--d-state", type=int, default=SSM_D_STATE, dest="dState")
    parser.add_argument("--n-layers", type=int, default=SSM_N_LAYERS, dest="nLayers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 = non-deterministic)")
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weightDecay")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process, safe on Windows)",
    dest="numWorkers")
    # ---- Architecture flags (Phase 2 & 3) ----
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help="Use frozen SentenceTransformer encoder (recommended for new runs)",
    dest="useSbert")
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace model name for SBERT encoder",
    dest="sbertModel")
    parser.add_argument(
        "--no-freeze-sbert",
        action="store_true",
        help="Fine-tune SBERT weights instead of freezing them",
    dest="noFreezeSbert")
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
    dest="gradientCheckpointing")
    parser.add_argument(
        "--use-film",
        action="store_true",
        help="FiLM conditioning at every SSM layer (text scale+shift). "
        "Recommended for new runs: text is injected at every layer, not just input.",
    dest="useFilm")
    parser.add_argument(
        "--max-motion-length",
        type=int,
        default=200,
        help="Maximum motion sequence length in frames (default: 200 = 6.7s at 30fps).",
    dest="maxMotionLength")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size (None = all). Useful for smoke tests.",
    dest="maxSamples")
    parser.add_argument(
        "--rvq-checkpoint",
        type=str,
        default="checkpoints/rvq_tokenizer/best_model.pt",
        help="Path to the frozen RVQ tokenizer checkpoint (trained first).",
    dest="rvqCheckpoint")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["amass", "arctic", "humanml3d", "interx"],
        default=None,
        help="Data sources for unified training. Defaults to [amass, arctic]. "
             "Example: --sources amass humanml3d interx",
    )
    parser.add_argument("--arctic-dir", type=str, default="data/arctic/unpack",
                        dest="arcticDir",
                        help="ARCTIC dataset root (used when --sources includes arctic)")
    parser.add_argument("--humanml3d-dir", type=str, default="data/humanml3d",
                        dest="humanml3dDir",
                        help="HumanML3D root (used when --sources includes humanml3d)")
    parser.add_argument("--interx-dir", type=str, default="data/inter-x",
                        dest="interxDir",
                        help="Inter-X root (used when --sources includes interx)")
    parser.add_argument("--amass-dir", type=str, default="data/AMASS",
                        dest="amassDir",
                        help="AMASS backing store path (used as the motion source "
                             "for HumanML3D, and for unified mode amass)")
    args = parser.parse_args()

    # Resolve defaults
    if args.dataDir is None:
        args.dataDir = "data/humanml3d" if args.dataSource == "humanml3d" else "data/AMASS"
    if args.checkpointDir is None:
        suffix = "_hml3d" if args.dataSource == "humanml3d" else ""
        args.checkpointDir = f"checkpoints/motion_ssm{suffix}"

    # Per-run directory: checkpoints/<base>/<run_id>/ -- isolates artifacts, configs, logs
    runBase = Path(args.checkpointDir)
    run_dir, run_id = makeRunDir(str(runBase))
    args.checkpointDir = str(run_dir)
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
        args.dataSource,
        args.dataDir,
        args.epochs,
        args.device,
        args.batchSize,
    )

    checkPrereqs(args.dataSource, args.dataDir)

    config = TrainingConfig(
        dataDir=args.dataDir,
        batchSize=args.batchSize,
        learningRate=args.lr,
        numEpochs=args.epochs,
        device=args.device,
        checkpointDir=args.checkpointDir,
        resumeFrom=args.resume,
        dModel=args.dModel,
        dState=args.dState,
        nLayers=args.nLayers,
        seed=None if args.seed == -1 else args.seed,
        weightDecay=args.weightDecay,
        numWorkers=args.numWorkers,
        useSbert=args.useSbert,
        sbertModel=args.sbertModel,
        freezeSbert=not args.noFreezeSbert,
        bidirectional=args.bidirectional,
        gradientCheckpointing=args.gradientCheckpointing,
        useFilm=args.useFilm,
        maxMotionLength=args.maxMotionLength,
        maxSamples=args.maxSamples,
        rvqCheckpointPath=args.rvqCheckpoint,
        warmStart=args.warmStart,
    )
    if args.sources is not None and args.dataSource == "unified":
        config.unifiedSources = args.sources
    # Pass per-source data dirs through so unifiedFactory can find them
    config.arcticDataDir = args.arcticDir
    config.humanml3dDir = args.humanml3dDir
    config.interxDir = args.interxDir
    # In humanml3d mode the AMASS backing store is separate from --data-dir
    # (which points at HumanML3D texts/indices). Use --amass-dir for that.
    config.amassDir = args.amassDir if args.dataSource == "humanml3d" else args.dataDir
    log.info(
        "arch: use_sbert=%s  bidirectional=%s  use_film=%s  grad_ckpt=%s  "
        "d_model=%d  n_layers=%d  max_motion_length=%d  rvq_ckpt=%s",
        config.useSbert,
        config.bidirectional,
        config.useFilm,
        config.gradientCheckpointing,
        config.dModel,
        config.nLayers,
        config.maxMotionLength,
        config.rvqCheckpointPath,
    )

    log.info("[train_motion_ssm] run_id=%s  run_dir=%s", run_id, run_dir)
    logGpuSanity()
    snapshotConfig(run_dir, config)
    initWandb(
        project="motion_ssm",
        runId=run_id,
        runDir=run_dir,
        config=config,
        tags=[args.dataSource],
    )

    dispatch = {"humanml3d": trainHumanml3d, "unified": trainUnified, "amass": trainAmass}
    best = dispatch[args.dataSource](config)

    log.info(
        "Training complete. best_val_loss=%.4f  checkpoint: %s/best_model.pt",
        best,
        args.checkpointDir,
    )

if __name__ == "__main__":
    main()
