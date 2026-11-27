from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from text2motion.app.requests.data import TrainTokenizerRequest
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.run_log import log_metrics, resumed_run_dir, start_run
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.dataset.training.batching import dataset_size
from text2motion.dataset.training.module import MotionDataModule
from text2motion.evaluation.evaluator import HumanMl3dGenerationEvaluator
from text2motion.motion.contracts import RunOutputs, Split
from text2motion.motion.normalization import MotionScaler
from text2motion.tokenization.contracts import (
    TokenizerKind,
    TokenizerSchedule,
)
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.reconstruction_metrics import (
    ReconstructionSetup,
    evaluate_tokenizer_reconstruction,
)
from text2motion.tokenization.training.loop import train_tokenizer
from text2motion.tokenization.training.trainer import TokenizerTrainer

logger = logging.getLogger(__name__)


def with_tokenizer_kind(app: ApplicationContext, kind: TokenizerKind) -> ApplicationContext:
    tokenizer = replace(app.config.tokenizer, kind=kind)
    return replace(app, config=replace(app.config, tokenizer=tokenizer))


def create_tokenizer_trainer(
    app: ApplicationContext, args: TrainTokenizerRequest
) -> TokenizerTrainer:
    config = app.config
    module = app.models.create_tokenizer_network().to(app.device)
    commitment = (
        config.rvq_baseline.commitment_beta
        if args.tokenizer is TokenizerKind.RESIDUAL_VECTOR
        else 0.0
    )
    return TokenizerTrainer(
        module,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        ema_decay=args.ema_decay,
        commitment_beta=commitment,
    )


def require_tokenizer_protocol(app: ApplicationContext, args: TrainTokenizerRequest) -> None:
    require_protocol_run(app.seed, args.exploratory, "train-tokenizer")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )


def create_training_loader(repository: MotionDataModule, args: TrainTokenizerRequest) -> DataLoader:
    if args.amass_share > 0:
        if args.amass_features is None:
            raise ValueError("amass_features must be set when amass_share is above 0")
        return repository.mixed_window_batch_loader(
            args.window,
            args.batch_size,
            args.amass_features,
            args.amass_share,
            num_workers=args.num_workers,
        )
    return repository.window_batch_loader(
        Split.TRAIN, args.window, args.batch_size, num_workers=args.num_workers
    )


def build_reconstruction_evaluator(
    app: ApplicationContext,
    module: TokenizerNetwork,
    repository: MotionDataModule,
    evaluator: HumanMl3dGenerationEvaluator,
    scaler: MotionScaler,
    args: TrainTokenizerRequest,
) -> Callable[[Split], dict[str, float]]:
    config = app.config

    def evaluate(split: Split) -> dict[str, float]:
        return evaluate_tokenizer_reconstruction(
            module,
            ReconstructionSetup(
                root=repository.root,
                scaler=scaler,
                max_frames=config.data.max_motion_len,
                device=app.device,
                max_clips=args.max_eval_clips,
                split=split,
                reconstruction_fid_scorer=evaluator.calculate_reconstruction_fid,
            ),
            downsample=config.tokenizer.downsample,
        )

    return evaluate


def tokenizer_checkpoint_paths(
    app: ApplicationContext, args: TrainTokenizerRequest
) -> tuple[str, Path, Path]:
    name = args.ckpt_name or f"tokenizer_{args.tokenizer.value}.pt"
    checkpoint_dir = app.checkpoint_directory("tokenizer")
    resume_path = checkpoint_dir / f"{Path(name).stem}_last.pt"
    return name, checkpoint_dir / name, resume_path


def start_tokenizer_run(
    app: ApplicationContext, args: TrainTokenizerRequest, name: str, resume_path: Path
) -> Path:
    return start_run(
        f"tokenizer_{Path(name).stem}",
        app.config,
        app.config.paths.logs_dir,
        args.model_dump(),
        existing_run_dir=resumed_run_dir(resume_path, None) if args.resume else None,
    )


def log_tokenizer_setup(
    app: ApplicationContext, loader: DataLoader, module: TokenizerNetwork
) -> None:
    logger.info(
        f"train clips: {dataset_size(loader)}  device: {app.device}  "
        f"codebook: {module.codebook_size}"
    )


def fit_tokenizer(
    app: ApplicationContext,
    trainer: TokenizerTrainer,
    loader: DataLoader,
    args: TrainTokenizerRequest,
    evaluate: Callable[[Split], dict[str, float]],
    checkpoint_path: Path,
    resume_path: Path,
    run_dir: Path,
) -> None:
    train_tokenizer(
        trainer,
        loader,
        TokenizerSchedule(
            args.epochs,
            args.eval_every,
            args.resume,
            args.accept_legacy_resume,
            args.max_steps,
        ),
        device=app.device,
        evaluate=evaluate,
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
        outputs=RunOutputs(
            run_dir=run_dir,
            on_metrics=lambda record: log_metrics(run_dir, record),
            failure_dump_path=app.failure_dump_path(run_dir),
        ),
    )


def evaluate_best_checkpoint(
    app: ApplicationContext,
    module: TokenizerNetwork,
    evaluate: Callable[[Split], dict[str, float]],
    best_path: Path,
    run_dir: Path,
    args: TrainTokenizerRequest,
    final_split: Split,
) -> None:
    if not best_path.is_file():
        return
    module.load_state_dict(torch.load(best_path, map_location=app.device, weights_only=True))
    module.eval()
    metrics = evaluate(final_split)
    log_metrics(
        run_dir,
        {"epoch": args.epochs, "split": final_split.value, "final": True, **metrics},
    )
    logger.info(
        f"  [{final_split.value} once | val-selected best] clips {metrics['clips']}  "
        f"recon-FID {metrics['recon_fid']:.4f}  MPJPE {metrics['mpjpe_mm']:.1f}mm"
    )


def run_train_tokenizer(
    app: ApplicationContext, args: TrainTokenizerRequest, final_split: Split = Split.TEST
) -> None:
    require_tokenizer_protocol(app, args)
    app = with_tokenizer_kind(app, args.tokenizer)
    trainer = create_tokenizer_trainer(app, args)
    module = trainer.tokenizer_network
    repository = app.create_motion_repository()
    loader = create_training_loader(repository, args)
    evaluator = app.create_humanml3d_generation_evaluator()
    evaluate = build_reconstruction_evaluator(
        app, module, repository, evaluator, repository.load_scaler(), args
    )
    name, checkpoint_path, resume_path = tokenizer_checkpoint_paths(app, args)
    run_dir = start_tokenizer_run(app, args, name, resume_path)
    log_tokenizer_setup(app, loader, module)
    fit_tokenizer(app, trainer, loader, args, evaluate, checkpoint_path, resume_path, run_dir)
    evaluate_best_checkpoint(app, module, evaluate, checkpoint_path, run_dir, args, final_split)
