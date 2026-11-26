from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.app.requests.data import TrainTokenizerRequest
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.run_log import log_metrics, resumed_run_dir, start_run
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.dataset.training.batching import dataset_size
from text2motion.motion.contracts import RunOutputs, Split
from text2motion.tokenization.contracts import (
    TokenizerKind,
    TokenizerSchedule,
)
from text2motion.tokenization.reconstruction_metrics import (
    ReconstructionSetup,
    evaluate_tokenizer_reconstruction,
)
from text2motion.tokenization.tokenizer_trainer import (
    TokenizerTrainer,
    train_tokenizer,
)

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


def run_train_tokenizer(
    app: ApplicationContext, args: TrainTokenizerRequest, final_split: Split = Split.TEST
) -> None:
    require_protocol_run(app.seed, args.exploratory, "train-tokenizer")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )
    app = with_tokenizer_kind(app, args.tokenizer)
    config = app.config
    trainer = create_tokenizer_trainer(app, args)
    module = trainer.tokenizer_network
    repository = app.create_motion_repository()
    if args.amass_share > 0:
        if args.amass_features is None:
            raise ValueError("amass_features must be set when amass_share is above 0")
        loader = repository.mixed_window_batch_loader(
            args.window,
            args.batch_size,
            args.amass_features,
            args.amass_share,
            num_workers=args.num_workers,
        )
    else:
        loader = repository.window_batch_loader(
            Split.TRAIN, args.window, args.batch_size, num_workers=args.num_workers
        )
    evaluator = app.create_humanml3d_generation_evaluator()
    scaler = repository.load_scaler()

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

    name = args.ckpt_name or f"tokenizer_{args.tokenizer.value}.pt"
    checkpoint_dir = app.checkpoint_directory("tokenizer")
    resume_path = checkpoint_dir / f"{Path(name).stem}_last.pt"
    run_dir = start_run(
        f"tokenizer_{Path(name).stem}",
        config,
        app.config.paths.logs_dir,
        args.model_dump(),
        existing_run_dir=resumed_run_dir(resume_path, None) if args.resume else None,
    )
    logger.info(
        f"train clips: {dataset_size(loader)}  device: {app.device}  "
        f"codebook: {module.codebook_size}"
    )
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
        checkpoint_path=checkpoint_dir / name,
        resume_path=resume_path,
        outputs=RunOutputs(
            run_dir=run_dir,
            on_metrics=lambda record: log_metrics(run_dir, record),
            failure_dump_path=app.failure_dump_path(run_dir),
        ),
    )

    best_path = checkpoint_dir / name
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
