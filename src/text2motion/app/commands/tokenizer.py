from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.run_log import log_metrics, start_run
from text2motion.motion.contracts import Split
from text2motion.tokenization.contracts import TokenizerKind, TokenizerTrainingRequest
from text2motion.tokenization.metrics import evaluate_tokenizer_reconstruction
from text2motion.tokenization.trainer import (
    TokenizerTrainer,
    train_tokenizer,
)


def run_train_tokenizer(app: ApplicationContext, args: argparse.Namespace) -> None:
    config = replace(app.config, tokenizer=replace(app.config.tokenizer, kind=args.tokenizer))
    app.config = config

    module = app.create_tokenizer_network().to(app.device)
    commitment = (
        config.rvq_baseline.commitment_beta if args.tokenizer is TokenizerKind.RVQ else 0.0
    )
    trainer = TokenizerTrainer(
        module,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        ema_decay=args.ema_decay,
        commit_beta=commitment,
    )
    repository = app.create_motion_repository()
    loader = repository.window_batch_loader(
        Split.TRAIN, args.window, args.batch_size, num_workers=args.num_workers
    )
    evaluator = app.create_humanml3d_generation_evaluator()
    scaler = repository.load_scaler()

    def evaluate(split: Split) -> dict[str, float]:
        return evaluate_tokenizer_reconstruction(
            module,
            repository.root,
            scaler,
            max_frames=config.data.max_motion_len,
            downsample=config.tokenizer.downsample,
            device=app.device,
            max_clips=args.max_eval_clips,
            split=split,
            recon_fid=evaluator.calculate_reconstruction_fid,
        )

    name = args.ckpt_name or f"tokenizer_{args.tokenizer.value}.pt"
    checkpoint_dir = app.checkpoint_directory("tokenizer")
    run_dir = start_run(
        f"tokenizer_{Path(name).stem}", config, app.config.paths.logs_dir, vars(args)
    )
    print(
        f"train clips: {len(loader.dataset)}  device: {app.device}  "
        f"codebook: {module.codebook_size}"
    )
    train_tokenizer(
        trainer,
        loader,
        TokenizerTrainingRequest(
            epochs=args.epochs,
            window=args.window,
            batch_size=args.batch_size,
            ema_decay=args.ema_decay,
            eval_every=args.eval_every,
            max_eval_clips=args.max_eval_clips,
            num_workers=args.num_workers,
            checkpoint_name=name,
            resume=args.resume,
        ),
        device=app.device,
        evaluate=evaluate,
        checkpoint_path=checkpoint_dir / name,
        resume_path=checkpoint_dir / f"{Path(name).stem}_last.pt",
        on_metrics=lambda record: log_metrics(run_dir, record),
    )

    best_path = checkpoint_dir / name
    if not best_path.is_file():
        return
    module.load_state_dict(torch.load(best_path, map_location=app.device))
    module.eval()
    metrics = evaluate(Split.TEST)
    log_metrics(
        run_dir,
        {"epoch": args.epochs, "split": Split.TEST.value, "final": True, **metrics},
    )
    print(
        f"  [TEST once | val-selected best] clips {metrics['clips']}  "
        f"recon-FID {metrics['recon_fid']:.4f}  MPJPE {metrics['mpjpe_mm']:.1f}mm"
    )
