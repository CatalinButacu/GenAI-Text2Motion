from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.checkpoint import GeneratorBundle, OverfitGate, sha256_file
from text2motion.app.run_log import log_metrics, start_run
from text2motion.generation.contracts import (
    GeneratorPretrainingRequest,
    GeneratorTrainingRequest,
    MixedPrecisionMode,
    SamplingConfig,
)
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.generation.pretrain import pretrain_generator
from text2motion.generation.trainer import (
    GeneratorTrainer,
    overfit_single_batch,
    train_generator,
)
from text2motion.motion.contracts import Split


def load_tokenizer_and_generator_spec(app: ApplicationContext, args: argparse.Namespace, dropout: float | None = None):
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    overrides = {} if dropout is None else {"dropout": dropout}
    architecture = app.resolve_generator_spec(args.backbone, tokenizer.codebook_size, **overrides)
    return tokenizer, architecture


def run_sanity_overfit(app: ApplicationContext, args: argparse.Namespace) -> None:
    repository = app.create_motion_repository()
    batch = next(iter(repository.captioned_batch_loader(Split.TRAIN, args.batch_size)))
    tokenizer, architecture = load_tokenizer_and_generator_spec(app, args, dropout=0.0)
    tokenizer.tokenizer_model.requires_grad_(False)
    scaler = repository.load_scaler()
    train_config = replace(
        app.config.train,
        lr=args.lr,
        cfg_dropout=0.0,
        pkeep=1.0,
        amp=MixedPrecisionMode.OFF,
        grad_accum=1,
        ema_decay=0.0,
    )
    trainer = GeneratorTrainer(
        architecture.create_model(app.device),
        tokenizer.tokenizer_model,
        train_config,
        downsample=app.config.tokenizer.downsample,
        text_encoder=app.load_text_encoder(),
        mean=scaler.mean,
        std=scaler.std,
    )
    metrics = overfit_single_batch(
        trainer,
        batch.features.to(app.device),
        batch.lengths.to(app.device),
        list(batch.captions),
        args.steps,
    )
    gate = OverfitGate(
        config_sha256=sha256_file(args.config),
        tokenizer_sha256=sha256_file(args.tokenizer_ckpt),
        backbone=args.backbone,
        resolved_generator_config=architecture.as_dict(),
        batch_size=args.batch_size,
        **metrics,
    )
    gate.validate_metrics()
    gate.save(args.out)
    print(f"PASS: wrote real-batch overfit gate -> {args.out}")


def run_train_generator(app: ApplicationContext, args: argparse.Namespace) -> None:
    tokenizer, architecture = load_tokenizer_and_generator_spec(app, args)
    OverfitGate.load(args.overfit_gate).verify_setup(
        config_path=args.config,
        tokenizer_ckpt=args.tokenizer_ckpt,
        backbone=args.backbone,
        resolved_config=architecture.as_dict(),
    )
    print(f"verified mandatory real-batch overfit gate: {args.overfit_gate}")

    module = architecture.create_model(app.device)
    if args.init_ckpt:
        module.load_state_dict(torch.load(args.init_ckpt, map_location=app.device))
        print(f"initialised generator from pretrained {args.init_ckpt}")

    text_encoder = app.load_text_encoder()
    repository = app.create_motion_repository()
    loader = repository.captioned_batch_loader(Split.TRAIN, args.batch_size)
    scaler = repository.load_scaler()
    trainer = GeneratorTrainer(
        module,
        tokenizer.tokenizer_model,
        replace(app.config.train, grad_accum=args.grad_accum),
        downsample=app.config.tokenizer.downsample,
        text_encoder=text_encoder,
        mean=scaler.mean,
        std=scaler.std,
    )
    trainer.build_scheduler(args.epochs * max(1, -(-len(loader) // args.grad_accum)))
    print(
        f"generator params: {sum(p.numel() for p in module.parameters()):,} "
        f"({architecture.architecture_label})"
    )

    name = args.ckpt_name or f"generator_{args.backbone.value}.pt"
    request = GeneratorTrainingRequest(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        eval_split=args.eval_split,
        max_eval_clips=args.max_eval_clips,
        eval_batch_size=args.eval_batch,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        checkpoint_name=name,
        resume=args.resume,
        patience=args.patience,
    )
    evaluator = app.create_humanml3d_generation_evaluator(args.our_vab_dir)
    pipeline = TextToMotionGenerator(module, text_encoder, tokenizer)
    sampling = SamplingConfig(temperature=request.temperature, cfg_scale=request.cfg_scale)
    checkpoint_dir = app.checkpoint_directory("generator")
    checkpoint_path = checkpoint_dir / name
    run_dir = start_run(
        f"generator_{args.backbone.value}",
        app.config,
        app.config.paths.logs_dir,
        extra=vars(args),
    )
    log_metrics(run_dir, {"train_clips": len(loader.dataset), "device": app.device})
    print(
        f"train clips: {len(loader.dataset)}  device: {app.device}  "
        f"backbone: {args.backbone.value}"
    )

    def evaluate(split: Split) -> dict[str, float]:
        return evaluator.evaluate_checkpoint_selection_metrics(
            pipeline,
            split=split,
            max_clips=request.max_eval_clips,
            sampling=sampling,
            batch_size=request.eval_batch_size,
        )

    def save_best(epoch: int, validation_fid: float) -> None:
        GeneratorBundle.capture(
            module,
            text_encoder,
            backbone=args.backbone,
            tokenizer_ckpt=args.tokenizer_ckpt,
            resolved_generator_config=architecture.as_dict(),
            epoch=epoch,
            validation_fid=validation_fid,
        ).save(checkpoint_path)
        print(f"  saved atomic best bundle -> {checkpoint_path} (FID {validation_fid:.4f})")

    train_generator(
        trainer,
        loader,
        request,
        device=app.device,
        evaluate=evaluate,
        save_best=save_best,
        resume_path=checkpoint_dir / f"{Path(name).stem}_last.pt",
        run_dir=run_dir,
        on_metrics=lambda record: log_metrics(run_dir, record),
    )


def run_pretrain(app: ApplicationContext, args: argparse.Namespace) -> None:
    vocab = 1
    for level in app.config.tokenizer.fsq_levels:
        vocab *= level

    architecture = app.resolve_generator_spec(args.backbone, vocab)
    module = architecture.create_model(app.device)
    print(
        f"pretrain {args.backbone.value}: {sum(p.numel() for p in module.parameters()):,} params, "
        f"codebooks {architecture.config.num_codebooks} x vocab {vocab}"
    )
    out_path = Path(
        args.out
        or app.checkpoint_directory("generator")
        / f"generator_{args.backbone.value}_pretrained.pt"
    )
    run_dir = start_run(
        f"pretrain_{args.backbone.value}", app.config, app.config.paths.logs_dir, vars(args)
    )
    pretrain_generator(
        architecture,
        module,
        GeneratorPretrainingRequest(
            token_pack=Path(args.token_pack),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            val_fraction=args.val_fraction,
            out_path=out_path,
            resume=args.resume,
        ),
        train=app.config.train,
        device=app.device,
        out_path=out_path,
        on_metrics=lambda record: log_metrics(run_dir, record),
        seed=app.config.seed,
    )
