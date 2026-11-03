from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationContext, required_path
from text2motion.app.checkpoint import GeneratorBundle, OverfitGate
from text2motion.app.run_log import (
    log_metrics,
    resumed_run_dir,
    run_dir_to_continue,
    start_run,
)
from text2motion.generation.contracts import (
    GeneratorPretrainingRequest,
    GeneratorTrainingRequest,
    MixedPrecisionMode,
    PretrainRequest,
    SamplingConfig,
    SanityOverfitRequest,
    TrainGeneratorRequest,
)
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.generation.pretrain import pretrain_generator, pretraining_resume_path
from text2motion.generation.trainer import (
    GeneratorTrainer,
    overfit_single_batch,
    train_generator,
)
from text2motion.motion.contracts import PrepareRequest, Split
from text2motion.motion.datasets import dataset_size
from text2motion.motion.preparation import (
    build_amass_pretraining_corpus,
    convert_official_release,
    heldout_feature_names,
    run_hml3d_preparation,
)
from text2motion.motion.storage import MEAN_FILE, SPLIT_FILES, STD_FILE, sha256_file
from text2motion.tokenization.contracts import (
    TokenizeRequest,
    TokenizerKind,
    TrainTokenizerRequest,
)
from text2motion.tokenization.corpus import (
    build_token_pack,
    check_pack_precision,
    read_pack_identity,
)
from text2motion.tokenization.metrics import evaluate_tokenizer_reconstruction
from text2motion.tokenization.trainer import (
    TokenizerTrainer,
    train_tokenizer,
)

logger = logging.getLogger(__name__)


def run_prepare(app: ApplicationContext, args: PrepareRequest) -> None:
    if args.source_release:
        convert_official_release(Path(args.source_release), Path(args.converted_dir))
        return

    request = app.create_hml3d_preparation_request()
    if not args.corpus:
        regen_dir = Path(args.regen_dir)
        if regen_dir.resolve() == Path(request.out_dir).resolve():
            raise ValueError(
                f"--regen_dir {regen_dir} is the training dataset (paths.hml3d_out_dir); "
                "prepare writes only into a separate regeneration folder"
            )
        run_hml3d_preparation(
            replace(request, out_dir=regen_dir), args.stage, args.adopt_pose_folder
        )
        return

    run_dir = start_run("pretrain_corpus", app.config, app.config.paths.logs_dir, args.model_dump())
    summary = build_amass_pretraining_corpus(
        request, Path(args.regen_dir), Path(args.corpus_dir), args.min_frames, args.limit
    )
    log_metrics(run_dir, summary)
    logger.info(summary)


def run_tokenize(app: ApplicationContext, request: TokenizeRequest) -> None:
    paths = app.config.paths
    index_csv = required_path(paths.hml3d_index_csv, "hml3d_index_csv")
    dataset_dir = required_path(paths.hml3d_out_dir, "hml3d_out_dir")
    checkpoint = request.tokenizer_ckpt or paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(checkpoint)
    excluded = heldout_feature_names(index_csv, dataset_dir, Path(request.regen_dir) / "pose_data")
    identity: dict[str, object] = {
        "tokenizer_sha256": sha256_file(checkpoint),
        "precision": app.config.precision.value,
        "mean_sha256": sha256_file(dataset_dir / MEAN_FILE),
        "std_sha256": sha256_file(dataset_dir / STD_FILE),
        "index_sha256": sha256_file(index_csv),
        "val_split_sha256": sha256_file(dataset_dir / SPLIT_FILES[Split.VALIDATION]),
        "test_split_sha256": sha256_file(dataset_dir / SPLIT_FILES[Split.TEST]),
    }
    run_dir = start_run("tokenize_corpus", app.config, paths.logs_dir, request.model_dump())
    summary = build_token_pack(
        tokenizer,
        app.create_motion_repository().load_scaler(),
        request,
        excluded,
        identity,
    )
    log_metrics(run_dir, summary)
    logger.info(summary)


def run_train_tokenizer(app: ApplicationContext, args: TrainTokenizerRequest) -> None:
    config = replace(app.config, tokenizer=replace(app.config.tokenizer, kind=args.tokenizer))
    app.config = config

    module = app.create_tokenizer_network().to(app.device)
    commitment = (
        config.rvq_baseline.commitment_beta
        if args.tokenizer is TokenizerKind.RESIDUAL_VECTOR
        else 0.0
    )
    trainer = TokenizerTrainer(
        module,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        ema_decay=args.ema_decay,
        commitment_beta=commitment,
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
            reconstruction_fid_scorer=evaluator.calculate_reconstruction_fid,
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
        args,
        device=app.device,
        evaluate=evaluate,
        checkpoint_path=checkpoint_dir / name,
        resume_path=resume_path,
        on_metrics=lambda record: log_metrics(run_dir, record),
        run_dir=run_dir,
        failure_dump_path=app.failure_dump_path(run_dir),
    )

    best_path = checkpoint_dir / name
    if not best_path.is_file():
        return
    module.load_state_dict(torch.load(best_path, map_location=app.device, weights_only=True))
    module.eval()
    metrics = evaluate(Split.TEST)
    log_metrics(
        run_dir,
        {"epoch": args.epochs, "split": Split.TEST.value, "final": True, **metrics},
    )
    logger.info(
        f"  [TEST once | val-selected best] clips {metrics['clips']}  "
        f"recon-FID {metrics['recon_fid']:.4f}  MPJPE {metrics['mpjpe_mm']:.1f}mm"
    )


def run_sanity_overfit(app: ApplicationContext, args: SanityOverfitRequest) -> None:
    repository = app.create_motion_repository()
    batch = next(iter(repository.captioned_batch_loader(Split.TRAIN, args.batch_size)))
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    architecture = app.resolve_generator_spec(args.backbone, tokenizer.codebook_size, dropout=0.0)
    tokenizer.network.requires_grad_(False)
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
        tokenizer.network,
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
        steps=int(metrics["steps"]),
        first_total=metrics["first_total"],
        final_total=metrics["final_total"],
        token_accuracy=metrics["token_accuracy"],
        ce=metrics["ce"],
    )
    gate.validate_metrics()
    gate.save(args.overfit_gate)
    logger.info(f"PASS: wrote real-batch overfit gate -> {args.overfit_gate}")


def run_train_generator(app: ApplicationContext, args: TrainGeneratorRequest) -> None:
    tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(tokenizer_ckpt)
    architecture = app.resolve_generator_spec(args.backbone, tokenizer.codebook_size)
    OverfitGate.load(args.overfit_gate).verify_setup(
        config_path=args.config,
        tokenizer_ckpt=tokenizer_ckpt,
        backbone=args.backbone,
        resolved_config=architecture.as_dict(),
    )
    logger.info(f"verified mandatory real-batch overfit gate: {args.overfit_gate}")

    module = architecture.create_model(app.device)
    if args.init_ckpt:
        module.load_state_dict(torch.load(args.init_ckpt, map_location=app.device))
        logger.info(f"initialised generator from pretrained {args.init_ckpt}")

    text_encoder = app.load_text_encoder()
    repository = app.create_motion_repository()
    loader = repository.captioned_batch_loader(
        Split.TRAIN, args.batch_size, num_workers=args.num_workers
    )
    scaler = repository.load_scaler()
    trainer = GeneratorTrainer(
        module,
        tokenizer.network,
        replace(app.config.train, grad_accum=args.grad_accum),
        downsample=app.config.tokenizer.downsample,
        text_encoder=text_encoder,
        mean=scaler.mean,
        std=scaler.std,
    )
    trainer.build_scheduler(args.epochs * max(1, -(-len(loader) // args.grad_accum)))
    logger.info(
        f"generator params: {sum(p.numel() for p in module.parameters()):,} "
        f"({architecture.architecture_label})"
    )

    name = args.ckpt_name or f"generator_{args.backbone.value}.pt"
    request = GeneratorTrainingRequest(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        max_eval_clips=args.max_eval_clips,
        eval_batch_size=args.eval_batch_size,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        checkpoint_name=name,
        resume=args.resume,
        fresh_selection=args.fresh_selection,
        patience=args.patience,
        accept_legacy_resume=args.accept_legacy_resume,
        extend_schedule=args.extend_schedule,
    )
    if args.fresh_selection and not args.resume:
        raise SystemExit("--fresh_selection only applies to a resumed run (pass --resume)")
    evaluator = app.create_humanml3d_generation_evaluator(args.vocab_dir)
    pipeline = TextToMotionGenerator(module, text_encoder, tokenizer)
    sampling = SamplingConfig(temperature=request.temperature, cfg_scale=request.cfg_scale)
    checkpoint_dir = app.checkpoint_directory("generator")
    checkpoint_path = checkpoint_dir / name
    resume_path = checkpoint_dir / f"{Path(name).stem}_last.pt"
    requested_run_dir = Path(args.resume_run_dir) if args.resume_run_dir else None
    run_dir = start_run(
        f"generator_{args.backbone.value}",
        app.config,
        app.config.paths.logs_dir,
        extra=args.model_dump(),
        existing_run_dir=run_dir_to_continue(
            args.resume, args.fresh_selection, resume_path, requested_run_dir
        ),
    )
    log_metrics(run_dir, {"train_clips": dataset_size(loader), "device": app.device})
    logger.info(
        f"train clips: {dataset_size(loader)}  device: {app.device}  backbone: {args.backbone.value}"
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
            tokenizer_ckpt=tokenizer_ckpt,
            resolved_generator_config=architecture.as_dict(),
            epoch=epoch,
            validation_fid=validation_fid,
        ).save(checkpoint_path)
        logger.info(f"  saved atomic best bundle -> {checkpoint_path} (FID {validation_fid:.4f})")

    train_generator(
        trainer,
        loader,
        request,
        device=app.device,
        evaluate=evaluate,
        save_best=save_best,
        resume_path=resume_path,
        run_dir=run_dir,
        on_metrics=lambda record: log_metrics(run_dir, record),
        failure_dump_path=app.failure_dump_path(run_dir),
    )


def run_pretrain(app: ApplicationContext, args: PretrainRequest) -> None:
    from text2motion.tokenization.contracts import TokenizerKind

    tokenizer_checkpoint = app.config.paths.tokenizer_checkpoint
    pack_identity = read_pack_identity(args.token_pack)
    if pack_identity["tokenizer_sha256"] != sha256_file(tokenizer_checkpoint):
        raise ValueError(
            f"{args.token_pack} was encoded with another tokenizer than {tokenizer_checkpoint}"
        )
    check_pack_precision(pack_identity, app.config.precision.value)
    if app.config.tokenizer.kind == TokenizerKind.RESIDUAL_VECTOR:
        vocab = app.config.rvq_baseline.codebook_size
    else:
        vocab = 1
        for level in app.config.tokenizer.fsq_levels:
            vocab *= level

    architecture = app.resolve_generator_spec(args.backbone, vocab)
    module = architecture.create_model(app.device)
    logger.info(
        f"pretrain {args.backbone.value}: {sum(p.numel() for p in module.parameters()):,} params, "
        f"codebooks {architecture.config.num_codebooks} x vocab {vocab}"
    )
    out_path = Path(
        args.ckpt
        or app.checkpoint_directory("generator") / f"generator_{args.backbone.value}_pretrained.pt"
    )
    run_dir = start_run(
        f"pretrain_{args.backbone.value}",
        app.config,
        app.config.paths.logs_dir,
        args.model_dump(),
        existing_run_dir=(
            resumed_run_dir(pretraining_resume_path(out_path), None) if args.resume else None
        ),
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
            accept_legacy_resume=args.accept_legacy_resume,
            extend_schedule=args.extend_schedule,
        ),
        train=app.config.train,
        device=app.device,
        out_path=out_path,
        on_metrics=lambda record: log_metrics(run_dir, record),
        seed=app.config.seed,
        run_dir=run_dir,
        failure_dump_path=app.failure_dump_path(run_dir),
    )
