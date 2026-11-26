from __future__ import annotations

import importlib.util
import logging
from dataclasses import replace
from pathlib import Path
from typing import cast

from text2motion.app.requests.generator import (
    TrainGeneratorRequest,
)
from text2motion.app.runtime.checkpoint import GeneratorBundle, load_generator_checkpoint
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.overfit_gate import OverfitGate
from text2motion.app.runtime.run_log import log_metrics, run_dir_to_continue, start_run
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.dataset.training.batching import dataset_size
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.generation.contracts import (
    DEFAULT_OVERFIT_CRITERIA,
    PRETRAINED_OVERFIT_CRITERIA,
    OverfitCriteria,
    SamplingConfig,
)
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.epochs import train_generator
from text2motion.generation.training.trainer import CleanCrossEntropyLoaders, GeneratorTrainer
from text2motion.motion.contracts import RunOutputs, Split

logger = logging.getLogger(__name__)


def initial_generator(
    architecture: GeneratorModelSpec, device: str, init_ckpt: Path | None
) -> MotionTokenGenerator:
    module = architecture.create_model(device)
    if init_ckpt is None:
        return module
    generator_state, _, _ = load_generator_checkpoint(init_ckpt, map_location=device)
    module.load_state_dict(generator_state)
    logger.info(f"initialised generator from pretrained {init_ckpt}")
    return module


def compile_generator(module: MotionTokenGenerator, device: str) -> bool:
    if not str(device).startswith("cuda") or importlib.util.find_spec("triton") is None:
        logger.warning("--compile needs CUDA and Triton; the generator runs eager")
        return False
    module.compile()
    return True


def run_train_generator(app: ApplicationContext, args: TrainGeneratorRequest) -> None:
    require_protocol_run(app.seed, args.exploratory, "train-generator")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )
    tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(tokenizer_ckpt)
    architecture = app.resolve_generator_spec(args.backbone, tokenizer.codebook_size)
    OverfitGate.load(args.overfit_gate).verify_setup(
        config_path=args.config,
        tokenizer_ckpt=tokenizer_ckpt,
        backbone=args.backbone,
        resolved_config=architecture.as_dict(),
        criteria=overfit_criteria(args.init_ckpt),
    )
    logger.info(f"verified mandatory real-batch overfit gate: {args.overfit_gate}")

    module = initial_generator(architecture, app.device, args.init_ckpt)
    compiled = args.compile and compile_generator(module, app.device)
    logger.info(f"torch.compile: {'on' if compiled else 'off'}")

    text_encoder = app.load_text_encoder()
    repository = app.create_motion_repository()
    loader = repository.captioned_batch_loader(
        Split.TRAIN, args.batch_size, num_workers=args.num_workers
    )
    cache = (
        app.attach_text_cache(
            text_encoder, cast(CaptionedMotionDataset, loader.dataset).captions(), args.batch_size
        )
        if args.text_cache
        else None
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
        clean_ce=args.clean_ce,
        ce_batches=args.ce_batches,
        val_ce_patience=args.val_ce_patience,
        stop_epoch=args.stop_epoch,
        select_on_val_ce=args.select_on_val_ce,
        save_r1_floor=args.save_r1_floor,
    )
    ce_loaders = None
    if request.clean_ce:
        ce_loaders = CleanCrossEntropyLoaders(
            train=repository.captioned_batch_loader(
                Split.TRAIN, request.eval_batch_size, shuffle=False
            ),
            validation=repository.captioned_batch_loader(
                Split.VALIDATION, request.eval_batch_size, shuffle=False
            ),
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
        outputs=RunOutputs(
            run_dir=run_dir,
            on_metrics=lambda record: log_metrics(run_dir, record),
            failure_dump_path=app.failure_dump_path(run_dir),
        ),
        ce_loaders=ce_loaders,
    )
    if cache is not None:
        logger.info(f"text cache hits {cache.hits} misses {cache.misses}")


def overfit_criteria(init_ckpt: Path | None) -> OverfitCriteria:
    return DEFAULT_OVERFIT_CRITERIA if init_ckpt is None else PRETRAINED_OVERFIT_CRITERIA
