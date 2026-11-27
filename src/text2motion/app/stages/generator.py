from __future__ import annotations

import importlib.util
import logging
from dataclasses import replace
from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader

from text2motion.app.requests.generator import (
    TrainGeneratorRequest,
)
from text2motion.app.runtime.checkpoint import GeneratorBundle, load_generator_checkpoint
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.overfit_gate import OverfitGate
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.app.stages.generator_training import (
    GeneratorSetup,
    build_ce_loaders,
    build_training_request,
    make_evaluate,
    make_run_outputs,
    open_generator_run,
)
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.dataset.training.module import MotionDataModule
from text2motion.generation.contracts import (
    DEFAULT_OVERFIT_CRITERIA,
    PRETRAINED_OVERFIT_CRITERIA,
    OverfitCriteria,
)
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.model.text_encoder import CLIPTextEncoder, FrozenTextStates
from text2motion.generation.training.epochs import CheckpointSink, train_generator
from text2motion.generation.training.trainer import GeneratorTrainer
from text2motion.motion.contracts import Split
from text2motion.tokenization.network.frozen import FrozenMotionTokenizer

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
    setup = prepare_generator(app, args)
    fit_generator(app, args, setup)


def prepare_generator(app: ApplicationContext, args: TrainGeneratorRequest) -> GeneratorSetup:
    verify_generator_inputs(app, args)
    tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.models.load_tokenizer(tokenizer_ckpt)
    architecture = app.models.resolve_generator_spec(args.backbone, tokenizer.codebook_size)
    verify_overfit_gate(args, tokenizer_ckpt, architecture)
    module = build_generator_module(app, args, architecture)
    text_encoder = app.models.load_text_encoder()
    repository = app.create_motion_repository()
    loader = repository.captioned_batch_loader(
        Split.TRAIN, args.batch_size, num_workers=args.num_workers
    )
    cache = attach_caption_cache(app, args, text_encoder, loader)
    trainer = make_generator_trainer(app, args, module, tokenizer, text_encoder, repository)
    schedule_generator_trainer(trainer, args, loader, module, architecture)
    return GeneratorSetup(
        tokenizer_ckpt,
        tokenizer,
        architecture,
        module,
        text_encoder,
        repository,
        loader,
        cache,
        trainer,
    )


def verify_generator_inputs(app: ApplicationContext, args: TrainGeneratorRequest) -> None:
    require_protocol_run(app.seed, args.exploratory, "train-generator")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )


def verify_overfit_gate(
    args: TrainGeneratorRequest, tokenizer_ckpt: str | Path, architecture: GeneratorModelSpec
) -> None:
    OverfitGate.load(args.overfit_gate).verify_setup(
        config_path=args.config,
        tokenizer_ckpt=tokenizer_ckpt,
        backbone=args.backbone,
        resolved_config=architecture.as_dict(),
        criteria=overfit_criteria(args.init_ckpt),
    )
    logger.info(f"verified mandatory real-batch overfit gate: {args.overfit_gate}")


def build_generator_module(
    app: ApplicationContext, args: TrainGeneratorRequest, architecture: GeneratorModelSpec
) -> MotionTokenGenerator:
    module = initial_generator(architecture, app.device, args.init_ckpt)
    compiled = args.compile and compile_generator(module, app.device)
    logger.info(f"torch.compile: {'on' if compiled else 'off'}")
    return module


def attach_caption_cache(
    app: ApplicationContext,
    args: TrainGeneratorRequest,
    text_encoder: CLIPTextEncoder,
    loader: DataLoader,
) -> FrozenTextStates | None:
    return (
        app.models.attach_text_cache(
            text_encoder, cast(CaptionedMotionDataset, loader.dataset).captions(), args.batch_size
        )
        if args.text_cache
        else None
    )


def make_generator_trainer(
    app: ApplicationContext,
    args: TrainGeneratorRequest,
    module: MotionTokenGenerator,
    tokenizer: FrozenMotionTokenizer,
    text_encoder: CLIPTextEncoder,
    repository: MotionDataModule,
) -> GeneratorTrainer:
    scaler = repository.load_scaler()
    return GeneratorTrainer(
        module,
        tokenizer.network,
        replace(app.config.train, grad_accum=args.grad_accum),
        downsample=app.config.tokenizer.downsample,
        text_encoder=text_encoder,
        mean=scaler.mean,
        std=scaler.std,
    )


def schedule_generator_trainer(
    trainer: GeneratorTrainer,
    args: TrainGeneratorRequest,
    loader: DataLoader,
    module: MotionTokenGenerator,
    architecture: GeneratorModelSpec,
) -> None:
    trainer.build_scheduler(args.epochs * max(1, -(-len(loader) // args.grad_accum)))
    logger.info(
        f"generator params: {sum(p.numel() for p in module.parameters()):,} "
        f"({architecture.architecture_label})"
    )


def fit_generator(
    app: ApplicationContext, args: TrainGeneratorRequest, setup: GeneratorSetup
) -> None:
    name = args.ckpt_name or f"generator_{args.backbone.value}.pt"
    request = build_training_request(args, name)
    ce_loaders = build_ce_loaders(setup.repository, request)
    if args.fresh_selection and not args.resume:
        raise SystemExit("--fresh_selection only applies to a resumed run (pass --resume)")
    evaluate = make_evaluate(app, args, setup, request)
    checkpoint_dir = app.checkpoint_directory("generator")
    checkpoint_path = checkpoint_dir / name
    resume_path = checkpoint_dir / f"{Path(name).stem}_last.pt"
    run_dir = open_generator_run(app, args, setup.loader, resume_path)
    train_generator(
        setup.trainer,
        setup.loader,
        request,
        device=app.device,
        evaluate=evaluate,
        save_best=make_save_best(args, setup, checkpoint_path),
        resume_path=resume_path,
        outputs=make_run_outputs(app, run_dir),
        ce_loaders=ce_loaders,
    )
    if setup.cache is not None:
        logger.info(f"text cache hits {setup.cache.hits} misses {setup.cache.misses}")


def make_save_best(
    args: TrainGeneratorRequest, setup: GeneratorSetup, checkpoint_path: Path
) -> CheckpointSink:
    def save_best(epoch: int, validation_fid: float) -> None:
        GeneratorBundle.capture(
            setup.module,
            setup.text_encoder,
            backbone=args.backbone,
            tokenizer_ckpt=setup.tokenizer_ckpt,
            resolved_generator_config=setup.architecture.as_dict(),
            epoch=epoch,
            validation_fid=validation_fid,
        ).save(checkpoint_path)
        logger.info(f"  saved atomic best bundle -> {checkpoint_path} (FID {validation_fid:.4f})")

    return save_best


def overfit_criteria(init_ckpt: Path | None) -> OverfitCriteria:
    return DEFAULT_OVERFIT_CRITERIA if init_ckpt is None else PRETRAINED_OVERFIT_CRITERIA
