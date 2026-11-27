from __future__ import annotations

import logging
from dataclasses import replace

from text2motion.app.config.schema import GeneratorTrainingConfig
from text2motion.app.requests.generator import (
    SanityOverfitRequest,
)
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.overfit_gate import OverfitGate
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.app.stages.generator import initial_generator, overfit_criteria
from text2motion.dataset.training.module import MotionDataModule
from text2motion.generation.contracts import (
    MixedPrecisionMode,
)
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.training.overfit import overfit_single_batch
from text2motion.generation.training.trainer import GeneratorTrainer
from text2motion.motion.contracts import Split
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.network.frozen import FrozenMotionTokenizer

logger = logging.getLogger(__name__)


def check_overfit_protocol(app: ApplicationContext, args: SanityOverfitRequest) -> None:
    require_protocol_run(app.seed, args.exploratory, "sanity-overfit")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )


def overfit_train_config(
    app: ApplicationContext, args: SanityOverfitRequest
) -> GeneratorTrainingConfig:
    return replace(
        app.config.train,
        lr=args.lr,
        cfg_dropout=0.0,
        pkeep=1.0,
        amp=MixedPrecisionMode.OFF,
        grad_accum=1,
        ema_decay=0.0,
    )


def make_overfit_trainer(
    app: ApplicationContext,
    args: SanityOverfitRequest,
    repository: MotionDataModule,
    tokenizer: FrozenMotionTokenizer,
    architecture: GeneratorModelSpec,
) -> GeneratorTrainer:
    scaler = repository.load_scaler()
    return GeneratorTrainer(
        initial_generator(architecture, app.device, args.init_ckpt),
        tokenizer.network,
        overfit_train_config(app, args),
        downsample=app.config.tokenizer.downsample,
        text_encoder=app.models.load_text_encoder(),
        mean=scaler.mean,
        std=scaler.std,
    )


def make_overfit_gate(
    args: SanityOverfitRequest, architecture: GeneratorModelSpec, metrics: dict[str, float]
) -> OverfitGate:
    return OverfitGate(
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


def run_sanity_overfit(app: ApplicationContext, args: SanityOverfitRequest) -> None:
    check_overfit_protocol(app, args)
    repository = app.create_motion_repository()
    batch = next(iter(repository.captioned_batch_loader(Split.TRAIN, args.batch_size)))
    tokenizer = app.models.load_tokenizer(args.tokenizer_ckpt)
    architecture = app.models.resolve_generator_spec(
        args.backbone, tokenizer.codebook_size, dropout=0.0
    )
    tokenizer.network.requires_grad_(False)
    trainer = make_overfit_trainer(app, args, repository, tokenizer, architecture)
    metrics = overfit_single_batch(
        trainer,
        batch.features.to(app.device),
        batch.lengths.to(app.device),
        list(batch.captions),
        args.steps,
        overfit_criteria(args.init_ckpt),
    )
    gate = make_overfit_gate(args, architecture, metrics)
    gate.validate_metrics(overfit_criteria(args.init_ckpt))
    gate.save(args.overfit_gate)
    logger.info(f"PASS: wrote real-batch overfit gate -> {args.overfit_gate}")
