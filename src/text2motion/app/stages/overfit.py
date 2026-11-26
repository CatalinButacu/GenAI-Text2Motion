from __future__ import annotations

import logging
from dataclasses import replace

from text2motion.app.requests.generator import (
    SanityOverfitRequest,
)
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.overfit_gate import OverfitGate
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.app.stages.generator import initial_generator, overfit_criteria
from text2motion.generation.contracts import (
    MixedPrecisionMode,
)
from text2motion.generation.generator_trainer import (
    GeneratorTrainer,
    overfit_single_batch,
)
from text2motion.motion.contracts import Split
from text2motion.motion.storage import sha256_file

logger = logging.getLogger(__name__)


def run_sanity_overfit(app: ApplicationContext, args: SanityOverfitRequest) -> None:
    require_protocol_run(app.seed, args.exploratory, "sanity-overfit")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )
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
        initial_generator(architecture, app.device, args.init_ckpt),
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
        overfit_criteria(args.init_ckpt),
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
    gate.validate_metrics(overfit_criteria(args.init_ckpt))
    gate.save(args.overfit_gate)
    logger.info(f"PASS: wrote real-batch overfit gate -> {args.overfit_gate}")
