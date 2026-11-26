from __future__ import annotations

import logging
from pathlib import Path

from text2motion.app.requests.generator import (
    PretrainRequest,
)
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.run_log import log_metrics, resumed_run_dir, start_run
from text2motion.app.stages.data import require_protocol_run, verify_protocol_splits
from text2motion.generation.requests import GeneratorPretrainingRequest
from text2motion.generation.training.pretrain import pretrain_generator, pretraining_resume_path
from text2motion.motion.contracts import RunOutputs
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.corpus import (
    check_pack_precision,
    read_pack_identity,
)

logger = logging.getLogger(__name__)


def run_pretrain(app: ApplicationContext, args: PretrainRequest) -> None:
    from text2motion.tokenization.contracts import TokenizerKind

    require_protocol_run(app.seed, args.exploratory, "pretrain")
    verify_protocol_splits(
        required_path(app.config.paths.hml3d_out_dir, "hml3d_out_dir"), args.exploratory
    )

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
        outputs=RunOutputs(
            run_dir=run_dir,
            on_metrics=lambda record: log_metrics(run_dir, record),
            failure_dump_path=app.failure_dump_path(run_dir),
        ),
        seed=app.config.seed,
    )
