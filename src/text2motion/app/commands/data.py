from __future__ import annotations

import logging
from pathlib import Path

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.run_log import log_metrics, start_run
from text2motion.motion.contracts import PrepareRequest
from text2motion.motion.preparation import (
    build_amass_pretraining_corpus,
    convert_official_release,
    run_hml3d_preparation,
)
from text2motion.tokenization.contracts import TokenizeRequest, TokenPackRequest
from text2motion.tokenization.corpus import build_token_pack

logger = logging.getLogger(__name__)


def run_prepare(app: ApplicationContext, args: PrepareRequest) -> None:
    if args.source_release:
        convert_official_release(Path(args.source_release), Path(args.converted_dir))
        return

    request = app.create_hml3d_preparation_request()
    if not args.corpus:
        run_hml3d_preparation(request, args.stage)
        return

    run_dir = start_run("pretrain_corpus", app.config, app.config.paths.logs_dir, args.model_dump())
    summary = build_amass_pretraining_corpus(
        request, Path(args.regen_dir), Path(args.corpus_dir), args.min_frames, args.limit
    )
    log_metrics(run_dir, summary)
    logger.info(summary)


def run_tokenize(app: ApplicationContext, request: TokenizeRequest) -> None:
    checkpoint = request.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(checkpoint)
    run_dir = start_run(
        "tokenize_corpus", app.config, app.config.paths.logs_dir, request.model_dump()
    )
    summary = build_token_pack(
        tokenizer,
        app.create_motion_repository().load_scaler(),
        TokenPackRequest(
            features_dir=request.features_dir,
            out_path=request.token_pack,
            segment_frames=request.segment_frames,
            stride=request.stride,
        ),
    )
    log_metrics(run_dir, summary)
    logger.info(summary)
