from __future__ import annotations

import argparse
from pathlib import Path

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.run_log import log_metrics, start_run
from text2motion.motion.preparation import (
    build_amass_pretraining_corpus,
    convert_official_release,
    run_hml3d_preparation,
)
from text2motion.tokenization.contracts import TokenPackRequest
from text2motion.tokenization.corpus import build_token_pack


def run_prepare(app: ApplicationContext, args: argparse.Namespace) -> None:
    if args.source_release:
        convert_official_release(Path(args.source_release), Path(args.out))
        return

    request = app.create_hml3d_preparation_request()
    if not args.corpus:
        run_hml3d_preparation(request, args.stage)
        return

    run_dir = start_run("pretrain_corpus", app.config, app.config.paths.logs_dir, vars(args))
    summary = build_amass_pretraining_corpus(
        request, Path(args.regen_dir), Path(args.out_dir), args.min_frames, args.limit
    )
    log_metrics(run_dir, summary)
    print(summary)


def run_tokenize(app: ApplicationContext, args: argparse.Namespace) -> None:
    checkpoint = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(checkpoint)
    run_dir = start_run("tokenize_corpus", app.config, app.config.paths.logs_dir, vars(args))
    summary = build_token_pack(
        tokenizer,
        app.create_motion_repository().load_scaler(),
        TokenPackRequest(
            features_dir=Path(args.features_dir),
            out_path=Path(args.out),
            segment_frames=args.segment_frames,
            stride=args.stride,
        ),
    )
    log_metrics(run_dir, summary)
    print(summary)
