from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from text2motion.app.bootstrap import ApplicationContext, required_path
from text2motion.app.run_log import log_metrics, start_run
from text2motion.motion.contracts import PrepareRequest, Split
from text2motion.motion.preparation import (
    build_amass_pretraining_corpus,
    convert_official_release,
    heldout_feature_names,
    run_hml3d_preparation,
)
from text2motion.motion.storage import MEAN_FILE, SPLIT_FILES, STD_FILE, sha256_file
from text2motion.tokenization.contracts import TokenizeRequest, TokenPackRequest
from text2motion.tokenization.corpus import build_token_pack

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
        TokenPackRequest(
            features_dir=request.features_dir,
            out_path=request.token_pack,
            segment_frames=request.segment_frames,
            stride=request.stride,
        ),
        excluded,
        identity,
    )
    log_metrics(run_dir, summary)
    logger.info(summary)
