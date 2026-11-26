from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from text2motion.app.requests.data import PrepareRequest, TokenizeRequest
from text2motion.app.runtime.context import ApplicationContext, required_path
from text2motion.app.runtime.run_log import log_metrics, start_run
from text2motion.dataset.pipeline import run_pipeline
from text2motion.dataset.sources.amass.corpus import (
    build_amass_pretraining_corpus,
    heldout_feature_names,
)
from text2motion.dataset.sources.humanml3d.official import convert_official_release
from text2motion.motion.contracts import Split
from text2motion.motion.storage import MEAN_FILE, SPLIT_FILES, STD_FILE, sha256_file
from text2motion.tokenization.contracts import (
    TokenPackPlan,
)
from text2motion.tokenization.corpus import (
    build_token_pack,
)

logger = logging.getLogger(__name__)


PROTOCOL_SEED = 2026


PROTOCOL_SPLIT_RECORD = "protocol_splits.json"


def require_protocol_run(seed: int, exploratory: bool, command: str) -> None:
    if exploratory:
        logger.warning(f"{command}: exploratory run with seed {seed}; the result is not citable")
        return
    if seed != PROTOCOL_SEED:
        raise ValueError(
            f"{command} would use seed {seed}; a protocol run uses seed {PROTOCOL_SEED} "
            "(pass --exploratory to record a non-protocol run)"
        )


def freeze_protocol_splits(dataset_dir: Path) -> dict[str, str]:
    record = {name: sha256_file(dataset_dir / name) for name in SPLIT_FILES.values()}
    (dataset_dir / PROTOCOL_SPLIT_RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def verify_protocol_splits(dataset_dir: Path, exploratory: bool) -> None:
    if exploratory:
        return
    record_path = dataset_dir / PROTOCOL_SPLIT_RECORD
    if not record_path.is_file():
        raise FileNotFoundError(
            f"{record_path} is missing; freeze the splits before a protocol run"
        )
    frozen = json.loads(record_path.read_text(encoding="utf-8"))
    for name in SPLIT_FILES.values():
        actual = sha256_file(dataset_dir / name)
        if frozen.get(name) != actual:
            raise ValueError(
                f"split file {name} changed since it was frozen (hash is now {actual})"
            )


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
        run_pipeline(
            replace(request, out_dir=regen_dir, adopt_pose_folder=args.adopt_pose_folder),
            args.stage,
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
        TokenPackPlan(
            request.features_dir, request.token_pack, request.segment_frames, request.stride
        ),
        excluded,
        identity,
    )
    log_metrics(run_dir, summary)
    logger.info(summary)
