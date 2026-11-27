from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from text2motion.app.requests.generator import TrainGeneratorRequest
from text2motion.app.runtime.context import ApplicationContext
from text2motion.app.runtime.run_log import log_metrics, run_dir_to_continue, start_run
from text2motion.dataset.training.batching import dataset_size
from text2motion.dataset.training.module import MotionDataModule
from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.model.text_encoder import CLIPTextEncoder, FrozenTextStates
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.epochs import Evaluator
from text2motion.generation.training.trainer import CleanCrossEntropyLoaders, GeneratorTrainer
from text2motion.motion.contracts import RunOutputs, Split
from text2motion.tokenization.network.frozen import FrozenMotionTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorSetup:
    tokenizer_ckpt: str | Path
    tokenizer: FrozenMotionTokenizer
    architecture: GeneratorModelSpec
    module: MotionTokenGenerator
    text_encoder: CLIPTextEncoder
    repository: MotionDataModule
    loader: DataLoader
    cache: FrozenTextStates | None
    trainer: GeneratorTrainer


def build_training_request(args: TrainGeneratorRequest, name: str) -> GeneratorTrainingRequest:
    return GeneratorTrainingRequest(
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


def build_ce_loaders(
    repository: MotionDataModule, request: GeneratorTrainingRequest
) -> CleanCrossEntropyLoaders | None:
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
    return ce_loaders


def open_generator_run(
    app: ApplicationContext, args: TrainGeneratorRequest, loader: DataLoader, resume_path: Path
) -> Path:
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
    return run_dir


def make_evaluate(
    app: ApplicationContext,
    args: TrainGeneratorRequest,
    setup: GeneratorSetup,
    request: GeneratorTrainingRequest,
) -> Evaluator:
    evaluator = app.create_humanml3d_generation_evaluator(args.vocab_dir)
    pipeline = TextToMotionGenerator(setup.module, setup.text_encoder, setup.tokenizer)
    sampling = SamplingConfig(temperature=request.temperature, cfg_scale=request.cfg_scale)

    def evaluate(split: Split) -> dict[str, float]:
        return evaluator.evaluate_checkpoint_selection_metrics(
            pipeline,
            split=split,
            max_clips=request.max_eval_clips,
            sampling=sampling,
            batch_size=request.eval_batch_size,
        )

    return evaluate


def make_run_outputs(app: ApplicationContext, run_dir: Path) -> RunOutputs:
    return RunOutputs(
        run_dir=run_dir,
        on_metrics=lambda record: log_metrics(run_dir, record),
        failure_dump_path=app.failure_dump_path(run_dir),
    )
