from __future__ import annotations

import logging
from pathlib import Path

import torch

from text2motion.app.requests.evaluate import BenchmarkRequest, EvaluateRequest
from text2motion.app.runtime.context import ApplicationContext
from text2motion.app.runtime.run_log import log_metrics, start_run
from text2motion.evaluation.benchmark import run_streaming_state_latency_benchmark
from text2motion.evaluation.contracts import (
    GenerationEvaluationProtocol,
    GenerationEvaluationReport,
    GenerationLengthPolicy,
    StreamingBenchmarkProtocol,
)
from text2motion.generation.contracts import Backbone, BackboneChoice, SamplingConfig
from text2motion.motion.storage import listed_clip_ids

logger = logging.getLogger(__name__)


def run_evaluate(app: ApplicationContext, args: EvaluateRequest) -> None:
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.models.load_tokenizer(args.tokenizer_ckpt)
    evaluator = app.create_humanml3d_generation_evaluator()
    backbones = selected_backbones(args.backbone)
    request = build_evaluation_protocol(args)
    run_dir = start_run(
        f"evaluate_{args.split.value}", app.config, app.config.paths.logs_dir, args.model_dump()
    )
    log_evaluate_header(app, args, request)

    for backbone in backbones:
        checkpoint = resolve_checkpoint(app, args, backbone, len(backbones))
        if not Path(checkpoint).is_file():
            logger.info(f"{backbone.value:12s} SKIP (no checkpoint yet)")
            continue
        generator = app.models.load_text_to_motion_generator(
            backbone,
            checkpoint,
            tokenizer,
            tokenizer_checkpoint=args.tokenizer_ckpt,
            text_encoder_checkpoint=args.text_encoder_ckpt,
        )
        report = evaluator.run_generation_evaluation_protocol(generator, request)
        record_report(run_dir, backbone, checkpoint, report)
        del generator
        if app.device == "cuda":
            torch.cuda.empty_cache()


def selected_backbones(choice: BackboneChoice) -> list[Backbone]:
    if choice is BackboneChoice.BOTH:
        return [Backbone.TRANSFORMER, Backbone.MAMBA]
    return [Backbone(choice)]


def build_evaluation_protocol(args: EvaluateRequest) -> GenerationEvaluationProtocol:
    return GenerationEvaluationProtocol(
        split=args.split,
        max_clips=args.max_clips,
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            stop_at_end=args.length_mode is GenerationLengthPolicy.END_TOKEN,
        ),
        mm_clips=args.mm_clips,
        mm_repeats=args.mm_repeats,
        bootstrap=args.bootstrap,
        reps=args.reps,
        batch_size=args.eval_batch_size,
        clip_list=None if args.clip_list is None else tuple(listed_clip_ids(args.clip_list)),
    )


def resolve_checkpoint(
    app: ApplicationContext, args: EvaluateRequest, backbone: Backbone, backbone_count: int
) -> Path:
    checkpoint = args.ckpt if backbone_count == 1 else None
    return checkpoint or app.config.paths.generator_checkpoint(backbone)


def log_evaluate_header(
    app: ApplicationContext, args: EvaluateRequest, request: GenerationEvaluationProtocol
) -> None:
    logger.info(
        f"device {app.device}  split {args.split.value}  cfg_scale {args.cfg_scale}  "
        f"temp {args.temperature}  top_p {args.top_p}  length {args.length_mode.value}  "
        f"({request.reps} generation runs, our_vab)"
    )


def record_report(
    run_dir: Path, backbone: Backbone, checkpoint: Path, report: GenerationEvaluationReport
) -> None:
    log_metrics(run_dir, {"backbone": backbone.value, "ckpt": str(checkpoint), **report.as_dict()})
    log_report_summary(backbone, report)


def log_report_summary(backbone: Backbone, report: GenerationEvaluationReport) -> None:
    top1, top2, top3 = report.r_precision
    top1_ci, top2_ci, top3_ci = report.r_precision_ci95
    logger.info(
        f"{backbone.value:12s} clips {report.clips:4d}  runs {report.runs}  "
        f"FID {report.fid:6.3f}+/-{report.fid_ci95:.3f}  "
        f"R@1 {top1:.3f}+/-{top1_ci:.3f}  R@2 {top2:.3f}+/-{top2_ci:.3f}  "
        f"R@3 {top3:.3f}+/-{top3_ci:.3f}  "
        f"MM {report.matching_distance:.3f}+/-{report.matching_distance_ci95:.3f}  "
        f"Div {report.diversity:.3f}+/-{report.diversity_ci95:.3f}  "
        f"MModality {report.multimodality:.3f}"
    )


def run_bench(app: ApplicationContext, request: BenchmarkRequest) -> None:
    run_streaming_state_latency_benchmark(
        StreamingBenchmarkProtocol(
            horizons=request.horizons,
            warmup=request.warmup,
            streams=request.streams,
            streams_horizon=request.streams_horizon,
            out_path=request.report or app.config.paths.benchmark_report,
            label=str(request.config),
        ),
        generator=app.config.generator,
        codebook_size=app.config.generator.codebook_size,
        num_codebooks=app.config.tokenizer.num_quantizers,
        seed=app.config.seed,
        device=app.device,
    )
