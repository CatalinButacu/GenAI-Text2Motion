from __future__ import annotations

import argparse
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.run_log import log_metrics, start_run
from text2motion.evaluation.benchmark import run_streaming_state_latency_benchmark
from text2motion.evaluation.contracts import (
    GenerationEvaluationProtocol,
    GenerationLengthPolicy,
    StreamingBenchmarkProtocol,
)
from text2motion.generation.contracts import Backbone, BackboneChoice, SamplingConfig


def run_evaluate(app: ApplicationContext, args: argparse.Namespace) -> None:
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    evaluator = app.create_humanml3d_generation_evaluator()
    backbones = (
        [Backbone.TRANSFORMER, Backbone.MAMBA]
        if args.backbone is BackboneChoice.BOTH
        else [Backbone(args.backbone)]
    )
    request = GenerationEvaluationProtocol(
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
        batch_size=args.eval_batch,
    )
    run_dir = start_run(
        f"evaluate_{args.split.value}", app.config, app.config.paths.logs_dir, vars(args)
    )
    print(
        f"device {app.device}  split {args.split.value}  cfg_scale {args.cfg_scale}  "
        f"temp {args.temperature}  top_p {args.top_p}  length {args.length_mode.value}  "
        f"({request.reps}-rep, our_vab)"
    )

    for backbone in backbones:
        checkpoint = args.ckpt if len(backbones) == 1 else None
        checkpoint = checkpoint or app.config.paths.generator_checkpoint(backbone)
        if not Path(checkpoint).is_file():
            print(f"{backbone.value:12s} SKIP (no checkpoint yet)")
            continue
        generator = app.load_text_to_motion_generator(
            backbone,
            checkpoint,
            tokenizer,
            tokenizer_checkpoint=args.tokenizer_ckpt,
            text_encoder_checkpoint=args.text_encoder_ckpt,
        )
        report = evaluator.run_generation_evaluation_protocol(generator, request)
        log_metrics(
            run_dir, {"backbone": backbone.value, "ckpt": str(checkpoint), **report.as_dict()}
        )
        top1, top2, top3 = report.r_precision
        print(
            f"{backbone.value:12s} clips {report.clips:4d}  FID {report.fid:6.3f}  "
            f"R@1 {top1:.3f}+/-{report.r_top1_std:.3f}  R@2 {top2:.3f}  R@3 {top3:.3f}  "
            f"MM {report.matching_distance:.3f}  Div {report.diversity:.3f}  "
            f"MModality {report.multimodality:.3f}"
        )
        del generator
        if app.device == "cuda":
            torch.cuda.empty_cache()


def run_bench(app: ApplicationContext, args: argparse.Namespace) -> None:
    run_streaming_state_latency_benchmark(
        StreamingBenchmarkProtocol(
            horizons=tuple(args.horizons),
            warmup=args.warmup,
            streams=tuple(args.streams),
            streams_horizon=args.streams_horizon,
            out_path=Path(args.out) if args.out else app.config.paths.benchmark_report,
            label=str(args.config),
        ),
        generator=app.config.generator,
        codebook_size=app.config.generator.codebook_size,
        num_codebooks=app.config.tokenizer.num_quantizers,
        seed=app.config.seed,
        device=app.device,
    )
