from __future__ import annotations

import logging
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.cli_requests import (
    HealthRequest,
    ModelServiceRequest,
    ServeRequest,
    StudioRequest,
)
from text2motion.app.config import load_dataclass, service_policy
from text2motion.app.run_log import log_metrics, start_run
from text2motion.app.service import (
    LoadedInferencePipeline,
    MotionInferenceServer,
    release_identity,
)
from text2motion.evaluation.benchmark import run_streaming_state_latency_benchmark
from text2motion.evaluation.contracts import (
    BenchmarkRequest,
    EvaluateRequest,
    GenerationEvaluationProtocol,
    GenerationLengthPolicy,
    StreamingBenchmarkProtocol,
)
from text2motion.generation.contracts import Backbone, BackboneChoice, SamplingConfig
from text2motion.generation.decoder import StreamingMotionDecoder

logger = logging.getLogger(__name__)


def load_inference_pipeline(
    app: ApplicationContext, args: ModelServiceRequest
) -> LoadedInferencePipeline:
    args.ckpt = args.ckpt or app.config.paths.generator_checkpoint(args.backbone)
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    generator = app.load_text_to_motion_generator(
        args.backbone,
        args.ckpt,
        tokenizer,
        tokenizer_checkpoint=args.tokenizer_ckpt,
    )
    if app.config.service.cuda_graph_decode and Backbone(args.backbone) is Backbone.MAMBA:
        generator.token_generator.enable_cuda_graph_decode()
    scaler = app.create_motion_repository().load_scaler()
    decoder = StreamingMotionDecoder(
        tokenizer.network,
        downsample=app.config.tokenizer.downsample,
        mean=torch.from_numpy(scaler.mean).to(app.device),
        std=torch.from_numpy(scaler.std).to(app.device),
    )
    service = app.config.service
    position_limit = getattr(generator.token_generator.backbone, "max_positions", None)
    max_steps = (
        0 if position_limit is None else position_limit - app.config.generator.text_prefix_len
    )
    if service.anchor_every_tokens:
        generator.token_generator.check_stream_budget(
            0,
            app.config.generator.text_prefix_len,
            None,
            service.anchor_every_tokens,
            service.anchor_context_tokens,
        )
        max_steps = 0
    return LoadedInferencePipeline(
        generator=generator,
        decoder=decoder,
        backbone=str(args.backbone),
        checkpoint=str(args.ckpt),
        downsample=app.config.tokenizer.downsample,
        trained_steps=app.config.data.max_motion_len // app.config.tokenizer.downsample,
        max_steps=max_steps,
        device=app.device,
        identity=release_identity(
            str(args.config), str(args.ckpt), str(args.tokenizer_ckpt), str(args.backbone)
        ),
        anchor_every=service.anchor_every_tokens,
        anchor_context=service.anchor_context_tokens,
    )


def run_serve(app: ApplicationContext, args: ServeRequest) -> None:
    app = app.with_service_file(args.service_config)

    def load_pipeline(request: dict) -> LoadedInferencePipeline:
        loaded_app = app.with_config(request["config"]).with_service(app.config.service)
        return load_inference_pipeline(
            loaded_app,
            ModelServiceRequest(
                config=request["config"],
                ckpt=request["ckpt"],
                tokenizer_ckpt=request["tokenizer_ckpt"],
                backbone=Backbone(request["backbone"]),
            ),
        )

    MotionInferenceServer(
        load_inference_pipeline(app, args),
        pipeline_loader=load_pipeline,
        inference_log=app.config.paths.inference_log,
        policy=service_policy(app.config.service),
    ).serve(
        args.host or app.config.service.bind_host,
        args.port or app.config.service.port,
        args.idle_seconds or app.config.service.idle_seconds,
    )


def __select_model(args: StudioRequest, model: str, registry: Path) -> None:
    from text2motion.studio.session import load_model_registry

    entries, _ = load_model_registry(None, registry)
    chosen = next((entry for entry in entries if entry.key == model), None)
    chosen = chosen or next((entry for entry in entries if model in entry.key), None)
    if chosen is None:
        available = ", ".join(entry.key for entry in entries)
        raise SystemExit(f"unknown model {model!r}. Available in {registry}: {available}")
    args.config, args.ckpt = Path(chosen.config), Path(chosen.ckpt)
    args.tokenizer_ckpt, args.backbone = Path(chosen.tokenizer_ckpt), Backbone(chosen.backbone)
    logger.info(f"[load] model {chosen.key!r}: {chosen.label}")


def run_studio(app: ApplicationContext, args: StudioRequest) -> None:
    from text2motion.app.service import motion_service_client
    from text2motion.studio.config import StudioConfig
    from text2motion.studio.viewer import StreamingStudioViewer

    studio_config = (
        load_dataclass(StudioConfig, args.studio_config)
        if Path(args.studio_config).is_file()
        else StudioConfig()
    )
    if args.model:
        __select_model(args, args.model, studio_config.model_registry)
        app = app.with_config(args.config)
    app = app.with_service_file(args.service_config)
    args.ckpt = args.ckpt or app.config.paths.generator_checkpoint(args.backbone)
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    args.host = args.host or app.config.service.connect_host
    args.port = args.port or app.config.service.port

    device = studio_config.fit.device
    if device == "cpu":
        torch.set_num_threads(studio_config.fit.threads)
    logger.info(f"[load] SMPL-X body fit runs on {device} ({torch.get_num_threads()} threads)")
    logger.info(f"[load] connecting to the motion service (backbone={args.backbone})...")
    client = motion_service_client(
        str(args.config),
        str(args.ckpt),
        str(args.tokenizer_ckpt),
        str(args.backbone),
        args.host,
        args.port,
        app.config.paths.service_log,
        app.config.service.startup_timeout_seconds,
        service_config=args.service_config,
        policy=service_policy(app.config.service),
    )
    model_dir = str(args.model_dir or studio_config.fit.model_dir)
    model_dir = model_dir if model_dir and Path(model_dir).exists() else ""
    if not model_dir:
        logger.info("[load] SMPL-X model dir not found -> skeleton-only studio")

    StreamingStudioViewer(
        client=client,
        model_dir=model_dir,
        device=device,
        args=args,
        config=studio_config,
        reload_config=lambda: load_dataclass(StudioConfig, args.studio_config),
    ).run()


def run_health(app: ApplicationContext, request: HealthRequest) -> None:
    from text2motion.app.service import MotionInferenceClient

    app = app.with_service_file(request.service_config)
    host = request.host or app.config.service.connect_host
    port = request.port or app.config.service.port
    client = MotionInferenceClient(host, port, app.config.service.connect_timeout_seconds)
    try:
        if client.hello.get("type") != "hello":
            raise RuntimeError(f"unexpected service response: {client.hello}")
        logger.info(f"healthy: {host}:{port} ({client.hello.get('backbone')})")
    finally:
        client.close()


def run_evaluate(app: ApplicationContext, args: EvaluateRequest) -> None:
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
        reps=args.reps,
        batch_size=args.eval_batch_size,
    )
    run_dir = start_run(
        f"evaluate_{args.split.value}", app.config, app.config.paths.logs_dir, args.model_dump()
    )
    logger.info(
        f"device {app.device}  split {args.split.value}  cfg_scale {args.cfg_scale}  "
        f"temp {args.temperature}  top_p {args.top_p}  length {args.length_mode.value}  "
        f"({request.reps} generation runs, our_vab)"
    )

    for backbone in backbones:
        checkpoint = args.ckpt if len(backbones) == 1 else None
        checkpoint = checkpoint or app.config.paths.generator_checkpoint(backbone)
        if not Path(checkpoint).is_file():
            logger.info(f"{backbone.value:12s} SKIP (no checkpoint yet)")
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
        del generator
        if app.device == "cuda":
            torch.cuda.empty_cache()


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
