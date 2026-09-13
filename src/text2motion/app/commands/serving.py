from __future__ import annotations

import argparse
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.schema import load_dataclass
from text2motion.generation.contracts import Backbone
from text2motion.streaming.decoder import StreamingMotionDecoder
from text2motion.streaming.service import LoadedInferencePipeline, MotionInferenceServer


def load_inference_pipeline(app: ApplicationContext, args: argparse.Namespace) -> LoadedInferencePipeline:
    args.ckpt = args.ckpt or app.config.paths.generator_checkpoint(args.backbone)
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    generator = app.load_text_to_motion_generator(
        args.backbone,
        args.ckpt,
        tokenizer,
        tokenizer_checkpoint=args.tokenizer_ckpt,
    )
    scaler = app.create_motion_repository().load_scaler()
    decoder = StreamingMotionDecoder(
        tokenizer.tokenizer_model,
        downsample=app.config.tokenizer.downsample,
        mean=torch.from_numpy(scaler.mean).to(app.device),
        std=torch.from_numpy(scaler.std).to(app.device),
    )
    position_limit = getattr(generator.token_generator.backbone, "max_positions", None)
    max_steps = 0 if position_limit is None else position_limit - app.config.generator.text_prefix_len
    return LoadedInferencePipeline(
        generator=generator,
        decoder=decoder,
        backbone=str(args.backbone),
        checkpoint=str(args.ckpt),
        downsample=app.config.tokenizer.downsample,
        trained_steps=app.config.data.max_motion_len // app.config.tokenizer.downsample,
        max_steps=max_steps,
        device=app.device,
    )


def run_serve(app: ApplicationContext, args: argparse.Namespace) -> None:
    def load_pipeline(request: dict) -> LoadedInferencePipeline:
        loaded_app = app.with_config(request["config"])
        return load_inference_pipeline(
            loaded_app,
            argparse.Namespace(
                config=request["config"],
                ckpt=request["ckpt"],
                tokenizer_ckpt=request["tokenizer_ckpt"],
                backbone=Backbone(request["backbone"]),
            )
        )

    MotionInferenceServer(
        load_inference_pipeline(app, args),
        pipeline_loader=load_pipeline,
        inference_log=app.config.paths.inference_log,
    ).serve(
        args.host or app.config.service.bind_host,
        args.port or app.config.service.port,
        args.idle_seconds or app.config.service.idle_seconds,
    )


def _select_model(args: argparse.Namespace, registry: Path) -> None:
    from text2motion.studio.scene import load_model_registry

    entries, _ = load_model_registry(None, registry)
    chosen = next((entry for entry in entries if entry.key == args.model), None)
    chosen = chosen or next((entry for entry in entries if args.model in entry.key), None)
    if chosen is None:
        available = ", ".join(entry.key for entry in entries)
        raise SystemExit(f"unknown model {args.model!r}. Available in {registry}: {available}")
    args.config, args.ckpt = chosen.config, chosen.ckpt
    args.tokenizer_ckpt, args.backbone = chosen.tokenizer_ckpt, Backbone(chosen.backbone)
    print(f"[load] model {chosen.key!r}: {chosen.label}")


def run_studio(app: ApplicationContext, args: argparse.Namespace) -> None:
    from text2motion.streaming.protocol import connect_or_start_motion_server
    from text2motion.studio.config import StudioConfig
    from text2motion.studio.viewer import StreamingStudioViewer

    args.ckpt = args.ckpt or app.config.paths.generator_checkpoint(args.backbone)
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    studio_config = (
        load_dataclass(StudioConfig, args.studio_config)
        if Path(args.studio_config).is_file()
        else StudioConfig()
    )
    if args.model:
        _select_model(args, studio_config.model_registry)
        app = app.with_config(args.config)
    args.host = args.host or app.config.service.connect_host
    args.port = args.port or app.config.service.port

    device = studio_config.fit.device
    if device == "cpu":
        torch.set_num_threads(studio_config.fit.threads)
    print(f"[load] SMPL-X body fit runs on {device} ({torch.get_num_threads()} threads)")
    print(f"[load] connecting to the motion service (backbone={args.backbone})...")
    client = connect_or_start_motion_server(
        args.config,
        args.ckpt,
        args.tokenizer_ckpt,
        str(args.backbone),
        args.host,
        args.port,
        app.config.paths.service_log,
        app.config.service.startup_timeout_seconds,
    )
    model_dir = args.model_dir or str(studio_config.fit.model_dir)
    model_dir = model_dir if model_dir and Path(model_dir).exists() else ""
    if not model_dir:
        print("[load] SMPL-X model dir not found -> skeleton-only studio")

    StreamingStudioViewer(
        client=client,
        model_dir=model_dir,
        device=device,
        args=args,
        config=studio_config,
        reload_config=lambda: load_dataclass(StudioConfig, args.studio_config),
    ).run()


def run_health(app: ApplicationContext, args: argparse.Namespace) -> None:
    from text2motion.streaming.protocol import MotionInferenceClient

    host = args.host or app.config.service.connect_host
    port = args.port or app.config.service.port
    client = MotionInferenceClient(host, port, app.config.service.connect_timeout_seconds)
    try:
        if client.hello.get("type") != "hello":
            raise RuntimeError(f"unexpected service response: {client.hello}")
        print(f"healthy: {host}:{port} ({client.hello.get('backbone')})")
    finally:
        client.close()
