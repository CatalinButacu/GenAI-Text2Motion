from __future__ import annotations

import logging
from pathlib import Path

import torch

from text2motion.app.config.loading import load_dataclass, service_policy
from text2motion.app.requests.base import HealthRequest
from text2motion.app.requests.serve import ModelServiceRequest, ServeRequest, StudioRequest
from text2motion.app.runtime.context import ApplicationContext
from text2motion.app.service.client import MotionInferenceClient
from text2motion.app.service.handlers import MotionInferenceServer
from text2motion.app.service.protocol import Release, ServiceAddress
from text2motion.app.service.state import LoadedInferencePipeline
from text2motion.app.stages.inference_pipeline import (
    build_decoder,
    build_generator,
    build_loaded_pipeline,
    resolve_max_steps,
    resolve_model_paths,
)
from text2motion.generation.contracts import Backbone
from text2motion.studio.studio_config import StudioConfig

logger = logging.getLogger(__name__)


def load_inference_pipeline(
    app: ApplicationContext, args: ModelServiceRequest
) -> LoadedInferencePipeline:
    ckpt, tokenizer_ckpt = resolve_model_paths(app, args)
    tokenizer = app.models.load_tokenizer(tokenizer_ckpt)
    generator = build_generator(app, args, tokenizer, ckpt, tokenizer_ckpt)
    decoder = build_decoder(app, tokenizer)
    max_steps = resolve_max_steps(app, generator)
    return build_loaded_pipeline(app, args, generator, decoder, max_steps)


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


def select_model(args: StudioRequest, model: str, registry: Path) -> None:
    from text2motion.studio.session.registry import load_model_registry

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
    from text2motion.studio.viewer import StreamingStudioViewer

    studio_config = load_studio_config(args)
    app = prepare_studio_request(app, args, studio_config)
    address = resolve_service_address(app, args)
    device = configure_body_fit(studio_config)
    client = connect_studio_client(app, args, address)
    model_dir = resolve_model_dir(args, studio_config)

    StreamingStudioViewer(
        client=client,
        model_dir=model_dir,
        device=device,
        args=args,
        config=studio_config,
        reload_config=lambda: load_dataclass(StudioConfig, args.studio_config),
    ).run()


def load_studio_config(args: StudioRequest) -> StudioConfig:
    return (
        load_dataclass(StudioConfig, args.studio_config)
        if Path(args.studio_config).is_file()
        else StudioConfig()
    )


def prepare_studio_request(
    app: ApplicationContext, args: StudioRequest, studio_config: StudioConfig
) -> ApplicationContext:
    if args.model:
        select_model(args, args.model, studio_config.model_registry)
        app = app.with_config(args.config)
    app = app.with_service_file(args.service_config)
    args.ckpt = args.ckpt or app.config.paths.generator_checkpoint(args.backbone)
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    return app


def resolve_service_address(app: ApplicationContext, args: StudioRequest) -> ServiceAddress:
    args.host = args.host or app.config.service.connect_host
    args.port = args.port or app.config.service.port
    return ServiceAddress(args.host, args.port)


def configure_body_fit(studio_config: StudioConfig) -> str:
    device = studio_config.fit.device
    if device == "cpu":
        torch.set_num_threads(studio_config.fit.threads)
    logger.info(f"[load] SMPL-X body fit runs on {device} ({torch.get_num_threads()} threads)")
    return device


def connect_studio_client(
    app: ApplicationContext, args: StudioRequest, address: ServiceAddress
) -> MotionInferenceClient:
    from text2motion.app.service.client import motion_service_client

    logger.info(f"[load] connecting to the motion service (backbone={args.backbone})...")
    return motion_service_client(
        Release(
            str(args.config),
            str(args.ckpt),
            str(args.tokenizer_ckpt),
            str(args.backbone),
        ),
        address,
        app.config.paths.service_log,
        service_config=args.service_config,
        policy=service_policy(app.config.service),
        spawn_wait_seconds=app.config.service.startup_timeout_seconds,
    )


def resolve_model_dir(args: StudioRequest, studio_config: StudioConfig) -> str:
    model_dir = str(args.model_dir or studio_config.fit.model_dir)
    model_dir = model_dir if model_dir and Path(model_dir).exists() else ""
    if not model_dir:
        logger.info("[load] SMPL-X model dir not found -> skeleton-only studio")
    return model_dir


def run_health(app: ApplicationContext, request: HealthRequest) -> None:
    from text2motion.app.service.client import MotionInferenceClient

    app = app.with_service_file(request.service_config)
    address = ServiceAddress(
        request.host or app.config.service.connect_host,
        request.port or app.config.service.port,
    )
    client = MotionInferenceClient(address, app.config.service.connect_timeout_seconds)
    try:
        if client.hello.get("type") != "hello":
            raise RuntimeError(f"unexpected service response: {client.hello}")
        logger.info(f"healthy: {address.host}:{address.port} ({client.hello.get('backbone')})")
    finally:
        client.close()
