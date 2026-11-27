from __future__ import annotations

from pathlib import Path

import torch

from text2motion.app.requests.serve import ModelServiceRequest
from text2motion.app.runtime.context import ApplicationContext
from text2motion.app.service.pipeline import LoadedInferencePipeline
from text2motion.app.service.protocol import Release
from text2motion.generation.contracts import Backbone
from text2motion.generation.decoder import StreamingMotionDecoder
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.tokenization.network.frozen import FrozenMotionTokenizer


def resolve_model_paths(app: ApplicationContext, args: ModelServiceRequest) -> tuple[Path, Path]:
    args.ckpt = args.ckpt or app.config.paths.generator_checkpoint(args.backbone)
    args.tokenizer_ckpt = args.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint
    return args.ckpt, args.tokenizer_ckpt


def build_generator(
    app: ApplicationContext,
    args: ModelServiceRequest,
    tokenizer: FrozenMotionTokenizer,
    ckpt: Path,
    tokenizer_ckpt: Path,
) -> TextToMotionGenerator:
    generator = app.models.load_text_to_motion_generator(
        args.backbone,
        ckpt,
        tokenizer,
        tokenizer_checkpoint=tokenizer_ckpt,
    )
    if app.config.service.cuda_graph_decode and Backbone(args.backbone) is Backbone.MAMBA:
        generator.token_generator.cuda_graph.enable()
    return generator


def build_decoder(
    app: ApplicationContext, tokenizer: FrozenMotionTokenizer
) -> StreamingMotionDecoder:
    scaler = app.create_motion_repository().load_scaler()
    return StreamingMotionDecoder(
        tokenizer.network,
        downsample=app.config.tokenizer.downsample,
        mean=torch.from_numpy(scaler.mean).to(app.device),
        std=torch.from_numpy(scaler.std).to(app.device),
    )


def resolve_max_steps(app: ApplicationContext, generator: TextToMotionGenerator) -> int:
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
    return max_steps


def build_loaded_pipeline(
    app: ApplicationContext,
    args: ModelServiceRequest,
    generator: TextToMotionGenerator,
    decoder: StreamingMotionDecoder,
    max_steps: int,
) -> LoadedInferencePipeline:
    service = app.config.service
    return LoadedInferencePipeline(
        generator=generator,
        decoder=decoder,
        backbone=str(args.backbone),
        checkpoint=str(args.ckpt),
        downsample=app.config.tokenizer.downsample,
        trained_steps=app.config.data.max_motion_len // app.config.tokenizer.downsample,
        max_steps=max_steps,
        device=app.device,
        identity=Release(
            str(args.config), str(args.ckpt), str(args.tokenizer_ckpt), str(args.backbone)
        ).identity(),
        anchor_every=service.anchor_every_tokens,
        anchor_context=service.anchor_context_tokens,
    )
