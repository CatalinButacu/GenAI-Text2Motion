from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from text2motion.app.config.schema import ApplicationConfig
from text2motion.app.runtime.checkpoint import load_generator_checkpoint
from text2motion.app.runtime.generator_loading import (
    require_text_encoder_state,
    resolve_text_encoder_state,
    verify_generator_bundle,
)
from text2motion.generation.contracts import Backbone
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.model.text_encoder import CLIPTextEncoder, FrozenTextStates
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.tokenization.nn_tokenizer import FrozenMotionTokenizer, build_tokenizer_network

logger = logging.getLogger(__name__)


############################################################
#       ModelLoader
############################################################
@dataclass
class ModelLoader:
    config: ApplicationConfig
    device: str

    def create_tokenizer_network(self):
        return build_tokenizer_network(self.config.tokenizer, self.config.rvq_baseline)

    def load_tokenizer(self, checkpoint: str | Path | None = None) -> FrozenMotionTokenizer:
        checkpoint = checkpoint or self.config.paths.tokenizer_checkpoint
        tokenizer_network = self.create_tokenizer_network()
        tokenizer_network.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        return FrozenMotionTokenizer(
            tokenizer_network,
            device=self.device,
        )

    def load_text_encoder(self, state: dict | None = None) -> CLIPTextEncoder:
        encoder = CLIPTextEncoder(self.config.text_encoder).to(self.device)
        if state is not None:
            encoder.load_state_dict(state)
        return encoder

    def attach_text_cache(
        self, encoder: CLIPTextEncoder, captions: list[str], batch_size: int
    ) -> FrozenTextStates:
        digest = hashlib.sha256("\n".join(sorted(set(captions))).encode("utf-8")).hexdigest()[:12]
        folder = Path(self.config.paths.cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        model = encoder.cfg.model_id.replace("/", "_")
        path = folder / (
            f"clip_frozen_{model}_keep{encoder.frozen_layer_count}_b{batch_size}_{digest}.npy"
        )
        if path.exists():
            cache = FrozenTextStates.load(encoder, path)
            logger.info(f"loaded text cache {path}")
        else:
            started = time.perf_counter()
            cache = FrozenTextStates.build(encoder, captions, batch_size, path)
            logger.info(f"built text cache {path} in {time.perf_counter() - started:.1f} s")
        encoder.frozen_cache = cache
        return cache

    def resolve_generator_spec(
        self,
        backbone: Backbone | str,
        codebook_size: int,
        **overrides,
    ) -> GeneratorModelSpec:
        return GeneratorModelSpec.resolve(
            self.config.generator,
            backbone,
            codebook_size,
            self.config.tokenizer.num_quantizers,
            **overrides,
        )

    def load_text_to_motion_generator(
        self,
        backbone: Backbone | str,
        checkpoint: str | Path,
        tokenizer: FrozenMotionTokenizer,
        tokenizer_checkpoint: str | Path | None = None,
        text_encoder_checkpoint: str | Path | None = None,
        **overrides,
    ) -> TextToMotionGenerator:
        tokenizer_checkpoint = tokenizer_checkpoint or self.config.paths.tokenizer_checkpoint
        generator_spec = self.resolve_generator_spec(backbone, tokenizer.codebook_size, **overrides)
        token_generator = generator_spec.create_model(self.device).eval()

        token_generator_state, text_encoder_state, generator_bundle = load_generator_checkpoint(
            checkpoint, map_location=self.device
        )
        verify_generator_bundle(generator_bundle, tokenizer_checkpoint, generator_spec)
        token_generator.load_state_dict(token_generator_state)

        text_encoder_state = resolve_text_encoder_state(
            text_encoder_state, text_encoder_checkpoint, self.device
        )
        require_text_encoder_state(self.config, text_encoder_state)

        return TextToMotionGenerator(
            token_generator, self.load_text_encoder(text_encoder_state).eval(), tokenizer
        ).eval()
