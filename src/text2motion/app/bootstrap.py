from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from text2motion.app.checkpoint import load_generator_checkpoint
from text2motion.app.config import (
    ApplicationConfig,
    ComputeDevice,
    Float32Precision,
    MotionServiceConfig,
    load_config,
)
from text2motion.app.run_log import ensure_logging_configured
from text2motion.evaluation.evaluator import HumanMl3dGenerationEvaluator
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.generation.contracts import Backbone
from text2motion.generation.nn_model import GeneratorModelSpec
from text2motion.generation.nn_text import CLIPTextEncoder
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.contracts import PreparationRequest
from text2motion.motion.datasets import MotionDataModule
from text2motion.motion.storage import FAILURE_DUMP_FILE
from text2motion.tokenization.nn_model import FrozenMotionTokenizer, build_tokenizer_network


def required_path(value: Path | None, key: str) -> Path:
    if value is None:
        raise ValueError(f"paths.{key} must be set in the config")
    return value


############################################################
#       ApplicationContext
############################################################
@dataclass
class ApplicationContext:
    config: ApplicationConfig
    device: str
    config_path: Path

    def create_motion_repository(self) -> MotionDataModule:
        root = self.config.paths.hml3d_out_dir
        if root is None:
            raise ValueError("paths.hml3d_out_dir must be set (regenerated 263 output root)")
        return MotionDataModule(
            root=Path(root),
            data=self.config.data,
            texts_dir=self.config.paths.texts_dir,
            seed=self.config.seed,
        )

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
            downsample=self.config.tokenizer.downsample,
            device=self.device,
        )

    def load_text_encoder(self, state: dict | None = None) -> CLIPTextEncoder:
        encoder = CLIPTextEncoder(self.config.text_encoder).to(self.device)
        if state is not None:
            encoder.load_state_dict(state)
        return encoder

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
        if generator_bundle is not None:
            generator_bundle.verify_tokenizer(tokenizer_checkpoint)
        token_generator.load_state_dict(token_generator_state)

        if text_encoder_state is None and text_encoder_checkpoint:
            text_encoder_checkpoint_payload = torch.load(
                text_encoder_checkpoint, map_location=self.device, weights_only=True
            )
            if "text_encoder" not in text_encoder_checkpoint_payload:
                raise KeyError(
                    f"{text_encoder_checkpoint} has no 'text_encoder' entry (keys: {list(text_encoder_checkpoint_payload)})"
                )
            text_encoder_state = text_encoder_checkpoint_payload["text_encoder"]

        trains_clip = (
            self.config.text_encoder.unfreeze_last_n > 0
            or self.config.text_encoder.unfreeze_projection
        )
        if text_encoder_state is None and trains_clip:
            raise RuntimeError(
                "this config trains CLIP, but the generator checkpoint contains no matching "
                "text-encoder state. Use an atomic generator bundle, or pass an explicit "
                "text-encoder checkpoint for a documented legacy evaluation."
            )

        return TextToMotionGenerator(
            token_generator, self.load_text_encoder(text_encoder_state).eval(), tokenizer
        ).eval()

    def load_guo_evaluation_resources(
        self, vocab_dir: str | Path | None = None
    ) -> GuoEvaluationResources:
        path_config = self.config.paths
        if path_config.eval_matcher is None or path_config.eval_stats_dir is None:
            raise ValueError(
                "paths.eval_matcher and paths.eval_stats_dir must be set to score anything: FID "
                "and R-precision are defined by the frozen Guo et al. evaluator (ADR 0001)."
            )
        return GuoEvaluationResources.from_frozen_guo_assets(
            motion_dataset_dir=required_path(path_config.hml3d_out_dir, "hml3d_out_dir"),
            eval_stats_dir=path_config.eval_stats_dir,
            eval_matcher=path_config.eval_matcher,
            vocab_dir=Path(vocab_dir) if vocab_dir else path_config.our_vab_dir,
            tokenizer_downsample_factor=self.config.tokenizer.downsample,
            annotation_dir=path_config.texts_dir,
            device=self.device,
        )

    def create_humanml3d_generation_evaluator(
        self, vocab_dir: str | Path | None = None
    ) -> HumanMl3dGenerationEvaluator:
        return HumanMl3dGenerationEvaluator(self.load_guo_evaluation_resources(vocab_dir))

    def create_hml3d_preparation_request(self) -> PreparationRequest:
        path_config = self.config.paths
        return PreparationRequest(
            out_dir=required_path(path_config.hml3d_out_dir, "hml3d_out_dir"),
            amass_dir=path_config.amass_dir,
            smplx_models=path_config.smplx_models,
            index_csv=path_config.hml3d_index_csv,
            device=self.device,
        )

    def checkpoint_directory(self, stage: str | None = None) -> Path:
        directory = Path(self.config.paths.checkpoints_dir)
        if stage:
            directory = directory / stage
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def failure_dump_path(self, run_dir: Path) -> Path | None:
        if not self.config.failure_logs:
            return None
        return Path(run_dir) / FAILURE_DUMP_FILE

    def with_config(self, config_path: str | Path) -> ApplicationContext:
        return ApplicationBootstrap.from_config(config_path, device=self.device)

    def with_service(self, service: MotionServiceConfig) -> ApplicationContext:
        return ApplicationContext(
            config=replace(self.config, service=service),
            device=self.device,
            config_path=self.config_path,
        )

    def with_service_file(self, service_config: str | Path) -> ApplicationContext:
        from text2motion.app.config import load_service_config

        path = Path(service_config)
        if not path.is_file():
            raise FileNotFoundError(f"service config {path} not found (pass --service_config)")
        return self.with_service(load_service_config(path))


logger = logging.getLogger(__name__)


############################################################
#       ApplicationBootstrap
############################################################
class ApplicationBootstrap:
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        device: str | None = None,
        seed: int | None = None,
    ) -> ApplicationContext:
        ensure_logging_configured()
        config = load_config(config_path)
        resolved_seed = config.seed if seed is None else seed
        seed_everything(resolved_seed, config.deterministic)
        apply_precision(config.precision)
        torch.autograd.set_detect_anomaly(config.detect_anomaly)
        resolved_device = resolve_device(config.device, device)
        logger.info(
            f"config {config_path}  device {resolved_device}  seed {resolved_seed}  "
            f"deterministic {config.deterministic}  precision {config.precision.value}  "
            f"failure_logs {config.failure_logs}  detect_anomaly {config.detect_anomaly}"
        )
        return ApplicationContext(
            config=config,
            device=resolved_device,
            config_path=Path(config_path),
        )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def apply_precision(precision: Float32Precision | str) -> None:
    precision = Float32Precision(precision)
    allow_tf32 = precision is Float32Precision.TF32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def resolve_device(requested: ComputeDevice | str, override: str | None = None) -> str:
    if override:
        return override
    requested = ComputeDevice(requested)
    if requested is ComputeDevice.CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "the config asks for device cuda, but CUDA is not available; "
            "set device: cpu in the config or pass --device cpu"
        )
    return requested.value
