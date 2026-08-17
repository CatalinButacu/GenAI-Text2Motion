from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from text2motion.app.checkpoint import load_generator_checkpoint
from text2motion.app.config import Config, load_config
from text2motion.app.runtime import apply_precision, resolve_device, seed_everything
from text2motion.evaluation.evaluator import MotionEvaluator
from text2motion.evaluation.matcher import EvaluationContext
from text2motion.generation.model import Backbone, GeneratorArchitecture
from text2motion.generation.pipeline import MotionGenerator
from text2motion.generation.text import CLIPTextEncoder
from text2motion.motion.dataset import MotionRepository
from text2motion.motion.preparation import PreparationRequest
from text2motion.tokenization.model import MotionTokenizer, build_tokenizer_module

DEFAULT_TOKENIZER_CKPT = "checkpoints/tokenizer/tokenizer_fsq.pt"


@dataclass
class Application:
    config: Config
    device: str
    config_path: Path

    @property
    def paths(self):
        return self.config.paths

    def motions(self) -> MotionRepository:
        root = self.config.paths.hml3d_out_dir
        if root is None:
            raise ValueError("paths.hml3d_out_dir must be set (regenerated 263 output root)")
        return MotionRepository(
            root=Path(root),
            data=self.config.data,
            texts_dir=self.config.paths.texts_dir,
            seed=self.config.seed,
        )

    def tokenizer_module(self):
        return build_tokenizer_module(self.config.tokenizer, self.config.rvq_baseline)

    def tokenizer(self, checkpoint: str | Path = DEFAULT_TOKENIZER_CKPT) -> MotionTokenizer:
        module = self.tokenizer_module()
        module.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return MotionTokenizer(
            module,
            downsample=self.config.tokenizer.downsample,
            device=self.device,
            scaler=self.motions().scaler(),
        )

    def text_encoder(self, state: dict | None = None) -> CLIPTextEncoder:
        encoder = CLIPTextEncoder(self.config.text_encoder).to(self.device)
        if state is not None:
            encoder.load_state_dict(state)
        return encoder

    def architecture(
        self,
        backbone: Backbone | str,
        codebook_size: int,
        **overrides,
    ) -> GeneratorArchitecture:
        return GeneratorArchitecture.resolve(
            self.config.generator,
            backbone,
            codebook_size,
            self.config.tokenizer.num_quantizers,
            **overrides,
        )

    def generator(
        self,
        backbone: Backbone | str,
        checkpoint: str | Path,
        tokenizer: MotionTokenizer,
        tokenizer_checkpoint: str | Path = DEFAULT_TOKENIZER_CKPT,
        text_encoder_checkpoint: str | Path | None = None,
        **overrides,
    ) -> MotionGenerator:
        architecture = self.architecture(backbone, tokenizer.codebook_size, **overrides)
        module = architecture.build(self.device).eval()

        generator_state, encoder_state, bundle = load_generator_checkpoint(
            checkpoint, map_location=self.device
        )
        if bundle is not None:
            bundle.verify_tokenizer(tokenizer_checkpoint)
        module.load_state_dict(generator_state)

        if encoder_state is None and text_encoder_checkpoint:
            state = torch.load(
                text_encoder_checkpoint, map_location=self.device, weights_only=False
            )
            if "text_encoder" not in state:
                raise KeyError(
                    f"{text_encoder_checkpoint} has no 'text_encoder' entry (keys: {list(state)})"
                )
            encoder_state = state["text_encoder"]

        trains_clip = (
            self.config.text_encoder.unfreeze_last_n > 0
            or self.config.text_encoder.unfreeze_projection
        )
        if encoder_state is None and trains_clip:
            raise RuntimeError(
                "this config trains CLIP, but the generator checkpoint contains no matching "
                "text-encoder state. Use an atomic generator bundle, or pass an explicit "
                "text-encoder checkpoint for a documented legacy evaluation."
            )

        return MotionGenerator(module, self.text_encoder(encoder_state).eval(), tokenizer).eval()

    def evaluation_context(self, vocab_dir: str | Path | None = None) -> EvaluationContext:
        paths = self.config.paths
        if paths.eval_matcher is None or paths.eval_stats_dir is None:
            raise ValueError(
                "paths.eval_matcher and paths.eval_stats_dir must be set to score anything: FID "
                "and R-precision are defined by the frozen Guo et al. evaluator (ADR 0001)."
            )
        return EvaluationContext.load(
            out_dir=Path(paths.hml3d_out_dir),
            eval_stats_dir=paths.eval_stats_dir,
            eval_matcher=paths.eval_matcher,
            vocab_dir=Path(vocab_dir) if vocab_dir else paths.our_vab_dir,
            downsample=self.config.tokenizer.downsample,
            text_dir=paths.texts_dir,
            device=self.device,
        )

    def evaluator(self, vocab_dir: str | Path | None = None) -> MotionEvaluator:
        return MotionEvaluator(self.evaluation_context(vocab_dir))

    def preparation_request(self) -> PreparationRequest:
        paths = self.config.paths
        return PreparationRequest(
            out_dir=Path(paths.hml3d_out_dir),
            amass_dir=paths.amass_dir,
            smplx_models=paths.smplx_models,
            index_csv=paths.hml3d_index_csv,
            device=self.device,
        )

    def checkpoints_dir(self) -> Path:
        directory = Path(self.config.paths.checkpoints_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


class ApplicationFactory:
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        device: str | None = None,
        seed: int | None = None,
    ) -> Application:
        config = load_config(config_path)
        seed_everything(config.seed if seed is None else seed, config.deterministic)
        apply_precision(config.precision)
        return Application(
            config=config,
            device=resolve_device(config.device, device),
            config_path=Path(config_path),
        )
