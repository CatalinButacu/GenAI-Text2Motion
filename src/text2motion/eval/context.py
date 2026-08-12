from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from text2motion.data.hml3d.stats import MotionScaler
from text2motion.eval.matcher import load_eval_stats, load_matchers
from text2motion.shared.config import Config


@dataclass(frozen=True)
class SamplingCfg:
    temperature: float = 1.0
    top_p: float = 0.9
    cfg_scale: float = 1.0
    stop_at_end: bool = False

    @classmethod
    def from_args(cls, args: Any) -> SamplingCfg:
        return cls(
            temperature=args.temperature,
            top_p=getattr(args, "top_p", 0.9),
            cfg_scale=args.cfg_scale,
            stop_at_end=getattr(args, "length_mode", "fixed") == "end",
        )


def resolve_device(cfg: Config, override: str | None = None) -> str:
    if override:
        return override
    return cfg.device if torch.cuda.is_available() else "cpu"


def make_build_text(word_vectorizer: Any, max_tokens: int = 20) -> Callable:
    def build_text(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        items = ["sos/OTHER"] + tokens[:max_tokens] + ["eos/OTHER"]
        word_embeddings = np.stack([word_vectorizer[item][0] for item in items]).astype(np.float32)
        pos_onehots = np.stack([word_vectorizer[item][1] for item in items]).astype(np.float32)
        return word_embeddings, pos_onehots

    return build_text


@dataclass(frozen=True)
class GenerationPipeline:
    generator: Any
    text_encoder: Any
    tokenizer: Any

    def eval(self) -> GenerationPipeline:
        self.generator.eval()
        self.text_encoder.eval()
        self.tokenizer.eval()
        return self

    def stream_tokens(self, captions: list[str], token_len: int, sampling: SamplingCfg):
        text_emb = self.text_encoder(captions)
        steps = list(
            self.generator.stream(
                text_emb,
                token_len,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                cfg_scale=sampling.cfg_scale,
                stop_at_end=sampling.stop_at_end,
            )
        )
        if not steps:  # END on the very first step: no usable motion (counted by the caller)
            return None
        return torch.stack(steps, dim=1)

    def generate_batch(
        self, captions: list[str], token_len: int, sampling: SamplingCfg, ctx: EvalContext
    ) -> list[np.ndarray] | None:
        tokens = self.stream_tokens(captions, token_len, sampling)
        if tokens is None:
            return None
        decoded = ctx.denormalize(self.tokenizer.decode(tokens).cpu().numpy())
        return [decoded[row] for row in range(decoded.shape[0])]

    def generate(
        self, caption: str, token_len: int, sampling: SamplingCfg, ctx: EvalContext
    ) -> np.ndarray | None:
        batch = self.generate_batch([caption], token_len, sampling, ctx)
        return None if batch is None else batch[0]


@dataclass(frozen=True)
class EvalContext:
    device: str
    out_dir: Path
    text_dir: Path
    scaler: MotionScaler
    eval_mean: np.ndarray
    eval_std: np.ndarray
    motion_matcher: Any
    text_matcher: Any
    build_text: Callable
    downsample: int

    @classmethod
    def from_config(
        cls, cfg: Config, device: str | None = None, our_vab_dir: str | Path | None = None
    ) -> EvalContext:
        from text2motion.eval.word_vectorizer import WordVectorizer

        resolved_device = resolve_device(cfg, device)
        out_dir = Path(cfg.paths.hml3d_out_dir)
        text_dir = Path(cfg.paths.texts_dir) if cfg.paths.texts_dir else out_dir / "texts"
        eval_mean, eval_std = load_eval_stats(cfg.paths.eval_stats_dir)
        motion_matcher, text_matcher = load_matchers(cfg.paths.eval_matcher, device=resolved_device)
        vocab_dir = Path(our_vab_dir) if our_vab_dir else Path(cfg.paths.our_vab_dir)

        return cls(
            device=resolved_device,
            out_dir=out_dir,
            text_dir=text_dir,
            scaler=MotionScaler.load(out_dir, dim=cfg.hml3d.dim),
            eval_mean=eval_mean,
            eval_std=eval_std,
            motion_matcher=motion_matcher,
            text_matcher=text_matcher,
            build_text=make_build_text(WordVectorizer(str(vocab_dir), "our_vab")),
            downsample=cfg.tokenizer.downsample,
        )

    def clip_ids(self, split: str) -> list[str]:
        raw = (self.out_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        present = [name.strip() for name in raw if name.strip()]
        return list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in present))

    @property
    def our_mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def our_std(self) -> np.ndarray:
        return self.scaler.std

    def normalize(self, feats):
        return self.scaler.normalize(feats)

    def denormalize(self, feats):
        return self.scaler.denormalize(feats)
