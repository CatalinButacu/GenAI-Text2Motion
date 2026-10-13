# .\.venv\Scripts\python.exe scripts/qa/memory_probe.py
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.run_log import log_metrics, start_run
from text2motion.generation.contracts import Backbone, StudioSamplingConfig
from text2motion.generation.nn_backbones import MambaMixer
from text2motion.generation.nn_model import MotionTokenGenerator, seeded_sampling_uniforms

logger = logging.getLogger(__name__)
SPLIT_SEED = 2026
HALF_LIFE_MARKS = (8, 16, 48)


def test_captions(resources, count: int) -> list[str]:
    clip_ids = sorted(resources.clip_ids("test"))
    np.random.default_rng(SPLIT_SEED).shuffle(clip_ids)
    captions: list[str] = []
    for clip_id in clip_ids:
        text_path = resources.annotation_dir / f"{clip_id}.txt"
        if not text_path.exists():
            continue
        captions.append(text_path.read_text(encoding="utf-8").splitlines()[0].split("#")[0].strip())
        if len(captions) == count:
            return captions
    raise ValueError(f"only {len(captions)} captioned test clips found, {count} requested")


def last_token_log_probabilities(
    generator: MotionTokenGenerator, tokens: torch.Tensor, text: torch.Tensor
) -> torch.Tensor:
    return F.log_softmax(generator(tokens, text)[:, -1].float(), dim=-1)


def sensitivity_by_lag(
    generator: MotionTokenGenerator, tokens: torch.Tensor, text: torch.Tensor
) -> list[float]:
    reference = last_token_log_probabilities(generator, tokens, text)
    random_tokens = torch.randint_like(tokens, generator.cfg.codebook_size)
    divergences = []
    for lag in range(1, tokens.size(1)):
        changed = tokens.clone()
        position = tokens.size(1) - 1 - lag
        changed[:, position] = random_tokens[:, position]
        perturbed = last_token_log_probabilities(generator, changed, text)
        divergence = F.kl_div(perturbed, reference, reduction="none", log_target=True)
        divergences.append(float(divergence.sum(dim=(-1, -2)).mean()))
    return divergences


def state_half_lives(
    generator: MotionTokenGenerator, tokens: torch.Tensor, text: torch.Tensor
) -> list[dict[str, float]]:
    mixers = [module for module in generator.modules() if isinstance(module, MambaMixer)]
    inputs: list[torch.Tensor] = []
    hooks = [
        mixer.register_forward_pre_hook(lambda module, arguments: inputs.append(arguments[0]))
        for mixer in mixers
    ]
    generator(tokens, text)
    for hook in hooks:
        hook.remove()
    rows = []
    for layer, (mixer, sequence) in enumerate(zip(mixers, inputs, strict=True), start=1):
        features, _ = mixer.in_proj(sequence).chunk(2, dim=-1)
        padded = F.pad(features.transpose(1, 2), (mixer.d_conv - 1, 0))
        activated = F.silu(mixer.conv1d(padded).transpose(1, 2))
        step_raw = mixer.x_proj(activated)[..., : mixer.dt_rank]
        mean_step = F.softplus(mixer.dt_proj(step_raw)).float().mean(dim=(0, 1))
        decay_rate = mean_step.unsqueeze(-1) * torch.exp(mixer.a_log.float())
        half_life = (math.log(2.0) / decay_rate).flatten().cpu().numpy()
        row = {
            "layer": layer,
            "median_half_life_tokens": float(np.median(half_life)),
            "p90_half_life_tokens": float(np.percentile(half_life, 90)),
            "max_half_life_tokens": float(half_life.max()),
        }
        for mark in HALF_LIFE_MARKS:
            row[f"share_above_{mark}_tokens"] = float((half_life > mark).mean())
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument(
        "--tokenizer_ckpt",
        type=Path,
        default=Path("artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"),
    )
    parser.add_argument("--checkpoint_pattern", default="generator_{backbone}_bs8_e60.pt")
    parser.add_argument("--captions", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args()

    app = ApplicationBootstrap.from_config(args.config)
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt)
    resources = app.create_humanml3d_generation_evaluator().resources
    run_dir = start_run("memory_probe", app.config, app.config.paths.logs_dir, vars(args))
    captions = test_captions(resources, args.captions)
    sampling = StudioSamplingConfig()
    report: dict[str, dict] = {}

    for backbone in (Backbone.MAMBA, Backbone.TRANSFORMER):
        checkpoint = app.checkpoint_directory("generator") / args.checkpoint_pattern.format(
            backbone=backbone.value
        )
        pipeline = app.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
        ).eval()
        generator = pipeline.token_generator
        with torch.no_grad():
            text = pipeline.text_encoder(captions)
            uniforms = seeded_sampling_uniforms(
                range(len(captions)), args.tokens, generator.cfg.num_codebooks, app.device
            )
            stream = generator.stream_token_indices(
                text,
                args.tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                cfg_scale=sampling.cfg_scale,
                uniforms=uniforms,
            )
            tokens = torch.stack(list(stream), dim=1)
            torch.manual_seed(SPLIT_SEED)
            entry: dict = {"sensitivity_by_lag": sensitivity_by_lag(generator, tokens, text)}
            other_text = last_token_log_probabilities(generator, tokens, text.roll(1, dims=0))
            own_text = last_token_log_probabilities(generator, tokens, text)
            text_change = F.kl_div(other_text, own_text, reduction="none", log_target=True)
            entry["sensitivity_to_text"] = float(text_change.sum(dim=(-1, -2)).mean())
            if backbone is Backbone.MAMBA:
                entry["state_half_lives"] = state_half_lives(generator, tokens, text)
        report[backbone.value] = entry
        log_metrics(run_dir, {"backbone": backbone.value, **entry})
        lags = entry["sensitivity_by_lag"]
        logger.info(
            f"{backbone.value}: KL at lag 1 {lags[0]:.3f}, lag 8 {lags[7]:.4f}, "
            f"lag 16 {lags[15]:.4f}, lag 32 {lags[31]:.5f}, lag {len(lags)} {lags[-1]:.5f}; "
            f"other text {entry['sensitivity_to_text']:.3f}"
        )
        del pipeline
        torch.cuda.empty_cache()

    for row in report[Backbone.MAMBA.value]["state_half_lives"]:
        logger.info(f"mamba {row}")

    out_dir = app.config.paths.outputs_dir / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_dir / 'memory_probe.json'}")


if __name__ == "__main__":
    main()
