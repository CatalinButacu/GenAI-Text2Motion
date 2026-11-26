from __future__ import annotations

from pathlib import Path

import torch

from text2motion.app.config.schema import ApplicationConfig
from text2motion.app.runtime.checkpoint import GeneratorBundle
from text2motion.generation.model.spec import GeneratorModelSpec


def verify_generator_bundle(
    bundle: GeneratorBundle | None,
    tokenizer_checkpoint: str | Path,
    generator_spec: GeneratorModelSpec,
) -> None:
    if bundle is not None:
        bundle.verify_tokenizer(tokenizer_checkpoint)
        bundle.verify_generator_config(generator_spec.as_dict())


def load_text_encoder_payload(checkpoint: str | Path, device: str) -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if "text_encoder" not in payload:
        raise KeyError(f"{checkpoint} has no 'text_encoder' entry (keys: {list(payload)})")
    return payload["text_encoder"]


def resolve_text_encoder_state(
    text_encoder_state: dict | None,
    text_encoder_checkpoint: str | Path | None,
    device: str,
) -> dict | None:
    if text_encoder_state is None and text_encoder_checkpoint:
        return load_text_encoder_payload(text_encoder_checkpoint, device)
    return text_encoder_state


def require_text_encoder_state(config: ApplicationConfig, text_encoder_state: dict | None) -> None:
    trains_clip = config.text_encoder.unfreeze_last_n > 0 or config.text_encoder.unfreeze_projection
    if text_encoder_state is None and trains_clip:
        raise RuntimeError(
            "this config trains CLIP, but the generator checkpoint contains no matching "
            "text-encoder state. Use an atomic generator bundle, or pass an explicit "
            "text-encoder checkpoint for a documented legacy evaluation."
        )
