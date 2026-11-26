from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from text2motion.generation.contracts import Backbone
from text2motion.motion.storage import atomic_torch_save, sha256_file

_BUNDLE_FORMAT = "text2motion.generator_bundle"


_BUNDLE_VERSION = 1


GENERATOR_EXECUTION_FIELDS = frozenset({"use_kernel"})


GENERATOR_FIELDS_ADDED_LATER = MappingProxyType({"per_step_text": False})


############################################################
#       GeneratorBundle
############################################################
@dataclass(frozen=True)
class GeneratorBundle:
    backbone: Backbone
    generator_state: dict[str, torch.Tensor]
    text_encoder_state: dict[str, torch.Tensor]
    tokenizer_path: str
    tokenizer_sha256: str
    resolved_generator_config: dict[str, Any]
    epoch: int
    validation_fid: float

    @classmethod
    def capture(
        cls,
        generator: nn.Module,
        text_encoder: nn.Module,
        *,
        backbone: Backbone | str,
        tokenizer_ckpt: str | Path,
        resolved_generator_config: dict[str, Any],
        epoch: int,
        validation_fid: float,
    ) -> GeneratorBundle:
        tokenizer_path = Path(tokenizer_ckpt)
        return cls(
            backbone=Backbone(backbone),
            generator_state=generator.state_dict(),
            text_encoder_state=text_encoder.state_dict(),
            tokenizer_path=str(tokenizer_path),
            tokenizer_sha256=sha256_file(tokenizer_path),
            resolved_generator_config=resolved_generator_config,
            epoch=int(epoch),
            validation_fid=float(validation_fid),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "format": _BUNDLE_FORMAT,
            "version": _BUNDLE_VERSION,
            "backbone": self.backbone.value,
            "generator": self.generator_state,
            "text_encoder": self.text_encoder_state,
            "tokenizer": {
                "path": self.tokenizer_path,
                "sha256": self.tokenizer_sha256,
            },
            "resolved_generator_config": self.resolved_generator_config,
            "epoch": self.epoch,
            "validation_fid": self.validation_fid,
        }

    def save(self, path: str | Path) -> None:
        atomic_torch_save(self.payload(), path)

    def verify_tokenizer(self, tokenizer_ckpt: str | Path) -> None:
        actual = sha256_file(tokenizer_ckpt)
        if actual != self.tokenizer_sha256:
            raise ValueError(
                "tokenizer checkpoint does not match the generator bundle: "
                f"expected {self.tokenizer_sha256}, got {actual}"
            )

    def verify_generator_config(self, resolved_config: dict[str, Any]) -> None:
        stored = {
            key: value
            for key, value in {
                **GENERATOR_FIELDS_ADDED_LATER,
                **self.resolved_generator_config,
            }.items()
            if key not in GENERATOR_EXECUTION_FIELDS
        }
        expected = {
            key: value
            for key, value in resolved_config.items()
            if key not in GENERATOR_EXECUTION_FIELDS
        }
        differences = sorted(
            f"{key}: bundle {stored.get(key)!r}, config {expected.get(key)!r}"
            for key in stored.keys() | expected.keys()
            if stored.get(key) != expected.get(key)
        )
        if differences:
            raise ValueError(
                "the generator bundle was trained with another generator config; "
                + "; ".join(differences)
            )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GeneratorBundle:
        if payload.get("version") != _BUNDLE_VERSION:
            raise ValueError(
                f"unsupported generator bundle version {payload.get('version')}; "
                f"expected {_BUNDLE_VERSION}"
            )
        tokenizer = payload["tokenizer"]
        return cls(
            backbone=Backbone(payload["backbone"]),
            generator_state=payload["generator"],
            text_encoder_state=payload["text_encoder"],
            tokenizer_path=tokenizer["path"],
            tokenizer_sha256=tokenizer["sha256"],
            resolved_generator_config=payload["resolved_generator_config"],
            epoch=int(payload["epoch"]),
            validation_fid=float(payload["validation_fid"]),
        )


def load_generator_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, GeneratorBundle | None]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and state.get("format") == _BUNDLE_FORMAT:
        bundle = GeneratorBundle.from_payload(state)
        return bundle.generator_state, bundle.text_encoder_state, bundle
    if isinstance(state, dict) and "generator" in state:
        return state["generator"], state.get("text_encoder"), None
    if (
        isinstance(state, dict)
        and state
        and all(torch.is_tensor(value) for value in state.values())
    ):
        return state, None, None
    raise ValueError(f"{path} is not a recognized generator checkpoint")
