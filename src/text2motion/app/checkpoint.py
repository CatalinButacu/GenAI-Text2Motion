from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from text2motion.generation.model import Backbone

BUNDLE_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "format": "text2motion.generator_bundle",
            "version": BUNDLE_VERSION,
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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GeneratorBundle:
        if payload.get("version") != BUNDLE_VERSION:
            raise ValueError(
                f"unsupported generator bundle version {payload.get('version')}; "
                f"expected {BUNDLE_VERSION}"
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


def atomic_torch_save(state: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_generator_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, GeneratorBundle | None]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and state.get("format") == "text2motion.generator_bundle":
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


GATE_FORMAT = "text2motion.real_batch_overfit_gate"
GATE_VERSION = 1
MIN_TOKEN_ACCURACY = 0.99
MAX_CE = 0.1
MAX_TOTAL_RATIO = 0.05


@dataclass(frozen=True)
class OverfitGate:
    config_sha256: str
    tokenizer_sha256: str
    backbone: Backbone
    resolved_generator_config: dict[str, Any]
    batch_size: int
    steps: int
    first_total: float
    final_total: float
    token_accuracy: float
    ce: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone", Backbone(self.backbone))

    @property
    def total_ratio(self) -> float:
        return self.final_total / self.first_total

    def validate_metrics(self) -> None:
        if self.first_total <= 0:
            raise RuntimeError("overfit gate first_total must be positive")
        if self.token_accuracy < MIN_TOKEN_ACCURACY:
            raise RuntimeError("overfit gate token accuracy is below 99%")
        if self.ce > MAX_CE:
            raise RuntimeError("overfit gate CE is above 0.1")
        if self.total_ratio > MAX_TOTAL_RATIO:
            raise RuntimeError("overfit gate total loss did not fall by 95%")

    def verify_setup(
        self,
        *,
        config_path: str | Path,
        tokenizer_ckpt: str | Path,
        backbone: Backbone | str,
        resolved_config: dict[str, Any],
    ) -> None:
        expected = {
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(tokenizer_ckpt),
            "backbone": Backbone(backbone),
            "resolved_generator_config": resolved_config,
        }
        actual = {
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "backbone": self.backbone,
            "resolved_generator_config": self.resolved_generator_config,
        }
        for key, value in expected.items():
            if actual[key] != value:
                raise RuntimeError(f"overfit gate mismatch for {key}: expected {value!r}")
        self.validate_metrics()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": GATE_FORMAT,
            "version": GATE_VERSION,
            **asdict(self),
            "backbone": self.backbone.value,
            "total_ratio": self.total_ratio,
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> OverfitGate:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") not in (None, GATE_FORMAT):
            raise ValueError(f"unrecognized overfit gate format {payload.get('format')!r}")
        if payload.get("version", GATE_VERSION) != GATE_VERSION:
            raise ValueError(f"unsupported overfit gate version {payload.get('version')}")
        fields = {name: payload[name] for name in cls.__dataclass_fields__}
        fields["backbone"] = Backbone(fields["backbone"])
        return cls(**fields)
