from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from text2motion.generation.contracts import DEFAULT_OVERFIT_CRITERIA, Backbone, OverfitCriteria
from text2motion.motion.storage import sha256_file

_GATE_FORMAT = "text2motion.real_batch_overfit_gate"


_GATE_VERSION = 1


############################################################
#       OverfitGate
############################################################
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

    def validate_metrics(self, criteria: OverfitCriteria = DEFAULT_OVERFIT_CRITERIA) -> None:
        measured = {
            "first_total": self.first_total,
            "final_total": self.final_total,
            "token_accuracy": self.token_accuracy,
            "ce": self.ce,
        }
        not_finite = sorted(name for name, value in measured.items() if not math.isfinite(value))
        if not_finite:
            raise RuntimeError(f"overfit gate metrics are not finite: {not_finite}")
        if self.batch_size < 1:
            raise RuntimeError(f"overfit gate batch_size must be at least 1, got {self.batch_size}")
        if self.steps < 1:
            raise RuntimeError(f"overfit gate steps must be at least 1, got {self.steps}")
        if not 0.0 <= self.token_accuracy <= 1.0:
            raise RuntimeError(
                f"overfit gate token accuracy {self.token_accuracy} is not in [0, 1]"
            )
        if self.ce < 0:
            raise RuntimeError(f"overfit gate CE {self.ce} is negative")
        if self.first_total <= 0:
            raise RuntimeError("overfit gate first_total must be positive")
        failures = criteria.failure_messages(self.token_accuracy, self.ce, self.total_ratio)
        if failures:
            raise RuntimeError(failures[0])

    def verify_setup(
        self,
        *,
        config_path: str | Path,
        tokenizer_ckpt: str | Path,
        backbone: Backbone | str,
        resolved_config: dict[str, Any],
        criteria: OverfitCriteria = DEFAULT_OVERFIT_CRITERIA,
    ) -> None:
        expected = {
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(tokenizer_ckpt),
            "backbone": Backbone(backbone),
            "resolved_generator_config": {
                k: v for k, v in resolved_config.items() if k != "dropout"
            },
        }
        actual = {
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "backbone": self.backbone,
            "resolved_generator_config": {
                k: v for k, v in self.resolved_generator_config.items() if k != "dropout"
            },
        }
        for key, value in expected.items():
            if actual[key] != value:
                raise RuntimeError(f"overfit gate mismatch for {key}: expected {value!r}")
        self.validate_metrics(criteria)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _GATE_FORMAT,
            "version": _GATE_VERSION,
            **asdict(self),
            "backbone": self.backbone.value,
            "total_ratio": self.total_ratio,
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> OverfitGate:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") not in (None, _GATE_FORMAT):
            raise ValueError(f"unrecognized overfit gate format {payload.get('format')!r}")
        if payload.get("version", _GATE_VERSION) != _GATE_VERSION:
            raise ValueError(f"unsupported overfit gate version {payload.get('version')}")
        fields = {name: payload[name] for name in cls.__dataclass_fields__}
        fields["backbone"] = Backbone(fields["backbone"])
        return cls(**fields)
