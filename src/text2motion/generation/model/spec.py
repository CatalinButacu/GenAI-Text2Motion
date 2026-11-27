from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
)
from text2motion.generation.model.generator import MotionTokenGenerator

TWIN_PARAMETER_TOLERANCE = 0.02


############################################################
#       GeneratorModelSpec
############################################################
@dataclass(frozen=True)
class GeneratorModelSpec:
    config: GeneratorConfig

    @property
    def backbone(self) -> Backbone:
        return self.config.backbone

    @property
    def architecture_label(self) -> str:
        return f"{self.backbone}-{self.config.d_model}d-{self.config.n_layers}l"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def create_model(self, device: str | torch.device) -> MotionTokenGenerator:
        return MotionTokenGenerator(self.config).to(device)

    @classmethod
    def resolve(
        cls,
        generator: GeneratorConfig,
        backbone: Backbone | str,
        codebook_size: int,
        num_codebooks: int,
        *,
        dropout: float | None = None,
        use_kernel: bool | None = None,
        max_seq_len: int | None = None,
    ) -> GeneratorModelSpec:
        backbone = Backbone(backbone)
        layer_count = generator.mamba_n_layers if backbone is Backbone.MAMBA else generator.n_layers
        overrides: dict[str, Any] = {
            "backbone": backbone,
            "n_layers": layer_count,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
        }
        if dropout is not None:
            overrides["dropout"] = dropout
        if use_kernel is not None:
            overrides["use_kernel"] = use_kernel
        if max_seq_len is not None:
            overrides["max_seq_len"] = max_seq_len
        return cls(replace(generator, **overrides))


def twin_parameter_counts(
    generator: GeneratorConfig, codebook_size: int, num_codebooks: int
) -> dict[Backbone, int]:
    counts = {}
    for backbone in Backbone:
        spec = GeneratorModelSpec.resolve(
            generator, backbone, codebook_size, num_codebooks, use_kernel=False
        )
        with torch.device("meta"):
            model = spec.create_model("meta")
        counts[backbone] = sum(parameter.numel() for parameter in model.parameters())
    return counts


def check_twin_parameter_match(
    generator: GeneratorConfig, codebook_size: int, num_codebooks: int
) -> dict[Backbone, int]:
    counts = twin_parameter_counts(generator, codebook_size, num_codebooks)
    smallest, largest = min(counts.values()), max(counts.values())
    gap = (largest - smallest) / smallest
    if gap > TWIN_PARAMETER_TOLERANCE:
        readable = {backbone.value: f"{count:,}" for backbone, count in counts.items()}
        raise ValueError(
            f"twin parameter counts differ by {gap:.2%} (limit {TWIN_PARAMETER_TOLERANCE:.0%}): "
            f"{readable}; change n_layers or mamba_n_layers so the twins match"
        )
    return counts
