from __future__ import annotations

from types import MappingProxyType

from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
)
from text2motion.generation.model.layers import (
    SequenceBackbone,
)
from text2motion.generation.model.mamba import MambaBackbone
from text2motion.generation.model.transformer import TransformerBackbone

_BACKBONE_FACTORIES = MappingProxyType(
    {Backbone.MAMBA: MambaBackbone, Backbone.TRANSFORMER: TransformerBackbone}
)


def instantiate_backbone(cfg: GeneratorConfig) -> SequenceBackbone:
    build = _BACKBONE_FACTORIES.get(Backbone(cfg.backbone))
    if build is None:
        raise ValueError(
            f"unknown backbone {cfg.backbone!r} (expected one of {[b.value for b in Backbone]})"
        )
    return build(cfg)
