from __future__ import annotations

from typing import assert_never

from text2motion.tokenization.contracts import (
    ResidualVectorQuantizationConfig,
    TokenizerConfig,
    TokenizerKind,
)
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.network.finite_scalar import FiniteScalarQuantizationTokenizer
from text2motion.tokenization.network.residual import ResidualVectorQuantizationTokenizer


def build_tokenizer_network(
    finite_scalar_config: TokenizerConfig,
    residual_vector_config: ResidualVectorQuantizationConfig | None = None,
) -> TokenizerNetwork:
    match TokenizerKind(finite_scalar_config.kind):
        case TokenizerKind.FINITE_SCALAR:
            return FiniteScalarQuantizationTokenizer(finite_scalar_config)
        case TokenizerKind.RESIDUAL_VECTOR:
            if residual_vector_config is None:
                raise ValueError(
                    "an RVQ tokenizer needs its ResidualVectorQuantizationConfig; pass rvq_baseline"
                )
            return ResidualVectorQuantizationTokenizer(residual_vector_config)
        case unknown:
            assert_never(unknown)
