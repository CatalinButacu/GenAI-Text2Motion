from __future__ import annotations

from text2motion.tokenization.network.finite_scalar import FiniteScalarQuantizationTokenizer
from text2motion.tokenization.network.residual import ResidualVectorQuantizationTokenizer

__all__ = [
    "FiniteScalarQuantizationTokenizer",
    "ResidualVectorQuantizationTokenizer",
]
