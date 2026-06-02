"""Token-cached motion dataset.

Replaces the per-batch ``frozen_tokenizer.encode(motion)`` call inside
``run_train_epoch`` with a pre-computed (text, token_ids) cache. With the RVQ
tokenizer frozen for SSM training, the encode step is deterministic and pure
waste to repeat every epoch.

Build a cache once with ``scripts/training/precompute_rvq_tokens.py`` and load
it with ``TokenDataset`` from this module. The cache key embeds the tokenizer
checkpoint hash + RVQ hyperparams so a stale cache after retraining the
tokenizer is detected loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """Reads a joblib token cache emitted by precompute_rvq_tokens.py.

    Cache layout (list of dicts):
        {
            "text": str,
            "tokens": np.ndarray[int64] of shape (T_lat, K),
            "length": int,             # original motion length in frames
            "lat_length": int,         # tokens length T_lat = ceil(length / down_t)
        }

    Yields the same shape ``run_train_epoch`` expects for the token-cache path:
        {
            "texts": str,
            "token_ids": LongTensor(T_lat, K),
            "lat_length": int,
            "length": int,
        }
    """

    def __init__(self, cache_path: str | Path, vocab: dict[str, int] | None = None) -> None:
        cache_path = Path(cache_path)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Token cache not found at {cache_path!r}. "
                "Build it with scripts/training/precompute_rvq_tokens.py."
            )

        log.info("[TokenDataset] loading cache from %s", cache_path)
        payload = joblib.load(cache_path)
        self.meta = payload["meta"]
        self.samples: list[dict] = payload["samples"]
        self.vocab = vocab or payload.get("vocab", {})
        self.motion_stats = payload.get("motion_stats")
        log.info(
            "[TokenDataset] %d cached samples (rvq_ckpt_hash=%s, down_t=%d, K=%d)",
            len(self.samples),
            self.meta.get("rvq_ckpt_hash", "?"),
            self.meta.get("down_t", -1),
            self.meta.get("n_codebooks", -1),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        return {
            "texts": s["text"],
            "token_ids": torch.as_tensor(s["tokens"], dtype=torch.long),
            "lat_length": int(s["lat_length"]),
            "length": int(s["length"]),
        }
