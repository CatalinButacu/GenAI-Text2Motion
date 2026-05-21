from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np

log = logging.getLogger(__name__)


def split_samples(samples: list[dict], split: str, seed: int = 42) -> list[dict]:
    """80/10/10 deterministic split (train/val/test).

    Interaction clips (those with interaction_id) are split by group so that
    both agents of one interaction always land in the same split — avoids
    test-set leakage for two-person sequences.
    """
    rng = np.random.RandomState(seed)
    ix_samples = [s for s in samples if s.get("interaction_id")]
    other = [s for s in samples if not s.get("interaction_id")]

    # --- interaction groups: keep both agents in the same split ---
    groups: dict[str, list[dict]] = OrderedDict()
    for s in ix_samples:
        groups.setdefault(s["interaction_id"], []).append(s)
    gids = list(groups.keys())
    rng.shuffle(gids)
    n_g = len(gids)
    g_train_end = int(n_g * 0.8)
    g_val_end   = int(n_g * 0.9)
    if split == "train":
        keep_ids = set(gids[:g_train_end])
    elif split == "val":
        keep_ids = set(gids[g_train_end:g_val_end])
    else:  # test
        keep_ids = set(gids[g_val_end:])
    ix_kept = [s for gid in keep_ids for s in groups[gid]]

    # --- non-interaction samples: simple 80/10/10 ---
    rng.shuffle(other)  # type: ignore[arg-type]
    n_o = len(other)
    o_train_end = int(n_o * 0.8)
    o_val_end   = int(n_o * 0.9)
    if split == "train":
        other_kept = other[:o_train_end]
    elif split == "val":
        other_kept = other[o_train_end:o_val_end]
    else:  # test
        other_kept = other[o_val_end:]

    result = ix_kept + other_kept
    rng.shuffle(result)  # type: ignore[arg-type]
    log.info("[UnifiedDataset] %s split: %d samples", split, len(result))

    return result
