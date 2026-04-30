from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np

log = logging.getLogger(__name__)


def splitSamples(samples: list[dict], split: str, seed: int = 42) -> list[dict]:
    """80/10/10 deterministic split (train/val/test).

    Interaction clips (those with interaction_id) are split by group so that
    both agents of one interaction always land in the same split — avoids
    test-set leakage for two-person sequences.
    """
    rng = np.random.RandomState(seed)
    ixSamples = [s for s in samples if s.get("interaction_id")]
    other = [s for s in samples if not s.get("interaction_id")]

    # --- interaction groups: keep both agents in the same split ---
    groups: dict[str, list[dict]] = OrderedDict()
    for s in ixSamples:
        groups.setdefault(s["interaction_id"], []).append(s)
    gids = list(groups.keys())
    rng.shuffle(gids)
    nG = len(gids)
    gTrainEnd = int(nG * 0.8)
    gValEnd   = int(nG * 0.9)
    if split == "train":
        keepIds = set(gids[:gTrainEnd])
    elif split == "val":
        keepIds = set(gids[gTrainEnd:gValEnd])
    else:  # test
        keepIds = set(gids[gValEnd:])
    ixKept = [s for gid in keepIds for s in groups[gid]]

    # --- non-interaction samples: simple 80/10/10 ---
    rng.shuffle(other)  # type: ignore[arg-type]
    nO = len(other)
    oTrainEnd = int(nO * 0.8)
    oValEnd   = int(nO * 0.9)
    if split == "train":
        otherKept = other[:oTrainEnd]
    elif split == "val":
        otherKept = other[oTrainEnd:oValEnd]
    else:  # test
        otherKept = other[oValEnd:]

    result = ixKept + otherKept
    rng.shuffle(result)  # type: ignore[arg-type]
    log.info("[UnifiedDataset] %s split: %d samples", split, len(result))

    return result
