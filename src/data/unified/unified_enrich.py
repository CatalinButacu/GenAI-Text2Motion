from __future__ import annotations

import logging
import re

from src.data.humanml3d_loader import HumanML3DLoader
from src.shared.vocabulary import getActionByKeyword

log = logging.getLogger(__name__)


def buildActToDesc(corpus: list[str]) -> dict[str, str]:
    """Map action names to a representative HumanML3D description using keyword hits."""
    mapping: dict[str, str] = {}

    for desc in corpus:
        for tok in re.findall(r"[a-z]+", desc.lower()):
            act = getActionByKeyword(tok)

            if act and act.name not in mapping:
                mapping[act.name] = desc

    return mapping


def enrichUnified(samples: list[dict]) -> None:
    loader = HumanML3DLoader()

    if not loader.isAvailable():
        return

    corpus = loader.buildTextCorpus()

    if not corpus:
        return

    actToDesc = buildActToDesc(corpus)
    nEnriched = 0

    for s in samples:
        if s.get("source") != "amass":
            continue

        if not s.get("text", "").startswith("motion "):
            continue

        raw = s["text"].replace("motion ", "").lower()

        for tok in re.findall(r"[a-z]+", raw):
            act = getActionByKeyword(tok)

            if act and act.name in actToDesc:
                s["text"] = actToDesc[act.name]
                nEnriched += 1
                break

    log.info("[UnifiedDataset] HumanML3D enriched %d AMASS samples", nEnriched)
