from __future__ import annotations

import logging
import re

from src.data.humanml3d_loader import HumanML3DLoader
from src.shared.vocabulary import get_action_by_keyword

log = logging.getLogger(__name__)


def build_act_to_desc(corpus: list[str]) -> dict[str, str]:
    """Map action names to a representative HumanML3D description using keyword hits."""
    mapping: dict[str, str] = {}

    for desc in corpus:
        for tok in re.findall(r"[a-z]+", desc.lower()):
            act = get_action_by_keyword(tok)

            if act and act.name not in mapping:
                mapping[act.name] = desc

    return mapping


def enrich_unified(samples: list[dict]) -> None:
    loader = HumanML3DLoader()

    if not loader.is_available():
        return

    corpus = loader.build_text_corpus()

    if not corpus:
        return

    act_to_desc = build_act_to_desc(corpus)
    n_enriched = 0

    for s in samples:
        if s.get("source") != "amass":
            continue

        if not s.get("text", "").startswith("motion "):
            continue

        raw = s["text"].replace("motion ", "").lower()

        for tok in re.findall(r"[a-z]+", raw):
            act = get_action_by_keyword(tok)

            if act and act.name in act_to_desc:
                s["text"] = act_to_desc[act.name]
                n_enriched += 1
                break

    log.info("[UnifiedDataset] HumanML3D enriched %d AMASS samples", n_enriched)
