"""Atomic motion vocabulary built from human-labeled RVQ codebook entries.

Workflow:
  1. Train RVQ tokenizer (unsupervised on motion).
  2. Run scripts/evaluation/codebook_label.py -> writes labels.csv with empty
     label column for each top-K codebook entry.
  3. Human watches prototype GIFs, fills in `label` column.
  4. Use CodebookVocab(labels_csv) at inference to:
       - look up labels for token sequences
       - decompose compound text prompts into atomic-label sequences

Compound decomposition splits the prompt on connectives ("then", "and",
"after", commas) and SBERT-matches each segment to the closest label.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

CONNECTIVES = re.compile(
    r"\s*(?:,|;| then | and then | and | before | after | followed by )\s*",
    re.IGNORECASE,
)

NOISE_LABELS = {"", "noise", "skip", "drop"}


class CodebookVocab:
    def __init__(self, labelsCsv: str | Path):
        self.labelsCsv = Path(labelsCsv)
        self.labelToEntries: dict[str, list[tuple[int, int]]] = {}
        self.entryToLabel: dict[tuple[int, int], str] = {}
        self.loadLabels()
        self.encoder = None
        self.labelEmbeds: np.ndarray | None = None
        self.labelList: list[str] = []

    def loadLabels(self) -> None:
        if not self.labelsCsv.exists():
            raise FileNotFoundError(f"labels csv not found: {self.labelsCsv}")

        with open(self.labelsCsv, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                label = (row.get("label") or "").strip().lower()

                if label in NOISE_LABELS:
                    continue
                cb = int(row["codebook_idx"])
                entry = int(row["entry_idx"])
                key = (cb, entry)
                self.entryToLabel[key] = label
                self.labelToEntries.setdefault(label, []).append(key)
        log.info("[vocab] loaded %d labeled entries  %d unique labels",
                 len(self.entryToLabel), len(self.labelToEntries))

    def labels(self) -> list[str]:
        return sorted(self.labelToEntries.keys())

    def labelFor(self, codebook: int, entry: int) -> str:
        return self.entryToLabel.get((codebook, entry), "unknown")

    def decodeIndexSequence(self, codebookIdx: int,
                            indices: list[int] | np.ndarray) -> list[str]:
        # Returns per-step label, deduplicating consecutive repeats
        out: list[str] = []
        prev: str | None = None

        for e in indices:
            label = self.entryToLabel.get((codebookIdx, int(e)), "unknown")

            if label != prev:
                out.append(label)
                prev = label

        return out

    def loadEncoder(self) -> None:
        # Lazy SBERT load — only fired the first time decomposeText is called
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.labelList = self.labels()
        # Use a short sentence form for SBERT so the embedding focuses on action verb
        prompts = [f"a person {lab.replace('_', ' ')}" for lab in self.labelList]
        self.labelEmbeds = self.encoder.encode(prompts, normalize_embeddings=True)
        log.info("[vocab] SBERT-encoded %d labels for compound decomposition",
                 len(self.labelList))

    def decomposeText(self, text: str) -> list[tuple[str, float]]:
        # Returns [(label, score), ...] one per atomic segment of the prompt
        if self.encoder is None:
            self.loadEncoder()
        assert self.encoder is not None and self.labelEmbeds is not None

        segments = [s.strip() for s in CONNECTIVES.split(text) if s.strip()]

        if not segments:
            return []
        segEmbeds = self.encoder.encode(segments, normalize_embeddings=True)
        sims = segEmbeds @ self.labelEmbeds.T  # (n_seg, n_label)
        bestIdx = sims.argmax(axis=1)
        out: list[tuple[str, float]] = []

        for i, label_idx in enumerate(bestIdx):
            out.append((self.labelList[int(label_idx)], float(sims[i, label_idx])))

        return out
