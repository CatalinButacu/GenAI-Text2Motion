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
    def __init__(self, labels_csv: str | Path):
        self.labels_csv = Path(labels_csv)
        self.label_to_entries: dict[str, list[tuple[int, int]]] = {}
        self.entry_to_label: dict[tuple[int, int], str] = {}
        self.load_labels()
        self.encoder = None
        self.label_embeds: np.ndarray | None = None
        self.label_list: list[str] = []

    def load_labels(self) -> None:
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels csv not found: {self.labels_csv}")

        with open(self.labels_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                label = (row.get("label") or "").strip().lower()

                if label in NOISE_LABELS:
                    continue
                cb = int(row["codebook_idx"])
                entry = int(row["entry_idx"])
                key = (cb, entry)
                self.entry_to_label[key] = label
                self.label_to_entries.setdefault(label, []).append(key)
        log.info("[vocab] loaded %d labeled entries  %d unique labels",
                 len(self.entry_to_label), len(self.label_to_entries))

    def labels(self) -> list[str]:
        return sorted(self.label_to_entries.keys())

    def label_for(self, codebook: int, entry: int) -> str:
        return self.entry_to_label.get((codebook, entry), "unknown")

    def decode_index_sequence(self, codebook_idx: int,
                            indices: list[int] | np.ndarray) -> list[str]:
        # Returns per-step label, deduplicating consecutive repeats
        out: list[str] = []
        prev: str | None = None

        for e in indices:
            label = self.entry_to_label.get((codebook_idx, int(e)), "unknown")

            if label != prev:
                out.append(label)
                prev = label

        return out

    def load_encoder(self) -> None:
        # Lazy SBERT load — only fired the first time decompose_text is called
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.label_list = self.labels()
        # Use a short sentence form for SBERT so the embedding focuses on action verb
        prompts = [f"a person {lab.replace('_', ' ')}" for lab in self.label_list]
        self.label_embeds = self.encoder.encode(prompts, normalize_embeddings=True)
        log.info("[vocab] SBERT-encoded %d labels for compound decomposition",
                 len(self.label_list))

    def decompose_text(self, text: str) -> list[tuple[str, float]]:
        # Returns [(label, score), ...] one per atomic segment of the prompt
        if self.encoder is None:
            self.load_encoder()
        assert self.encoder is not None and self.label_embeds is not None

        segments = [s.strip() for s in CONNECTIVES.split(text) if s.strip()]

        if not segments:
            return []
        seg_embeds = self.encoder.encode(segments, normalize_embeddings=True)
        sims = seg_embeds @ self.label_embeds.T  # (n_seg, n_label)
        best_idx = sims.argmax(axis=1)
        out: list[tuple[str, float]] = []

        for i, label_idx in enumerate(best_idx):
            out.append((self.label_list[int(label_idx)], float(sims[i, label_idx])))

        return out
