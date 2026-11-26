from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SPLIT_FILES = ("train", "val", "test")
HELD_OUT = ("val", "test")


@dataclass(frozen=True)
class SplitPlan:
    split_of: Mapping[str, str]
    source_of: Mapping[str, str]

    @property
    def protected_sources(self) -> frozenset[str]:
        return frozenset(
            self.source_of[name]
            for name, split in self.split_of.items()
            if split in HELD_OUT and name in self.source_of
        )

    def names(self, split: str) -> frozenset[str]:
        return frozenset(name for name, found in self.split_of.items() if found == split)

    def training_names(self) -> frozenset[str]:
        protected = self.protected_sources
        return frozenset(
            name
            for name in self.names("train")
            if name in self.source_of and self.source_of[name] not in protected
        )

    @classmethod
    def from_official(cls, folder: Path) -> SplitPlan:
        split_of: dict[str, str] = {}
        for split in SPLIT_FILES:
            for line in (folder / f"{split}.txt").read_text(encoding="utf-8").splitlines():
                name = line.strip().removeprefix("M")
                if name:
                    split_of[name] = split
        with open(folder / "index.csv", newline="", encoding="utf-8") as handle:
            source_of = {
                row["new_name"].removesuffix(".npy"): row["source_path"]
                for row in csv.DictReader(handle)
            }
        return cls(split_of=split_of, source_of=source_of)
