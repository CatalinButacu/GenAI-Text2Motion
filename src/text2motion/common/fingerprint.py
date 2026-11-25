from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

HASH_CHARS = 12


def recipe_hash(recipe: Mapping[str, object]) -> str:
    text = json.dumps(recipe, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]


def output_folder(root: Path, source: str, recipe: Mapping[str, object]) -> Path:
    return root / f"{source}-{recipe_hash(recipe)}"
