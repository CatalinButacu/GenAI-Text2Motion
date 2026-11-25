from __future__ import annotations

import json
from pathlib import Path


def check_recipe_reuse(manifest_path: Path, recipe: dict[str, object], has_outputs: bool) -> None:
    folder = manifest_path.parent
    if not manifest_path.is_file():
        if has_outputs:
            raise ValueError(
                f"{folder} has outputs but no {manifest_path.name} (legacy folder); "
                "build into a new folder"
            )
        return
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))["recipe"]
    changed = sorted(key for key in set(stored) | set(recipe) if stored.get(key) != recipe.get(key))
    if changed:
        raise ValueError(
            f"{folder} was built with another recipe (different: {changed}); "
            "build into a new folder"
        )


def write_manifest(
    manifest_path: Path, recipe: dict[str, object], status: dict[str, object]
) -> None:
    manifest = {"recipe": recipe, **status}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
