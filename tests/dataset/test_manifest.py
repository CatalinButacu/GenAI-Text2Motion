from pathlib import Path

import pytest

from text2motion.common.manifest import check_recipe_reuse, write_manifest


def test_same_recipe_is_accepted(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, {"fps": 60}, {"complete": True})
    check_recipe_reuse(manifest, {"fps": 60}, has_outputs=True)


def test_changed_recipe_is_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, {"fps": 60}, {"complete": True})
    with pytest.raises(ValueError, match="different: \\['fps'\\]"):
        check_recipe_reuse(manifest, {"fps": 20}, has_outputs=True)


def test_legacy_folder_with_outputs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="legacy folder"):
        check_recipe_reuse(tmp_path / "manifest.json", {"fps": 60}, has_outputs=True)


def test_empty_folder_is_accepted(tmp_path: Path) -> None:
    check_recipe_reuse(tmp_path / "manifest.json", {"fps": 60}, has_outputs=False)
