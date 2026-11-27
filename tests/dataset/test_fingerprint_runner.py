from pathlib import Path

import pytest

from text2motion.common.fingerprint import output_folder, recipe_hash
from text2motion.common.runner import Step, run_cached_steps


def test_hash_is_the_same_for_the_same_recipe_in_any_key_order() -> None:
    assert recipe_hash({"a": 1, "b": 2}) == recipe_hash({"b": 2, "a": 1})


def test_hash_changes_when_the_recipe_changes() -> None:
    assert recipe_hash({"fps": 60}) != recipe_hash({"fps": 20})


def test_output_folder_names_the_source_and_the_recipe(tmp_path: Path) -> None:
    folder = output_folder(tmp_path, "amass", {"fps": 60})
    assert folder.parent == tmp_path
    assert folder.name.startswith("amass-")
    assert folder.name.endswith(recipe_hash({"fps": 60}))


def test_runner_runs_steps_in_dependency_order(tmp_path: Path) -> None:
    order: list[str] = []

    def make(name: str):
        def run(inputs: dict[str, Path], out: Path) -> None:
            order.append(name)
            out.write_text(name, encoding="utf-8")

        return run

    steps = [
        Step("features", ("joints",), "features.txt", make("features")),
        Step("joints", (), "joints.txt", make("joints")),
    ]
    run_cached_steps(steps, tmp_path, {"recipe": {"fps": 60}})
    assert order == ["joints", "features"]


def test_runner_skips_finished_steps_and_reruns_on_recipe_change(tmp_path: Path) -> None:
    calls: list[str] = []

    def run(inputs: dict[str, Path], out: Path) -> None:
        calls.append("x")
        out.write_text("x", encoding="utf-8")

    steps = [Step("joints", (), "joints.txt", run)]
    run_cached_steps(steps, tmp_path, {"recipe": {"fps": 60}})
    run_cached_steps(steps, tmp_path, {"recipe": {"fps": 60}})
    assert calls == ["x"]
    run_cached_steps(steps, tmp_path, {"recipe": {"fps": 20}})
    assert calls == ["x", "x"]


def test_runner_refuses_a_missing_input(tmp_path: Path) -> None:
    steps = [Step("features", ("joints",), "features.txt", lambda i, o: None)]
    with pytest.raises(KeyError, match="joints"):
        run_cached_steps(steps, tmp_path, {"recipe": {"fps": 60}})
