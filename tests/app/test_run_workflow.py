import json
from types import SimpleNamespace

import pytest

from text2motion.app import cli
from text2motion.app.workflows.requests import RunRequest, Workflow
from text2motion.app.workflows.run import MANIFEST_NAME, run_workflow
from text2motion.common.fingerprint import recipe_hash
from text2motion.dataset.fixture import build_fixture_dataset
from text2motion.tokenization.contracts import TokenizerKind


def test_run_help_lists_the_four_workflows(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["run", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for name in ("data", "tokenizer", "generator", "evaluate"):
        assert name in text


def test_data_then_tokenizer_on_the_fixture_writes_one_entry_per_step(tmp_path, monkeypatch):
    dataset = tmp_path / "fixture"
    build_fixture_dataset(dataset)
    calls = []
    monkeypatch.setattr(
        "text2motion.app.training.run_prepare", lambda app, args: calls.append("prepare")
    )
    monkeypatch.setattr(
        "text2motion.app.training.run_train_tokenizer",
        lambda app, args: calls.append(f"train-tokenizer:{args.tokenizer.value}"),
    )
    app = SimpleNamespace(config=SimpleNamespace(paths=SimpleNamespace(hml3d_out_dir=dataset)))
    request = RunRequest(
        workflow=Workflow.TOKENIZER, root=tmp_path / "run", tokenizer=TokenizerKind.FINITE_SCALAR
    )

    run_workflow(app, request)
    run_workflow(app, request)

    assert calls == ["prepare", f"train-tokenizer:{TokenizerKind.FINITE_SCALAR.value}"]
    manifest = json.loads((request.root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert sorted(manifest["steps"]) == ["data", "tokenizer"]
    stamp = recipe_hash(manifest["recipe"])
    assert all(entry["recipe_hash"] == stamp for entry in manifest["steps"].values())
    assert (request.root / f"tokenizer.{stamp}").is_file()
    assert (request.root / "data").read_text(encoding="utf-8") == str(dataset)
