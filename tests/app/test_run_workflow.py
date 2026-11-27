import json
from types import SimpleNamespace

import pytest

from text2motion.app import cli
from text2motion.app.config.schema import ProjectPathsConfig, twin_checkpoint_name
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
        "text2motion.app.stages.data.run_prepare", lambda app, args: calls.append("prepare")
    )
    monkeypatch.setattr(
        "text2motion.app.stages.train_tokenizer.run_train_tokenizer",
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


def test_evaluate_workflow_runs_both_twins_from_one_step_list_and_reruns_skip(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "fixture"
    build_fixture_dataset(dataset)
    calls = []

    def record(name):
        def handler(app, args):
            backbone = getattr(args, "backbone", None)
            calls.append(f"{name}:{getattr(backbone, 'value', backbone)}")
            if name == "sanity-overfit":
                args.overfit_gate.write_text("{}", encoding="utf-8")

        return handler

    for name, target in (
        ("prepare", "data.run_prepare"),
        ("train-tokenizer", "train_tokenizer.run_train_tokenizer"),
        ("tokenize", "data.run_tokenize"),
        ("sanity-overfit", "overfit.run_sanity_overfit"),
        ("train-generator", "train_generator.run_train_generator"),
    ):
        monkeypatch.setattr(f"text2motion.app.stages.{target}", record(name))
    monkeypatch.setattr("text2motion.app.stages.evaluate.run_evaluate", record("evaluate"))
    paths = SimpleNamespace(
        hml3d_out_dir=dataset, tokenizer_checkpoint=tmp_path / "tok.pt", checkpoints_dir=tmp_path
    )
    app = SimpleNamespace(config=SimpleNamespace(paths=paths))
    request = RunRequest(
        workflow=Workflow.EVALUATE, root=tmp_path / "run", tokenizer=TokenizerKind.FINITE_SCALAR
    )

    run_workflow(app, request)
    first = list(calls)
    run_workflow(app, request)

    assert calls == first
    twins = [call for call in first if call.startswith(("sanity-overfit", "train-generator"))]
    assert sorted(twins) == [
        "sanity-overfit:mamba",
        "sanity-overfit:transformer",
        "train-generator:mamba",
        "train-generator:transformer",
    ]
    assert first[-1] == "evaluate:both"
    manifest = json.loads((request.root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert len(manifest["steps"]) == 8


def test_pretrain_feeds_each_twin_and_the_run_name_names_every_checkpoint(tmp_path, monkeypatch):
    dataset = tmp_path / "fixture"
    build_fixture_dataset(dataset)
    seen = {}

    def record(name):
        def handler(app, args):
            seen[f"{name}:{args.backbone.value}"] = args
            if name == "sanity-overfit":
                args.overfit_gate.write_text("{}", encoding="utf-8")

        return handler

    def tokenize(app, args):
        seen["tokenize"] = args

    for name, target in (
        ("pretrain", "pretrain.run_pretrain"),
        ("sanity-overfit", "overfit.run_sanity_overfit"),
        ("train-generator", "train_generator.run_train_generator"),
    ):
        monkeypatch.setattr(f"text2motion.app.stages.{target}", record(name))
    monkeypatch.setattr("text2motion.app.stages.data.run_prepare", lambda app, args: None)
    monkeypatch.setattr(
        "text2motion.app.stages.train_tokenizer.run_train_tokenizer", lambda app, args: None
    )
    monkeypatch.setattr("text2motion.app.stages.data.run_tokenize", tokenize)
    paths = SimpleNamespace(
        hml3d_out_dir=dataset, tokenizer_checkpoint=tmp_path / "tok.pt", checkpoints_dir=tmp_path
    )
    app = SimpleNamespace(config=SimpleNamespace(paths=paths))
    request = RunRequest(
        workflow=Workflow.GENERATOR,
        root=tmp_path / "run",
        tokenizer=TokenizerKind.FINITE_SCALAR,
        run_name="causal",
        pretrain_epochs=3,
    )

    run_workflow(app, request)

    for backbone in ("transformer", "mamba"):
        pretrained = tmp_path / "generator" / f"generator_{backbone}_causal_pretrained.pt"
        assert seen[f"pretrain:{backbone}"].ckpt == pretrained
        assert seen[f"pretrain:{backbone}"].token_pack == seen["tokenize"].token_pack
        assert seen[f"pretrain:{backbone}"].epochs == 3
        assert seen[f"sanity-overfit:{backbone}"].init_ckpt == pretrained
        generator = seen[f"train-generator:{backbone}"]
        assert generator.init_ckpt == pretrained
        assert generator.ckpt_name == f"generator_{backbone}_causal.pt"
        assert generator.resume is False
        assert (generator.batch_size, generator.eval_every) == (8, 5)
        assert (generator.val_ce_patience, generator.cfg_scale) == (10, 6.0)


def test_a_run_name_picks_the_twin_checkpoint_instead_of_the_configured_one(tmp_path):
    paths = ProjectPathsConfig(
        checkpoints_dir=tmp_path, generator_checkpoint_paths={"mamba": tmp_path / "old.pt"}
    )

    assert paths.generator_checkpoint("mamba") == tmp_path / "old.pt"
    assert paths.generator_checkpoint("mamba", "causal") == (
        tmp_path / "generator" / "generator_mamba_causal.pt"
    )
    assert paths.generator_checkpoint("transformer", "causal").name == (
        twin_checkpoint_name("transformer", "causal")
    )


def test_a_given_tokenizer_checkpoint_skips_tokenizer_training(tmp_path, monkeypatch):
    dataset = tmp_path / "fixture"
    build_fixture_dataset(dataset)
    calls = []

    def record(name):
        def handler(app, args):
            calls.append(name)
            if name == "sanity-overfit":
                args.overfit_gate.write_text("{}", encoding="utf-8")

        return handler

    for name, target in (
        ("prepare", "data.run_prepare"),
        ("train-tokenizer", "train_tokenizer.run_train_tokenizer"),
        ("tokenize", "data.run_tokenize"),
        ("sanity-overfit", "overfit.run_sanity_overfit"),
        ("train-generator", "train_generator.run_train_generator"),
    ):
        monkeypatch.setattr(f"text2motion.app.stages.{target}", record(name))
    paths = SimpleNamespace(hml3d_out_dir=dataset, checkpoints_dir=tmp_path)
    app = SimpleNamespace(config=SimpleNamespace(paths=paths))
    request = RunRequest(
        workflow=Workflow.GENERATOR, root=tmp_path / "run", tokenizer_ckpt=tmp_path / "tok.pt"
    )

    run_workflow(app, request)

    assert "train-tokenizer" not in calls
    assert calls[:2] == ["prepare", "tokenize"]
