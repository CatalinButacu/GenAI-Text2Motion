from pathlib import Path

from text2motion.dataset import pipeline
from text2motion.dataset.contracts import DatasetPaths
from text2motion.motion.contracts import PreparationStage


def stub_all(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(pipeline, "write_clock_table", lambda *a, **k: calls.append("clock"))
    monkeypatch.setattr(pipeline, "build_joint_clips", lambda *a, **k: calls.append("joints"))
    monkeypatch.setattr(pipeline, "encode_hml3d_features", lambda *a, **k: calls.append("features"))
    monkeypatch.setattr(pipeline, "fit_hml3d_statistics", lambda *a, **k: calls.append("stats"))


def make_request(root: Path, fill: bool):
    official = root / "data" / "HumanML3D_official"
    official.mkdir(parents=True, exist_ok=True)
    (official / "index.csv").write_text(
        "source_path,start_frame,end_frame,new_name\n", encoding="utf-8"
    )
    paths = DatasetPaths(root, root / "amass", work=root / "out")
    paths.clock_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.clock_csv.write_text("source_path\n", encoding="utf-8")
    return pipeline.request_from_paths(paths, fps=60, fill=fill)


def test_steps_run_in_order_once(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_all(monkeypatch, calls)
    pipeline.run_pipeline(make_request(tmp_path, fill=True))
    assert calls == ["clock", "joints", "features", "stats"]


def test_same_recipe_skips_finished_steps(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_all(monkeypatch, calls)
    request = make_request(tmp_path, fill=True)
    pipeline.run_pipeline(request)
    pipeline.run_pipeline(request)
    assert calls == ["clock", "joints", "features", "stats"]


def test_changed_recipe_runs_again(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_all(monkeypatch, calls)
    pipeline.run_pipeline(make_request(tmp_path, fill=True))
    pipeline.run_pipeline(make_request(tmp_path, fill=False))
    assert calls == ["clock", "joints", "features", "stats", "clock", "joints", "features"]


def test_partial_run_without_fill_skips_statistics(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_all(monkeypatch, calls)
    pipeline.run_pipeline(make_request(tmp_path, fill=False))
    assert calls == ["clock", "joints", "features"]


def test_one_stage_runs_alone_without_its_inputs(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_all(monkeypatch, calls)
    pipeline.run_pipeline(make_request(tmp_path, fill=True), PreparationStage.FEATURES)
    assert calls == ["features"]
