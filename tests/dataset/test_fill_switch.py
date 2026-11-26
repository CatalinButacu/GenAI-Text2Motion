from pathlib import Path

from text2motion.dataset import pipeline
from text2motion.dataset.contracts import DatasetPaths
from text2motion.dataset.sources.humanml3d import build


def make_repo(repo: Path) -> None:
    official = repo / "data" / "HumanML3D_official"
    official.mkdir(parents=True)
    (official / "index.csv").write_text(
        "source_path,start_frame,end_frame,new_name\n./pose_data/A/a.npy,0,10,000001.npy\n",
        encoding="utf-8",
    )
    for name, text in (("train.txt", "000001\n"), ("val.txt", ""), ("test.txt", "")):
        (official / name).write_text(text, encoding="utf-8")
    clock = repo / "work" / "ACCAD" / "amass_clock_check.csv"
    clock.parent.mkdir(parents=True)
    clock.write_text(
        "source_path,dataset,label_fps,clips,corr_60,corr_100,corr_120,stored_fps,margin,status\n",
        encoding="utf-8",
    )


def stub_steps(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(build, "extract_poses", lambda *a, **k: calls.append("extract"))
    monkeypatch.setattr(build, "write_clocked_poses", lambda *a, **k: {})
    monkeypatch.setattr(build, "build_indexed_joint_clips", lambda *a, **k: calls.append("index"))
    monkeypatch.setattr(build, "fill_from_official", lambda *a, **k: calls.append("fill") or 0)


def make_request(tmp_path: Path, fill: bool):
    make_repo(tmp_path)
    paths = DatasetPaths(tmp_path, tmp_path / "amass", work=tmp_path / "work" / "ACCAD")
    return pipeline.request_from_paths(paths, fps=60, fill=fill)


def test_fill_is_skipped_when_switched_off(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_steps(monkeypatch, calls)
    build.build_joint_clips(make_request(tmp_path, fill=False))
    assert "fill" not in calls
    assert "index" in calls


def test_fill_runs_by_default(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    stub_steps(monkeypatch, calls)
    build.build_joint_clips(make_request(tmp_path, fill=True))
    assert "fill" in calls
