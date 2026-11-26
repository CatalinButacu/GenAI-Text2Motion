from pathlib import Path

from text2motion.dataset.contracts import DatasetPaths


def test_notes_folder_is_docs_notes(tmp_path: Path) -> None:
    paths = DatasetPaths(tmp_path, tmp_path / "amass")
    assert paths.notes == tmp_path / "docs" / "notes"


def test_default_output_stays_under_data(tmp_path: Path) -> None:
    paths = DatasetPaths(tmp_path, tmp_path / "amass")
    assert paths.hml3d_at(60) == tmp_path / "data" / "HumanML3D_60fps"


def test_work_root_gives_each_run_its_own_output(tmp_path: Path) -> None:
    first = DatasetPaths(tmp_path, tmp_path / "amass", work=tmp_path / "work" / "ACCAD")
    second = DatasetPaths(tmp_path, tmp_path / "amass", work=tmp_path / "work" / "KIT")
    assert first.hml3d_at(60) == tmp_path / "work" / "ACCAD" / "HumanML3D_60fps"
    assert first.hml3d_at(60) != second.hml3d_at(60)


def test_clock_csv_follows_the_work_root(tmp_path: Path) -> None:
    default = DatasetPaths(tmp_path, tmp_path / "amass")
    scoped = DatasetPaths(tmp_path, tmp_path / "amass", work=tmp_path / "work" / "KIT")
    assert default.clock_csv == tmp_path / "docs" / "notes" / "amass_clock_check.csv"
    assert scoped.clock_csv == tmp_path / "work" / "KIT" / "amass_clock_check.csv"
