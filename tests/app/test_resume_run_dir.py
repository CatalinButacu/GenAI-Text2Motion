import pytest
import torch

from text2motion.app.run_log import resumed_run_dir


def test_a_resume_continues_the_run_directory_that_the_resume_file_records(tmp_path):
    run_dir = tmp_path / "logs" / "run"
    resume_path = tmp_path / "generator_last.pt"
    torch.save({"run_dir": str(run_dir)}, resume_path)

    assert resumed_run_dir(resume_path, None) == run_dir
    assert resumed_run_dir(resume_path, run_dir) == run_dir


def test_a_requested_run_directory_that_differs_from_the_recorded_one_fails(tmp_path):
    resume_path = tmp_path / "generator_last.pt"
    torch.save({"run_dir": str(tmp_path / "logs" / "run")}, resume_path)

    with pytest.raises(ValueError, match="differs"):
        resumed_run_dir(resume_path, tmp_path / "logs" / "other")


def test_an_old_resume_file_uses_the_requested_run_directory(tmp_path):
    resume_path = tmp_path / "generator_last.pt"
    torch.save({"epoch": 3}, resume_path)

    assert resumed_run_dir(resume_path, tmp_path / "run") == tmp_path / "run"
    assert resumed_run_dir(resume_path, None) is None


def test_a_resume_without_the_resume_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resumed_run_dir(tmp_path / "generator_last.pt", None)
