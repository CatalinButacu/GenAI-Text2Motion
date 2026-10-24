import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="the training scripts are Windows PowerShell scripts",
)

MODULE = Path("scripts/training/TrainingRun.psm1").resolve()


def __powershell(script: str, cwd: Path) -> str:
    command = f"Import-Module '{MODULE}' -Force; {script}"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
        check=False,
    )
    return result.stdout + result.stderr


def __step(tmp_path: Path, code: str, max_hours: float = 1.0) -> str:
    return (
        f"$code = Invoke-TrainingStep -Tag probe -Arguments @('-c', '{code}') "
        f"-LogFile '{tmp_path / 'probe.log'}' -Python '{sys.executable}' "
        f"-MaxHours {max_hours} -PollSeconds 1; "
        f"$ok = Write-DoneMarker -Path '{tmp_path / 'DONE.txt'}' -ExitCodes @($code); "
        'Write-Output "code=$code ok=$ok"'
    )


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "artifacts" / "outputs").mkdir(parents=True)
    return tmp_path


def test_a_failed_step_returns_its_exit_code_and_writes_no_done_marker(workdir):
    output = __powershell(__step(workdir, "import sys; sys.exit(3)"), workdir)

    assert "code=3 ok=False" in output
    assert not (workdir / "DONE.txt").exists()


def test_a_successful_step_writes_the_done_marker_and_prints_its_command(workdir):
    output = __powershell(__step(workdir, "print(1)"), workdir)

    assert "code=0 ok=True" in output
    assert '-u -c "print(1)"' in output or "-u -c print(1)" in output
    assert (workdir / "DONE.txt").exists()


def test_a_step_over_its_time_budget_is_killed_and_logged(workdir):
    output = __powershell(__step(workdir, "import time; time.sleep(60)", max_hours=0.0005), workdir)

    assert "code=124 ok=False" in output
    assert "KILLED" in (workdir / "artifacts" / "outputs" / "GUARD_KILL.txt").read_text("utf-8-sig")


def test_a_run_is_complete_only_when_its_resume_file_reached_the_last_epoch(workdir):
    resume = workdir / "run_last.pt"
    torch.save({"epoch": 59}, resume)
    script = (
        f"$python = '{sys.executable}'; "
        f"Write-Output \"e60=$(Test-RunComplete -ResumePath '{resume}' -Epochs 60 -Python $python)\"; "
        f"Write-Output \"e61=$(Test-RunComplete -ResumePath '{resume}' -Epochs 61 -Python $python)\"; "
        f"Write-Output \"none=$(Test-RunComplete -ResumePath '{workdir / 'x_last.pt'}' -Epochs 1)\""
    )

    output = __powershell(script, workdir)

    assert "e60=True" in output
    assert "e61=False" in output
    assert "none=False" in output


def test_an_empty_resume_flag_does_not_reach_the_command_line(workdir):
    script = (
        f"$null = Invoke-TrainingStep -Tag probe -Arguments @('-c', 'print(1)', '') "
        f"-LogFile '{workdir / 'probe.log'}' -Python '{sys.executable}' -PollSeconds 1"
    )

    output = __powershell(script, workdir)

    assert '-c "print(1)" ""' not in output
    assert "code 0" in output


def test_resume_is_passed_only_when_the_resume_file_exists(workdir):
    existing = workdir / "run_last.pt"
    existing.write_bytes(b"x")
    script = (
        f"Write-Output \"a=[$(Get-ResumeFlag -ResumePath '{existing}')]\"; "
        f"Write-Output \"b=[$(Get-ResumeFlag -ResumePath '{workdir / 'missing_last.pt'}')]\""
    )

    output = __powershell(script, workdir)

    assert "a=[--resume]" in output
    assert "b=[]" in output
