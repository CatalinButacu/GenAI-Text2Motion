import ctypes
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
DRIVER = MODULE.parent / "historical" / "run_size_sweep_local.ps1"


def powershell(script: str, cwd: Path) -> str:
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


def step(tmp_path: Path, code: str, max_hours: float = 1.0) -> str:
    return (
        f"$code = Invoke-TrainingStep -Tag probe -Arguments @('-c', '{code}') "
        f"-LogFile '{tmp_path / 'probe.log'}' -Python '{sys.executable}' "
        f"-MaxHours {max_hours} -PollSeconds 1; "
        f"$ok = Write-DoneMarker -Path '{tmp_path / 'DONE.txt'}' -ExitCodes @($code); "
        'Write-Output "code=$code ok=$ok"'
    )


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "runs" / "outputs").mkdir(parents=True)
    return tmp_path


def test_a_failed_step_returns_its_exit_code_and_writes_no_done_marker(workdir):
    output = powershell(step(workdir, "import sys; sys.exit(3)"), workdir)

    assert "code=3 ok=False" in output
    assert not (workdir / "DONE.txt").exists()


def test_a_successful_step_writes_the_done_marker_and_prints_its_command(workdir):
    output = powershell(step(workdir, "print(1)"), workdir)

    assert "code=0 ok=True" in output
    assert '-u -c "print(1)"' in output or "-u -c print(1)" in output
    assert (workdir / "DONE.txt").exists()


def test_a_step_over_its_time_budget_is_killed_and_logged(workdir):
    output = powershell(step(workdir, "import time; time.sleep(60)", max_hours=0.0005), workdir)

    assert "code=124 ok=False" in output
    assert "KILLED" in (workdir / "runs" / "outputs" / "GUARD_KILL.txt").read_text("utf-8-sig")


def test_time_cap_stops_the_actual_python_child_not_only_its_venv_launcher(workdir):
    code = 'import os,time; open("child.pid","w").write(str(os.getpid())); time.sleep(60)'
    output = powershell(step(workdir, code, max_hours=0.001), workdir)
    child_pid = int((workdir / "child.pid").read_text())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, 0, child_pid)
    if handle:
        try:
            assert kernel.WaitForSingleObject(handle, 0) == 0
        finally:
            kernel.CloseHandle(handle)
    else:
        assert ctypes.get_last_error() == 87
    assert "code=124 ok=False" in output


def test_a_run_is_complete_only_when_its_resume_file_reached_the_last_epoch(workdir):
    resume = workdir / "run_last.pt"
    torch.save({"epoch": 59}, resume)
    script = (
        f"$python = '{sys.executable}'; "
        f"Write-Output \"e60=$(Test-RunComplete -ResumePath '{resume}' -Epochs 60 -Python $python)\"; "
        f"Write-Output \"e61=$(Test-RunComplete -ResumePath '{resume}' -Epochs 61 -Python $python)\"; "
        f"Write-Output \"none=$(Test-RunComplete -ResumePath '{workdir / 'x_last.pt'}' -Epochs 1)\""
    )

    output = powershell(script, workdir)

    assert "e60=True" in output
    assert "e61=False" in output
    assert "none=False" in output


def test_an_empty_resume_flag_does_not_reach_the_command_line(workdir):
    script = (
        f"$null = Invoke-TrainingStep -Tag probe -Arguments @('-c', 'print(1)', '') "
        f"-LogFile '{workdir / 'probe.log'}' -Python '{sys.executable}' -PollSeconds 1"
    )

    output = powershell(script, workdir)

    assert '-c "print(1)" ""' not in output
    assert "code 0" in output


def test_resume_is_passed_only_when_the_resume_file_exists(workdir):
    existing = workdir / "run_last.pt"
    existing.write_bytes(b"x")
    script = (
        f"Write-Output \"a=[$(Get-ResumeFlag -ResumePath '{existing}')]\"; "
        f"Write-Output \"b=[$(Get-ResumeFlag -ResumePath '{workdir / 'missing_last.pt'}')]\""
    )

    output = powershell(script, workdir)

    assert "a=[--resume]" in output
    assert "b=[]" in output


def test_streaming_prints_both_logs_without_changing_the_exit_code(workdir):
    script = (
        f"$code = Invoke-TrainingStep -Tag probe -Arguments @('-c', "
        """'import sys; print("visible-out"); print("visible-err", file=sys.stderr)') """
        f"-LogFile '{workdir / 'stream.log'}' -Python '{sys.executable}' "
        '-StreamLogs -PollSeconds 1; Write-Output "code=$code"'
    )
    output = powershell(script, workdir)

    assert "code=0" in output
    assert "visible-out" in output
    assert "visible-err" in output
    assert "visible-out" in (workdir / "stream.log").read_text()
    assert "visible-err" in (workdir / "stream.log.err").read_text()


def test_a_stale_process_has_a_distinct_exit_code_when_requested(workdir):
    script = (
        "& (Get-Module TrainingRun) { function script:Get-NewestHeartbeat { "
        "param([datetime]$Since); [pscustomobject]@{LastWriteTime=$Since.AddHours(-1)} } }; "
        f"$code = Invoke-TrainingStep -Tag stale -Arguments @('-c', 'import time; time.sleep(60)') "
        f"-LogFile '{workdir / 'stale.log'}' -Python '{sys.executable}' "
        "-StaleMinutes 1 -StaleExitCode 125 -PollSeconds 1; "
        'Write-Output "code=$code"'
    )
    output = powershell(script, workdir)

    assert "code=125" in output
    assert "no heartbeat" in output


def pilot(
    workdir: Path,
    transformer: int,
    mamba: int,
    gate: int = 0,
    recover: bool = False,
    default_recipe: bool = False,
):
    (workdir / "runs" / "checkpoints" / "generator").mkdir(parents=True, exist_ok=True)
    (workdir / "runs" / "outputs" / "gates").mkdir(exist_ok=True)
    driver = workdir / "historical" / DRIVER.name
    driver.parent.mkdir(exist_ok=True)
    driver.write_text(DRIVER.read_text(encoding="utf-8"), encoding="utf-8")
    module = f"""
function Enter-RepoRoot {{ return (Get-Location).Path }}
function Invoke-TrainingStep {{
    param($Tag, $Arguments, $LogFile, $MaxHours, $StaleMinutes, $StaleExitCode, $PollSeconds, [switch]$StreamLogs)
    "$Tag|$MaxHours|$StaleExitCode|$($Arguments -join ' ')" | Add-Content calls.txt
    if ($Arguments -contains 'sanity-overfit') {{ return {gate} }}
    if ($Tag -like '*_transformer_*') {{ return {transformer} }}
    return {mamba}
}}
function Write-DoneMarker {{
    param($Path, $ExitCodes)
    if (@($ExitCodes | Where-Object {{ $_ -ne 0 }}).Count) {{ return $false }}
    'done' | Set-Content $Path
    return $true
}}
function Get-ResumeFlag {{
    param($ResumePath)
    if (Test-Path $ResumePath) {{ return '--resume' }}
    return ''
}}
Export-ModuleMember -Function Enter-RepoRoot, Invoke-TrainingStep, Write-DoneMarker, Get-ResumeFlag
"""
    (workdir / MODULE.name).write_text(module, encoding="utf-8")
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(driver),
            "-Pilot",
            "-Tag",
            "pilot_test",
            "-Epochs",
            "60",
            "-BudgetHoursPerRun",
            "1",
            *([] if default_recipe else ["-Widths", "256", "-ValCePatience", "3"]),
            *(["-ResumePilot"] if recover else []),
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("transformer", "mamba", "gate", "expected_code", "expected_calls"),
    [(124, 124, 0, 124, 4), (0, 0, 0, 0, 4), (3, 0, 0, 1, 3), (125, 0, 0, 1, 3), (0, 0, 3, 1, 1)],
)
def test_pilot_gates_both_twins_before_training_and_only_continues_time_caps(
    workdir, transformer, mamba, gate, expected_code, expected_calls
):
    result = pilot(workdir, transformer, mamba, gate)
    calls = (workdir / "calls.txt").read_text().splitlines()

    assert result.returncode == expected_code, result.stdout + result.stderr
    assert len(calls) == expected_calls
    assert "sanity-overfit" in calls[0]
    if gate == 0:
        assert "sanity-overfit" in calls[1]
        assert "transformer" in calls[0] and "mamba" in calls[1]
        assert all("train-generator" in call for call in calls[2:])
        for call in calls[2:]:
            assert "|1|125|" in call
            assert "width_256.yaml" in call
            assert "--val_ce_patience 3" in call and "--epochs 60" in call
            assert "--batch_size 8" in call and "--num_workers 0" in call
            assert "--resume" not in call and "--init_ckpt" not in call
    outputs = workdir / "runs" / "outputs"
    assert (outputs / "PILOT_TEST_DONE.txt").exists() == (expected_code == 0)
    assert (outputs / "PILOT_TEST_PARTIAL.txt").exists() == (expected_code == 124)


def test_pilot_refuses_existing_outputs_before_launching_a_gate(workdir):
    (workdir / "runs" / "outputs" / "pilot_test_existing.log").write_text("preserve")
    result = pilot(workdir, 0, 0)

    assert result.returncode != 0
    assert "choose a fresh Tag" in result.stderr
    assert not (workdir / "calls.txt").exists()


def test_default_recipe_is_width320_with_six_epoch_ce_patience_for_both_twins(workdir):
    result = pilot(workdir, 124, 124, default_recipe=True)
    calls = (workdir / "calls.txt").read_text().splitlines()

    assert result.returncode == 124, result.stdout + result.stderr
    assert len(calls) == 4
    assert all("width_320.yaml" in call for call in calls)
    assert all("--val_ce_patience 6" in call for call in calls[2:])
    assert "transformer" in calls[2] and "mamba" in calls[3]


def test_recovery_keeps_original_time_budget_and_does_not_resume_the_unstarted_twin(workdir):
    outputs = workdir / "runs" / "outputs"
    (outputs / "gates").mkdir()
    for backbone in ("transformer", "mamba"):
        (outputs / "gates" / f"pilot_test_{backbone}_w256_overfit.json").write_text("passed")
    (outputs / "pilot_test_transformer_w256.log").write_text("original evidence")
    checkpoints = workdir / "runs" / "checkpoints" / "generator"
    checkpoints.mkdir(parents=True)
    (checkpoints / "generator_transformer_w256_pilot_test_last.pt").write_bytes(b"checkpoint")
    result = pilot(workdir, 124, 124, recover=True)
    calls = (workdir / "calls.txt").read_text().splitlines()

    assert result.returncode == 124, result.stdout + result.stderr
    assert len(calls) == 2
    assert "--resume" in calls[0] and "--resume" not in calls[1]
    assert 0 < float(calls[0].split("|")[1]) < 1
    assert float(calls[1].split("|")[1]) == 1
    assert (outputs / "pilot_test_transformer_w256.log").read_text() == "original evidence"


def test_recovery_refuses_to_restart_an_interrupted_run_without_its_checkpoint(workdir):
    outputs = workdir / "runs" / "outputs"
    (outputs / "gates").mkdir()
    for backbone in ("transformer", "mamba"):
        (outputs / "gates" / f"pilot_test_{backbone}_w256_overfit.json").write_text("passed")
    (outputs / "pilot_test_transformer_w256.log").write_text("original evidence")
    result = pilot(workdir, 0, 0, recover=True)

    assert result.returncode != 0
    assert "durable checkpoint" in result.stderr
    assert not (workdir / "calls.txt").exists()


def test_recovery_skips_an_expired_original_budget_but_runs_the_unstarted_twin(workdir):
    outputs = workdir / "runs" / "outputs"
    (outputs / "gates").mkdir()
    for backbone in ("transformer", "mamba"):
        (outputs / "gates" / f"pilot_test_{backbone}_w256_overfit.json").write_text("passed")
    original = outputs / "pilot_test_transformer_w256.log"
    original.write_text("original evidence")
    powershell(
        f"(Get-Item '{original}').CreationTimeUtc = [datetime]::UtcNow.AddHours(-2)", workdir
    )
    result = pilot(workdir, 0, 0, recover=True)
    calls = (workdir / "calls.txt").read_text().splitlines()

    assert result.returncode == 124, result.stdout + result.stderr
    assert len(calls) == 1 and "mamba" in calls[0]
    assert "no time remaining" in result.stdout
    assert original.read_text() == "original evidence"
