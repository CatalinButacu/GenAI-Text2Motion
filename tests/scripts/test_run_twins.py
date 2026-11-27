import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="the training scripts are Windows PowerShell scripts",
)

LAUNCHER = Path("scripts/training/run_twins.ps1").resolve()
STUB = """
function Enter-RepoRoot { return (Get-Location).Path }
function Invoke-TrainingStep {
    param($Tag, $Arguments, $LogFile, $MaxHours, $StaleMinutes)
    "$Tag|$LogFile|$MaxHours|$StaleMinutes|$($Arguments -join ' ')" | Set-Content calls.txt
    return EXIT_CODE
}
function Write-DoneMarker {
    param($Path, $ExitCodes)
    if (@($ExitCodes | Where-Object { $_ -ne 0 }).Count) { return $false }
    'done' | Set-Content $Path
    return $true
}
Export-ModuleMember -Function Enter-RepoRoot, Invoke-TrainingStep, Write-DoneMarker
"""


def launch(
    workdir: Path, exit_code: int, *extra: str, tokenizer: str = "tok.pt"
) -> subprocess.CompletedProcess:
    shutil.copy(LAUNCHER, workdir / LAUNCHER.name)
    (workdir / "TrainingRun.psm1").write_text(
        STUB.replace("EXIT_CODE", str(exit_code)), encoding="utf-8"
    )
    (workdir / "twin.yaml").write_text("seed: 2026\n", encoding="utf-8")
    (workdir / "tok.pt").write_bytes(b"")
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(workdir / LAUNCHER.name),
            "-Config",
            "twin.yaml",
            "-RunName",
            "causal",
            "-TokenizerCkpt",
            tokenizer,
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=120,
        check=False,
    )


def test_one_guarded_workflow_call_trains_and_scores_both_twins(tmp_path):
    result = launch(tmp_path, 0, "-PretrainEpochs", "3")

    assert result.returncode == 0, result.stdout + result.stderr
    tag, log, hours, stale, arguments = (
        (tmp_path / "calls.txt").read_text(encoding="utf-8").strip().split("|")
    )
    assert (tag, log, hours, stale) == (
        "twins_causal",
        "runs/outputs/twins/causal/run.log",
        "72",
        "90",
    )
    assert arguments == (
        "-m text2motion.app.cli run --config twin.yaml --workflow EVALUATE "
        "--root runs/outputs/twins/causal --run_name causal --tokenizer_ckpt tok.pt "
        "--generator_epochs 60 --pretrain_epochs 3"
    )
    assert (tmp_path / "runs/outputs/twins/causal/DONE.txt").is_file()


def test_a_failed_workflow_writes_no_done_marker(tmp_path):
    result = launch(tmp_path, 1)

    assert result.returncode == 1
    assert "--pretrain_epochs" not in (tmp_path / "calls.txt").read_text(encoding="utf-8")
    assert not (tmp_path / "runs/outputs/twins/causal/DONE.txt").exists()


def test_a_missing_tokenizer_stops_before_any_run(tmp_path):
    result = launch(tmp_path, 0, tokenizer="missing.pt")

    assert result.returncode != 0
    assert "missing input: missing.pt" in result.stdout + result.stderr
    assert not (tmp_path / "calls.txt").exists()
