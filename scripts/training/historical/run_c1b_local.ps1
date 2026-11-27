# powershell -ExecutionPolicy Bypass -File scripts/training/historical/run_c1b_local.ps1
param([double]$BudgetHoursPerRun = 5)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "../TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
Write-Host "=== C1b fresh matched pair start $(Get-Date -Format u) ==="

$common = @(
    "--epochs", "500", "--window", "64", "--batch_size", "128", "--ema_decay", "0.99",
    "--eval_every", "25", "--num_workers", "0"
)
$runs = @(
    @{ Tag = "c1b_fsq"; Config = "configs/tokenizer/fsq_g8_v1024.yaml"; Kind = "fsq"; Name = "fsq_g8_v1024_c1b.pt" },
    @{ Tag = "c1b_rvq"; Config = "configs/tokenizer/rvq_l8_1024.yaml"; Kind = "rvq"; Name = "rvq_l8_1024_c1b.pt" }
)

$codes = @()
foreach ($run in $runs) {
    $arguments = @(
        "scripts/training/train_tokenizer_challenger.py", "--config", $run.Config,
        "--tokenizer", $run.Kind, "--ckpt_name", $run.Name
    ) + $common
    $code = Invoke-TrainingStep -Tag $run.Tag -LogFile "runs/outputs/$($run.Tag).log" `
        -MaxHours $BudgetHoursPerRun -Arguments $arguments
    $codes += $code
    if ($code -ne 0) {
        Write-Host "STOP: $($run.Tag) failed with code $code"
        exit 1
    }
}

$code = Invoke-TrainingStep -Tag "c1b_bootstrap" -LogFile "runs/outputs/c1b_bootstrap.log" -MaxHours 2 -Arguments @(
    "scripts/evaluation/tokenizer_pair_bootstrap.py", "--split", "val", "--resamples", "1000",
    "--pair", "fsq_g8_v1024_c1b", "rvq_l8_1024_c1b",
    "--out", "runs/outputs/qa/tokenizer_pair_bootstrap_c1b.json"
)
$codes += $code
if ($code -ne 0) {
    Write-Host "STOP: c1b_bootstrap failed with code $code"
    exit 1
}

if (-not (Write-DoneMarker -Path "runs/outputs/C1B_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
