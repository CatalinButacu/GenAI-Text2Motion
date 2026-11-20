# powershell -ExecutionPolicy Bypass -File scripts/training/run_a18_equal_depth_local.ps1
param([double]$BudgetHours = 1.5)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
$config = "configs/generator/a18/width_320_mamba8_expand4.yaml"
$arm = "a18_mamba8_e4"
$name = "generator_mamba_w320_$arm.pt"
Write-Host "=== A18 ED1 start $(Get-Date -Format u) ==="
if (Test-Path "runs/checkpoints/generator/generator_mamba_w320_${arm}_last.pt") {
    throw "a run with this name exists; a kill-test run is never resumed: $name"
}

$codes = @()
$gate = "runs/outputs/gates/${arm}_mamba_w320_overfit.json"
$code = Invoke-TrainingStep -Tag "gate_$arm" -LogFile "runs/outputs/${arm}_gate_mamba_w320.log" `
    -MaxHours 0.5 -Arguments @(
        "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config,
        "--backbone", "mamba", "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
    )
$codes += $code
if ($code -ne 0) { Write-Host "STOP: gate code $code"; exit 1 }
$code = Invoke-TrainingStep -Tag $arm -LogFile "runs/outputs/${arm}_mamba_w320.log" `
    -MaxHours $BudgetHours -StaleMinutes 45 -Arguments @(
        "-m", "text2motion.app.cli", "train-generator", "--config", $config,
        "--backbone", "mamba", "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
        "--ckpt_name", $name, "--epochs", "60", "--stop_epoch", "6", "--val_ce_patience", "6",
        "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
        "--cfg_scale", "6.0", "--temperature", "1.0"
    )
$codes += $code
if ($code -ne 0) { Write-Host "STOP: run code $code"; exit 1 }

if (-not (Write-DoneMarker -Path "runs/outputs/A18_ED1_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
