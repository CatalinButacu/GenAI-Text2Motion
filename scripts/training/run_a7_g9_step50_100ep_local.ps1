# powershell -ExecutionPolicy Bypass -File scripts/training/run_a7_g9_step50_100ep_local.ps1
param([double]$BudgetHoursPerRun = 12)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
Write-Host "=== A7 G9 step50 transformer 100 epochs start $(Get-Date -Format u) ==="

$code = Invoke-TrainingStep -Tag "a7_g9_step50_100ep_transformer" -LogFile "runs/outputs/a7_g9_step50_100ep_transformer_w320.log" `
    -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments @(
        "-m", "text2motion.app.cli", "train-generator", "--config", "configs/generator/a7/width_320_step50.yaml",
        "--backbone", "transformer", "--tokenizer_ckpt", $tokenizer,
        "--overfit_gate", "runs/outputs/gates/a7_step50_transformer_w320_overfit.json",
        "--ckpt_name", "generator_transformer_w320_a7_g9_100ep.pt",
        "--epochs", "100", "--val_ce_patience", "6", "--save_r1_floor", "0.490",
        "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
        "--cfg_scale", "6.0", "--temperature", "1.0"
    )
if (-not (Write-DoneMarker -Path "runs/outputs/A7_G9_DONE.txt" -ExitCodes @($code))) {
    exit 1
}
