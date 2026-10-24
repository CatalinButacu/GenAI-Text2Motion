# powershell -ExecutionPolicy Bypass -File scripts/training/resume_mamba60.ps1
param([double]$BudgetHoursPerRun = 72, [int]$StaleMinutes = 90)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$runDir = "artifacts/logs/train/20260921T161630Z_generator_mamba"
if (-not (Test-Path $runDir)) {
    throw "run dir not found: $runDir"
}
Write-Host "=== resume mamba 60-epoch twin from $runDir  start $(Get-Date -Format u) ==="

$code = Invoke-TrainingStep -Tag "resume_mamba60" -LogFile "artifacts/outputs/resume_mamba60.log" `
    -MaxHours $BudgetHoursPerRun -StaleMinutes $StaleMinutes -Arguments @(
        "-m", "text2motion.app.cli", "train-generator",
        "--config", "configs/generator/gen_pilot_fsq8x1024.yaml", "--backbone", "mamba",
        "--tokenizer_ckpt", "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt",
        "--overfit_gate", "artifacts/outputs/gates/mamba_34m_overfit.json",
        "--init_ckpt", "artifacts/checkpoints/generator/generator_mamba_pretrained.pt",
        "--ckpt_name", "generator_mamba_bs8_e60.pt", "--epochs", "60", "--batch_size", "8",
        "--num_workers", "4", "--eval_every", "6", "--cfg_scale", "6.0", "--temperature", "1.0",
        "--resume", "--accept_legacy_resume", "--resume_run_dir", $runDir
    )
exit $code
