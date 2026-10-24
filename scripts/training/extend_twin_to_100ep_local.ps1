# powershell -ExecutionPolicy Bypass -File scripts/training/extend_twin_to_100ep_local.ps1
param([double]$BudgetHoursPerRun = 72, [int]$StaleMinutes = 90)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
Write-Host "=== extend twin 60ep -> 100ep start $(Get-Date -Format u) ==="

function Invoke-Extend100([string]$label, [string]$backbone, [string]$config, [string]$tokenizer, [string]$gate, [string]$ckptName, [int]$numWorkers) {
    $resume = "artifacts/checkpoints/generator/$([IO.Path]::GetFileNameWithoutExtension($ckptName))_last.pt"
    if (Test-RunComplete -ResumePath $resume -Epochs 100) {
        Write-Host "SKIP $label (all 100 epochs are in $resume)"
        return 0
    }
    return Invoke-TrainingStep -Tag $label -LogFile "artifacts/outputs/$label.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes $StaleMinutes -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", $config,
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
            "--ckpt_name", $ckptName, "--epochs", "100", "--batch_size", "8",
            "--num_workers", "$numWorkers", "--eval_every", "6", "--cfg_scale", "6.0",
            "--temperature", "1.0", "--resume", "--accept_legacy_resume", "--extend_schedule"
        )
}

$codes = @()
$codes += Invoke-Extend100 "extend100_transformer_fsq" "transformer" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/outputs/gates/transformer_34m_overfit.json" `
    "generator_transformer_bs8_e100.pt" 4
$codes += Invoke-Extend100 "extend100_mamba_fsq" "mamba" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/outputs/gates/mamba_34m_overfit.json" `
    "generator_mamba_bs8_e100.pt" 2

if (-not (Write-DoneMarker -Path "artifacts/outputs/EXTEND_TWIN_100EP_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
