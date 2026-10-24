# powershell -ExecutionPolicy Bypass -File scripts/training/run_matched_twin_60ep_local.ps1
param([double]$BudgetHoursPerRun = 72, [int]$StaleMinutes = 90)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
Write-Host "=== matched 60-epoch twins start $(Get-Date -Format u) ==="

function Invoke-Twin60([string]$label, [string]$backbone, [string]$config, [string]$tokenizer, [string]$initCkpt, [string]$gate, [string]$ckptName) {
    $resume = "artifacts/checkpoints/generator/$([IO.Path]::GetFileNameWithoutExtension($ckptName))_last.pt"
    if (Test-RunComplete -ResumePath $resume -Epochs 60) {
        Write-Host "SKIP $label (all 60 epochs are in $resume)"
        return 0
    }
    return Invoke-TrainingStep -Tag $label -LogFile "artifacts/outputs/$label.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes $StaleMinutes -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", $config,
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
            "--init_ckpt", $initCkpt, "--ckpt_name", $ckptName, "--epochs", "60",
            "--batch_size", "8", "--num_workers", "4", "--eval_every", "6",
            "--cfg_scale", "6.0", "--temperature", "1.0", (Get-ResumeFlag -ResumePath $resume)
        )
}

$codes = @()
$codes += Invoke-Twin60 "twin60_transformer_fsq" "transformer" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/checkpoints/generator/generator_transformer_pretrained.pt" `
    "artifacts/outputs/gates/transformer_34m_overfit.json" `
    "generator_transformer_bs8_e60.pt"
$codes += Invoke-Twin60 "twin60_transformer_rvq" "transformer" `
    "configs/generator/gen_pilot_rvq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/rvq_l8_1024.pt" `
    "artifacts/checkpoints/generator/generator_transformer_rvq_pretrained.pt" `
    "artifacts/outputs/gates/transformer_rvq_34m_overfit.json" `
    "generator_transformer_rvq_bs8_e60.pt"
$codes += Invoke-Twin60 "twin60_mamba_fsq" "mamba" `
    "configs/generator/gen_pilot_fsq8x1024.yaml" `
    "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt" `
    "artifacts/checkpoints/generator/generator_mamba_pretrained.pt" `
    "artifacts/outputs/gates/mamba_34m_overfit.json" `
    "generator_mamba_bs8_e60.pt"

if (-not (Write-DoneMarker -Path "artifacts/outputs/MATCHED_TWIN_60EP_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
