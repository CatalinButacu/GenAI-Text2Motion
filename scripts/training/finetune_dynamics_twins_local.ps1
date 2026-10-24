# powershell -ExecutionPolicy Bypass -File scripts/training/finetune_dynamics_twins_local.ps1 -Epochs 72 -SeedFrom e60
param(
    [double]$BudgetHoursPerRun = 48,
    [int]$StaleMinutes = 90,
    [int]$Epochs = 72,
    [string]$SeedFrom = "e60",
    [switch]$ContinueSelection
)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$config = "configs/generator/gen_pilot_fsq8x1024_dynamics.yaml"
$tokenizer = "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
$checkpoints = "artifacts/checkpoints/generator"
Write-Host "=== dynamics fine-tune, both twins, from $SeedFrom -> epoch $Epochs, continue selection $ContinueSelection, start $(Get-Date -Format u) ==="

function Invoke-FineTune([string]$backbone, [int]$numWorkers) {
    $label = "finetune_dynamics_${backbone}_e$Epochs"
    $name = "generator_${backbone}_bs8_dyn_e$Epochs.pt"
    $resume = Join-Path $checkpoints ([IO.Path]::GetFileNameWithoutExtension($name) + "_last.pt")
    if (Test-RunComplete -ResumePath $resume -Epochs $Epochs) {
        Write-Host "SKIP $label (all $Epochs epochs are in $resume)"
        return 0
    }
    $seeded = $false
    if (-not (Test-Path $resume)) {
        $seed = Join-Path $checkpoints "generator_${backbone}_bs8_${SeedFrom}_last.pt"
        if (-not (Test-Path $seed)) {
            throw "seed state $seed not found"
        }
        Copy-Item $seed $resume
        $seeded = $true
        Write-Host "seeded $resume from $seed"
    }
    $selection = if ($seeded -and -not $ContinueSelection) { "--fresh_selection" } else { "" }
    return Invoke-TrainingStep -Tag $label -LogFile "artifacts/outputs/$label.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes $StaleMinutes -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", $config,
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer,
            "--overfit_gate", "artifacts/outputs/gates/${backbone}_34m_dynamics_overfit.json",
            "--ckpt_name", $name, "--epochs", "$Epochs", "--batch_size", "8",
            "--num_workers", "$numWorkers", "--eval_every", "6", "--cfg_scale", "6.0",
            "--temperature", "1.0", "--resume", "--accept_legacy_resume", "--extend_schedule",
            $selection
        )
}

$codes = @()
$codes += Invoke-FineTune "transformer" 4
$codes += Invoke-FineTune "mamba" 2

if (-not (Write-DoneMarker -Path "artifacts/outputs/FINETUNE_DYNAMICS_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
