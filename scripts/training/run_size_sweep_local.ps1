# powershell -ExecutionPolicy Bypass -File scripts/training/run_size_sweep_local.ps1
param([double]$BudgetHoursPerRun = 6, [int]$StaleMinutes = 45, [int]$Epochs = 20, [int]$ValCePatience = 3)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
Write-Host "=== size sweep start $(Get-Date -Format u) ==="

function Invoke-SweepGate([string]$width, [string]$backbone) {
    $gate = "artifacts/outputs/gates/size_sweep_${backbone}_w${width}_overfit.json"
    if (Test-Path $gate) {
        Write-Host "SKIP gate $backbone w$width ($gate exists)"
        return 0
    }
    return Invoke-TrainingStep -Tag "gate_${backbone}_w$width" -LogFile "artifacts/outputs/size_sweep_gate_${backbone}_w$width.log" `
        -MaxHours 1 -Arguments @(
            "-m", "text2motion.app.cli", "sanity-overfit", "--config", "configs/generator/size_sweep/width_$width.yaml",
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
        )
}

function Invoke-SweepRun([string]$width, [string]$backbone) {
    $name = "generator_${backbone}_w${width}_sweep.pt"
    $resume = "artifacts/checkpoints/generator/$([IO.Path]::GetFileNameWithoutExtension($name))_last.pt"
    if (Test-Path $resume) {
        Write-Host "SKIP run $backbone w$width ($resume exists; a sweep run is never resumed)"
        return 0
    }
    return Invoke-TrainingStep -Tag "sweep_${backbone}_w$width" -LogFile "artifacts/outputs/size_sweep_${backbone}_w$width.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes $StaleMinutes -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", "configs/generator/size_sweep/width_$width.yaml",
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer,
            "--overfit_gate", "artifacts/outputs/gates/size_sweep_${backbone}_w${width}_overfit.json",
            "--ckpt_name", $name, "--epochs", "$Epochs", "--val_ce_patience", "$ValCePatience",
            "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
            "--cfg_scale", "6.0", "--temperature", "1.0"
        )
}

$codes = @()
foreach ($width in @("256", "320", "384", "448", "512")) {
    foreach ($backbone in @("transformer", "mamba")) {
        $code = Invoke-SweepGate $width $backbone
        $codes += $code
        if ($code -ne 0) {
            Write-Host "STOP: gate $backbone w$width failed with code $code"
            exit 1
        }
        $code = Invoke-SweepRun $width $backbone
        $codes += $code
        if ($code -ne 0) {
            Write-Host "STOP: run $backbone w$width failed with code $code"
            exit 1
        }
    }
}

if (-not (Write-DoneMarker -Path "artifacts/outputs/SIZE_SWEEP_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
