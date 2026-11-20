# powershell -ExecutionPolicy Bypass -File scripts/training/run_a15_full_motionclass_local.ps1
param([double]$BudgetHoursPerRun = 11)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
$config = "configs/generator/a15/width_320_step50_motionclass.yaml"
Write-Host "=== A15 V11 full motion-class run start $(Get-Date -Format u) ==="

$codes = @()
foreach ($backbone in @("transformer", "mamba")) {
    $name = "generator_${backbone}_w320_a15_motionclass_full.pt"
    if (Test-Path "runs/checkpoints/generator/generator_${backbone}_w320_a15_motionclass_full_last.pt") {
        throw "a run with this name exists; resume it by hand: $name"
    }
    $gate = "runs/outputs/gates/a15_motionclass_full_${backbone}_w320_overfit.json"
    $code = Invoke-TrainingStep -Tag "gate_a15_motionclass_full_$backbone" -LogFile "runs/outputs/a15_motionclass_full_gate_${backbone}_w320.log" `
        -MaxHours 0.5 -Arguments @(
            "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config,
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
        )
    $codes += $code
    if ($code -ne 0) { Write-Host "STOP: gate $backbone code $code"; exit 1 }
    $code = Invoke-TrainingStep -Tag "a15_motionclass_full_$backbone" -LogFile "runs/outputs/a15_motionclass_full_${backbone}_w320.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", $config,
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
            "--ckpt_name", $name, "--epochs", "60", "--val_ce_patience", "6",
            "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
            "--cfg_scale", "6.0", "--temperature", "1.0"
        )
    $codes += $code
    if ($code -ne 0) { Write-Host "STOP: full $backbone code $code"; exit 1 }
}

if (-not (Write-DoneMarker -Path "runs/outputs/A15_V11_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
