# powershell -ExecutionPolicy Bypass -File scripts/training/run_a15_v9_kill_test_local.ps1
param([double]$BudgetHoursPerRun = 1.5)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
$backbones = @("transformer", "mamba")
$arms = [ordered]@{
    "a15_replacement" = "configs/generator/a15/width_320_step50_replacement.yaml"
}
Write-Host "=== A15 V9 kill test start $(Get-Date -Format u) ==="

$codes = @()
foreach ($arm in $arms.Keys) {
    $config = $arms[$arm]
    foreach ($backbone in $backbones) {
        $name = "generator_${backbone}_w320_$arm.pt"
        if (Test-Path "runs/checkpoints/generator/generator_${backbone}_w320_${arm}_last.pt") {
            throw "a run with this name exists; a kill-test run is never resumed: $name"
        }
        $gate = "runs/outputs/gates/${arm}_${backbone}_w320_overfit.json"
        $code = Invoke-TrainingStep -Tag "gate_${arm}_$backbone" -LogFile "runs/outputs/${arm}_gate_${backbone}_w320.log" `
            -MaxHours 0.5 -Arguments @(
                "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config,
                "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
            )
        $codes += $code
        if ($code -ne 0) { Write-Host "STOP: gate $arm $backbone code $code"; exit 1 }
        $code = Invoke-TrainingStep -Tag "${arm}_$backbone" -LogFile "runs/outputs/${arm}_${backbone}_w320.log" `
            -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments @(
                "-m", "text2motion.app.cli", "train-generator", "--config", $config,
                "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
                "--ckpt_name", $name, "--epochs", "60", "--stop_epoch", "6", "--val_ce_patience", "6",
                "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
                "--cfg_scale", "6.0", "--temperature", "1.0"
            )
        $codes += $code
        if ($code -ne 0) { Write-Host "STOP: $arm $backbone code $code"; exit 1 }
    }
}

if (-not (Write-DoneMarker -Path "runs/outputs/A15_V9_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
