# powershell -ExecutionPolicy Bypass -File scripts/training/historical/run_a7_factor_split_local.ps1
param([double]$BudgetHoursPerRun = 3.5)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "../TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
Write-Host "=== A7 G7 factor split start $(Get-Date -Format u) ==="

$codes = @()
foreach ($arm in @("pkeep50", "wholestep")) {
    $config = "configs/generator/a7/width_320_$arm.yaml"
    foreach ($backbone in @("transformer", "mamba")) {
        $tag = "a7_g7_$arm"
        $name = "generator_${backbone}_w320_$tag.pt"
        if (Test-Path "runs/checkpoints/generator/generator_${backbone}_w320_${tag}_last.pt") {
            throw "a run with this name exists; a screen run is never resumed: $name"
        }
        $gate = "runs/outputs/gates/${tag}_${backbone}_w320_overfit.json"
        $code = Invoke-TrainingStep -Tag "gate_${tag}_$backbone" -LogFile "runs/outputs/${tag}_gate_${backbone}_w320.log" `
            -MaxHours 0.5 -Arguments @(
                "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config,
                "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
            )
        $codes += $code
        if ($code -ne 0) { Write-Host "STOP: gate $tag $backbone code $code"; exit 1 }
        $code = Invoke-TrainingStep -Tag "${tag}_$backbone" -LogFile "runs/outputs/${tag}_${backbone}_w320.log" `
            -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments @(
                "-m", "text2motion.app.cli", "train-generator", "--config", $config,
                "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
                "--ckpt_name", $name, "--epochs", "60", "--stop_epoch", "12", "--val_ce_patience", "6",
                "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
                "--cfg_scale", "6.0", "--temperature", "1.0"
            )
        $codes += $code
        if ($code -ne 0) { Write-Host "STOP: $tag $backbone code $code"; exit 1 }
    }
}

if (-not (Write-DoneMarker -Path "runs/outputs/A7_G7_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
