# powershell -ExecutionPolicy Bypass -File scripts/training/historical/run_a8_f0_step50_local.ps1
param([double]$BudgetHoursPerRun = 11)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "../TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
$config = "configs/generator/a7/width_320_step50.yaml"
Write-Host "=== A8 M5: prior then full fine-tune with the step50 recipe, start $(Get-Date -Format u) ==="

$codes = @()
foreach ($backbone in @("transformer", "mamba")) {
    $gate = "runs/outputs/gates/a8_f0_step50_${backbone}_w320_overfit.json"
    $init = @("--init_ckpt", "runs/checkpoints/generator/generator_${backbone}_w320_a8_prior.pt")
    $code = Invoke-TrainingStep -Tag "gate_a8_f0_step50_$backbone" -LogFile "runs/outputs/a8_f0_step50_gate_${backbone}_w320.log" `
        -MaxHours 0.5 -Arguments (@(
            "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config, "--backbone", $backbone,
            "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate) + $init)
    if ($code -ne 0) { Write-Host "STOP: gate $backbone code $code"; exit 1 }
    $resume = @()
    if (Test-Path "runs/checkpoints/generator/generator_${backbone}_w320_a8_f0_step50_last.pt") { $resume = @("--resume") }
    $code = Invoke-TrainingStep -Tag "a8_f0_step50_$backbone" -LogFile "runs/outputs/a8_f0_step50_${backbone}_w320.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments (@(
            "-m", "text2motion.app.cli", "train-generator", "--config", $config,
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate,
            "--ckpt_name", "generator_${backbone}_w320_a8_f0_step50.pt",
            "--epochs", "60", "--val_ce_patience", "6",
            "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
            "--cfg_scale", "6.0", "--temperature", "1.0") + $init + $resume)
    $codes += $code
    if ($code -ne 0) { Write-Host "STOP: M5 $backbone code $code"; exit 1 }
}

if (-not (Write-DoneMarker -Path "runs/outputs/A8_F0_STEP50_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
