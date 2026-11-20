# powershell -ExecutionPolicy Bypass -File scripts/training/run_a7_step50_extend_local.ps1
param([double]$BudgetHoursPerRun = 1.5)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
Write-Host "=== A7 step50 extension to epoch 12 start $(Get-Date -Format u) ==="

$codes = @()
foreach ($backbone in @("transformer", "mamba")) {
    $code = Invoke-TrainingStep -Tag "a7_step50_extend_$backbone" -LogFile "runs/outputs/a7_step50_extend_${backbone}_w320.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", "configs/generator/a7/width_320_step50.yaml",
            "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer,
            "--overfit_gate", "runs/outputs/gates/a7_step50_${backbone}_w320_overfit.json",
            "--ckpt_name", "generator_${backbone}_w320_a7_step50.pt", "--resume",
            "--epochs", "60", "--stop_epoch", "12", "--val_ce_patience", "6",
            "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
            "--cfg_scale", "6.0", "--temperature", "1.0"
        )
    $codes += $code
    if ($code -ne 0) { Write-Host "STOP: extension $backbone code $code"; exit 1 }
}

& .\.venv\Scripts\python.exe -u scripts/qa/a7_screen_verdict.py extend --out docs/paper/pics/a7_step50_extend_validation.csv
$codes += $LASTEXITCODE
if (-not (Write-DoneMarker -Path "runs/outputs/A7_STEP50_EXTEND_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
