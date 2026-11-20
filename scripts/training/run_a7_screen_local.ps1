# powershell -ExecutionPolicy Bypass -File scripts/training/run_a7_screen_local.ps1
param([double]$BudgetHoursPerRun = 1.5)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
$backbones = @("transformer", "mamba")
Write-Host "=== A7 G2 screen start $(Get-Date -Format u) ==="

function Invoke-ScreenRun([string]$Tag, [string]$Config, [string]$Gate, [string]$Backbone, [int]$StopEpoch) {
    $name = "generator_${Backbone}_w320_$Tag.pt"
    if (Test-Path "runs/checkpoints/generator/generator_${Backbone}_w320_${Tag}_last.pt") {
        throw "a run with this name exists; a screen run is never resumed: $name"
    }
    return Invoke-TrainingStep -Tag "${Tag}_$Backbone" -LogFile "runs/outputs/${Tag}_${Backbone}_w320.log" `
        -MaxHours $BudgetHoursPerRun -StaleMinutes 45 -Arguments @(
            "-m", "text2motion.app.cli", "train-generator", "--config", $Config,
            "--backbone", $Backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $Gate,
            "--ckpt_name", $name, "--epochs", "60", "--stop_epoch", "$StopEpoch", "--val_ce_patience", "6",
            "--batch_size", "8", "--num_workers", "0", "--eval_every", "2",
            "--cfg_scale", "6.0", "--temperature", "1.0"
        )
}

$codes = @()
foreach ($backbone in $backbones) {
    $code = Invoke-ScreenRun -Tag "a7_replay" -Config "configs/generator/size_sweep/width_320.yaml" `
        -Gate "runs/outputs/gates/size_sweep_${backbone}_w320_overfit.json" -Backbone $backbone -StopEpoch 1
    $codes += $code
    if ($code -ne 0) { Write-Host "STOP: replay $backbone code $code"; exit 1 }
}
& .\.venv\Scripts\python.exe -u scripts/qa/a7_screen_verdict.py replay
if ($LASTEXITCODE -ne 0) { Write-Host "STOP: the replay does not match P1; the control needs a rerun"; exit 1 }

foreach ($arm in @("a7_step50", "a7_clipfrozen")) {
    $config = "configs/generator/a7/width_320_$($arm.Substring(3)).yaml"
    foreach ($backbone in $backbones) {
        $gate = "runs/outputs/gates/${arm}_${backbone}_w320_overfit.json"
        $code = Invoke-TrainingStep -Tag "gate_${arm}_$backbone" -LogFile "runs/outputs/${arm}_gate_${backbone}_w320.log" `
            -MaxHours 0.5 -Arguments @(
                "-m", "text2motion.app.cli", "sanity-overfit", "--config", $config,
                "--backbone", $backbone, "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate
            )
        $codes += $code
        if ($code -ne 0) { Write-Host "STOP: gate $arm $backbone code $code"; exit 1 }
        $code = Invoke-ScreenRun -Tag $arm -Config $config -Gate $gate -Backbone $backbone -StopEpoch 6
        $codes += $code
        if ($code -ne 0) { Write-Host "STOP: $arm $backbone code $code"; exit 1 }
    }
}

& .\.venv\Scripts\python.exe -u scripts/qa/a7_screen_verdict.py verdict --out docs/paper/pics/a7_screen_validation.csv
$codes += $LASTEXITCODE
if (-not (Write-DoneMarker -Path "runs/outputs/A7_G2_DONE.txt" -ExitCodes $codes)) {
    exit 1
}
