# powershell -ExecutionPolicy Bypass -File scripts/training/historical/run_a8_kill_tests_local.ps1
param([int]$WaitForPid = 0, [string]$WaitForMarker = "", [switch]$SkipPrior, [string[]]$Skip = @())
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "../TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$tokenizer = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
$backbones = @("transformer", "mamba")
$recipe = @("--val_ce_patience", "6", "--batch_size", "8", "--num_workers", "0", "--cfg_scale", "6.0", "--temperature", "1.0")

if ($WaitForPid -gt 0) {
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 300 }
    if ($WaitForMarker -and -not (Test-Path $WaitForMarker)) { Write-Host "STOP: pid $WaitForPid ended without $WaitForMarker"; exit 1 }
}
Write-Host "=== A8 kill tests start $(Get-Date -Format u) ==="

function Invoke-Gate([string]$Name, [string]$Config, [string]$Backbone, [string[]]$Extra) {
    $gate = "runs/outputs/gates/${Name}_${Backbone}_w320_overfit.json"
    $code = Invoke-TrainingStep -Tag "gate_${Name}_$Backbone" -LogFile "runs/outputs/${Name}_gate_${Backbone}_w320.log" -MaxHours 0.5 -Arguments (@(
        "-m", "text2motion.app.cli", "sanity-overfit", "--config", $Config, "--backbone", $Backbone,
        "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $gate) + $Extra)
    if ($code -ne 0) { Write-Host "STOP: gate $Name $Backbone code $code"; exit 1 }
    return $gate
}

function Invoke-Train([string]$Name, [string]$Config, [string]$Backbone, [string]$Gate, [double]$Hours, [string[]]$Extra) {
    $code = Invoke-TrainingStep -Tag "${Name}_$Backbone" -LogFile "runs/outputs/${Name}_${Backbone}_w320.log" -MaxHours $Hours -StaleMinutes 45 -Arguments (@(
        "-m", "text2motion.app.cli", "train-generator", "--config", $Config, "--backbone", $Backbone,
        "--tokenizer_ckpt", $tokenizer, "--overfit_gate", $Gate, "--ckpt_name", "generator_${Backbone}_w320_$Name.pt") + $recipe + $Extra)
    if ($code -ne 0) { Write-Host "STOP: $Name $Backbone code $code"; exit 1 }
}

$prior = "configs/generator/a8/width_320_prior.yaml"
$priorFlags = @("--epochs", "20", "--select_on_val_ce", "--eval_every", "1000")
if (-not $SkipPrior) {
    foreach ($backbone in $backbones) {
        $gate = Invoke-Gate -Name "a8_prior" -Config $prior -Backbone $backbone -Extra @()
        Invoke-Train -Name "a8_prior" -Config $prior -Backbone $backbone -Gate $gate -Hours 1 -Extra ($priorFlags + @("--stop_epoch", "2"))
    }
    & .\.venv\Scripts\python.exe -u scripts/qa/a8_kill_tests.py kt1
    if ($LASTEXITCODE -ne 0) { Write-Host "STOP: KT1 failed; the prior does not learn"; exit 1 }

    foreach ($backbone in $backbones) {
        $gate = "runs/outputs/gates/a8_prior_${backbone}_w320_overfit.json"
        Invoke-Train -Name "a8_prior" -Config $prior -Backbone $backbone -Gate $gate -Hours 5 -Extra ($priorFlags + @("--resume"))
    }
}

foreach ($arm in @("f0", "f1", "f2")) {
    $config = "configs/generator/a8/width_320_$arm.yaml"
    foreach ($backbone in $backbones) {
        if ($Skip -contains "${arm}_$backbone") { continue }
        $init = @("--init_ckpt", "runs/checkpoints/generator/generator_${backbone}_w320_a8_prior.pt")
        $gate = Invoke-Gate -Name "a8_$arm" -Config $config -Backbone $backbone -Extra $init
        Invoke-Train -Name "a8_$arm" -Config $config -Backbone $backbone -Gate $gate -Hours 1 -Extra ($init + @("--epochs", "60", "--stop_epoch", "2", "--eval_every", "2"))
    }
}
& .\.venv\Scripts\python.exe -u scripts/qa/a8_kill_tests.py kt2
$null = Write-DoneMarker -Path "runs/outputs/A8_KT_DONE.txt" -ExitCodes @($LASTEXITCODE)
