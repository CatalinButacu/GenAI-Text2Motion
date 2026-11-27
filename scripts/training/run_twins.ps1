# Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File scripts/training/run_twins.ps1 -Config configs/generator/X.yaml -RunName causal -TokenizerCkpt runs/checkpoints/tokenizer/T.pt" -WindowStyle Hidden
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][ValidatePattern("^[A-Za-z0-9_]+$")][string]$RunName,
    [Parameter(Mandatory = $true)][string]$TokenizerCkpt,
    [int]$Epochs = 60,
    [int]$PretrainEpochs = 0,
    [double]$MaxHours = 72,
    [int]$StaleMinutes = 90
)
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
foreach ($path in @($Config, $TokenizerCkpt)) {
    if (-not (Test-Path $path)) { throw "missing input: $path" }
}
$root = "runs/outputs/twins/$RunName"
New-Item -ItemType Directory -Force $root | Out-Null
$arguments = @(
    "-m", "text2motion.app.cli", "run", "--config", $Config, "--workflow", "EVALUATE",
    "--root", $root, "--run_name", $RunName, "--tokenizer_ckpt", $TokenizerCkpt,
    "--generator_epochs", "$Epochs"
)
if ($PretrainEpochs -gt 0) {
    $arguments += @("--pretrain_epochs", "$PretrainEpochs")
}
Write-Host "=== twins $RunName start $(Get-Date -Format u); finished steps are skipped, a stopped twin resumes ==="
$code = Invoke-TrainingStep -Tag "twins_$RunName" -LogFile "$root/run.log" `
    -MaxHours $MaxHours -StaleMinutes $StaleMinutes -Arguments $arguments
if (-not (Write-DoneMarker -Path "$root/DONE.txt" -ExitCodes @($code))) {
    exit 1
}
