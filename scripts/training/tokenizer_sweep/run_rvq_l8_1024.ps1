# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/run_rvq_l8_1024.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/RVQ_L8_1024_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/rvq_l8_1024.yaml"; mech = "rvq"; name = "rvq_l8_1024.pt" }
)
if (-not $done) {
    exit 1
}
