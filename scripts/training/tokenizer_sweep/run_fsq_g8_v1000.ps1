# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/run_fsq_g8_v1000.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/FSQ_G8_V1000_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/tok_g8_v1000.yaml"; mech = "fsq"; name = "fsq_g8_v1000.pt" }
)
if (-not $done) {
    exit 1
}
