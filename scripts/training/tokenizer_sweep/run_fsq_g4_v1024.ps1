# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/run_fsq_g4_v1024.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/FSQ_G4_V1024_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/fsq_g4_v1024.yaml"; mech = "fsq"; name = "fsq_g4_v1024.pt" }
)
if (-not $done) {
    exit 1
}
