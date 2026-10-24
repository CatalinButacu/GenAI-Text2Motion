# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/tokenizer_sweep.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/SWEEP_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/tok_g4_v1000.yaml"; mech = "fsq"; name = "tok_g4_v1000.pt" },
    @{ cfg = "configs/tokenizer/tok_g8_v1000.yaml"; mech = "fsq"; name = "tok_g8_v1000.pt" },
    @{ cfg = "configs/tokenizer/tok_g6_v2560.yaml"; mech = "fsq"; name = "tok_g6_v2560.pt" }
)
if (-not $done) {
    exit 1
}
