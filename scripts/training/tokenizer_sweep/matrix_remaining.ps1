# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/matrix_remaining.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/MATRIX_REMAINING_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/rvq_l8_1024.yaml"; mech = "rvq"; name = "rvq_l8_1024.pt" },
    @{ cfg = "configs/tokenizer/fsq_g8_v1024.yaml"; mech = "fsq"; name = "fsq_g8_v1024.pt" },
    @{ cfg = "configs/tokenizer/tok_g8_v1000.yaml"; mech = "fsq"; name = "fsq_g8_v1000.pt" },
    @{ cfg = "configs/tokenizer/fsq_g4_v1024.yaml"; mech = "fsq"; name = "fsq_g4_v1024.pt" },
    @{ cfg = "configs/tokenizer/tok_g4_v1000.yaml"; mech = "fsq"; name = "fsq_g4_v1000.pt" }
)
if (-not $done) {
    exit 1
}
