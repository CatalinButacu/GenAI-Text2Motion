# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/fsq_sweep.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/FSQ_SWEEP_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/tokenizer_isovocab.yaml"; mech = "fsq"; name = "fsq_g6_v512.pt" },
    @{ cfg = "configs/tokenizer/fsq_g6_v1024.yaml"; mech = "fsq"; name = "fsq_g6_v1024.pt" },
    @{ cfg = "configs/tokenizer/fsq_g4_v512.yaml"; mech = "fsq"; name = "fsq_g4_v512.pt" },
    @{ cfg = "configs/tokenizer/fsq_g8_v512.yaml"; mech = "fsq"; name = "fsq_g8_v512.pt" },
    @{ cfg = "configs/tokenizer/tok_g6_v1000.yaml"; mech = "fsq"; name = "fsq_g6_v1000.pt" }
)
if (-not $done) {
    exit 1
}
