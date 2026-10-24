# powershell -ExecutionPolicy Bypass -File scripts/training/tokenizer_sweep/rvq_sweep.ps1
$ErrorActionPreference = "Stop"
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) "TrainingRun.psm1") -Force
$null = Enter-RepoRoot -StartDirectory $PSScriptRoot
$done = Invoke-TokenizerSweep -DoneMarker "artifacts/outputs/RVQ_SWEEP_DONE.txt" -Variants @(
    @{ cfg = "configs/tokenizer/rvq_l6_512.yaml"; mech = "rvq"; name = "rvq_l6_512.pt" },
    @{ cfg = "configs/tokenizer/rvq_l4_512.yaml"; mech = "rvq"; name = "rvq_l4_512.pt" },
    @{ cfg = "configs/tokenizer/rvq_l8_512.yaml"; mech = "rvq"; name = "rvq_l8_512.pt" },
    @{ cfg = "configs/tokenizer/rvq_l6_1024.yaml"; mech = "rvq"; name = "rvq_l6_1024.pt" }
)
if (-not $done) {
    exit 1
}
