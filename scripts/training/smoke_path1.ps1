# Smoke test for Path 1: bigger model (d_model=384, n_layers=6) + all
# motion-coherence losses stacked together on V3 architecture.
#
# Validates in ~10 min:
#   - the new --velocity-loss-weight CLI plumbs through
#   - per-actor L4 fires in K=2 mode (postfix shows pose_l4=)
#   - velocity match fires (postfix shows vel=)
#   - separation (NaN-safe) fires (postfix shows sep=)
#   - no NaN / no OOM on d_model=384, n_layers=6 with cross-attention

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_path1_$ts.out"
$err = "smoke_path1_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "interx_paired",
    "--num-actors", "2",
    "--cross-actor-attention",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "384", "--d-state", "64", "--n-layers", "6",
    "--max-motion-length", "200",
    "--batch-size", "8",
    "--lr", "3e-4",
    "--epochs", "2",
    "--num-workers", "0",
    "--max-samples", "300",
    "--separation-loss-weight", "1.0",
    "--pose-loss-weight", "0.1",
    "--velocity-loss-weight", "0.5",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- smoke Path 1 (bigger model + all coherence losses) ---"
Write-Host "  d_model=384  n_layers=6  K=2  cross-attn=ON"
Write-Host "  pose_l4=0.1  velocity=0.5  separation=1.0"
Write-Host "  300 samples / 2 epochs ~ 10 min"
Write-Host ""
Write-Host "  stdout -> $out"
Write-Host "  stderr -> $err"
Write-Host ""

$p = Start-Process -FilePath python `
    -ArgumentList $argList `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru

$p.Id | Out-File -Encoding utf8 smoke_path1.pid

Write-Host "Launched PID=$($p.Id)"
