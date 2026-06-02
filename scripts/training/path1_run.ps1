# Path 1 real run: V3 architecture + all motion-coherence losses + bigger model.
#
# Stacks the three SSM-side fixes the previous V3 run lacked:
#   - velocity-match (forces frame-to-frame continuity, fixes "alzheimer" jitter)
#   - per-actor L4 pose decode (geometric supervision in both streams)
#   - NaN-safe inter-actor separation (forces actors not to overlap)
#
# Model bumped back to d_model=384 / n_layers=6 (the original "V1" size that
# proved trainable on this laptop). Cross-actor attention adds ~1 GB to VRAM
# at batch=8 -- smoke confirmed 3.6/4.0 GB fits without OOM.
#
# Wall-time estimate at 6.26 sec/step:
#   ~80 batches/epoch (800 max-samples -> 640 train pairs / 8) -> ~8.3 min/epoch
#   30 epochs -> ~4.2 hours

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "path1_run_$ts.out"
$err = "path1_run_$ts.err"

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
    "--epochs", "30",
    "--num-workers", "0",
    "--max-samples", "800",
    "--weight-decay", "0.1",
    "--separation-loss-weight", "1.0",
    "--pose-loss-weight", "0.1",
    "--velocity-loss-weight", "0.5",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- Path 1 real run (bigger model + all coherence losses) ---"
Write-Host "  d_model=384  n_layers=6  K=2  cross-attn=ON"
Write-Host "  pose_l4=0.1  velocity=0.5  separation=1.0"
Write-Host "  max_samples=800  epochs=30  batch=8  weight_decay=0.1"
Write-Host "  estimated wall time ~ 4.2 hours"
Write-Host ""
Write-Host "  stdout -> $out"
Write-Host "  stderr -> $err"

$p = Start-Process -FilePath python `
    -ArgumentList $argList `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru

$p.Id | Out-File -Encoding utf8 local_train.pid

Write-Host ""
Write-Host "Launched PID=$($p.Id)"
