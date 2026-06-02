# Local smoke test for the MotionSSM training pipeline -- coexistence mode.
# Goal: prove the model is actually LEARNING (val_ce drops monotonically)
# WITHOUT hogging the GPU so you can keep working on the laptop.
#
# Sized to share a 4 GB RTX 3050 Laptop with Brave/VS Code/Teams etc.:
#   - d_model 384 / n_layers 6 (same architecture as the 2026-05-11 run that
#     reached val_ce=4.6038 -- proven to work)
#   - batch_size 8 (very small VRAM footprint, ~1.2 GB; leaves >2 GB for the
#     desktop apps, also creates idle gaps between steps so typing stays smooth)
#   - max_samples 500 (small subset so each epoch is ~5-9 min, total ~30-45 min)
#   - num_workers 0 (avoid Windows fork issues + CPU contention with your work)
#   - epochs 5, early-stop=10 in config.py
#
# Expected: val_ce starts ~6.9 (random over 1024 codes), should drop to
# 5.8-6.2 by epoch 5. Any drop > 0.3 nats proves the training code works.
# If it stalls or NaNs, that is a real bug to chase, not a compute problem.
#
# Output:
#   stdout/stderr -> smoke_train_<ts>.{out,err} in project root
#   trainer's own log -> checkpoints/motion_ssm/<timestamp>/training.log

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_train_$ts.out"
$err = "smoke_train_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "amass", "humanml3d",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "384", "--d-state", "64", "--n-layers", "6",
    "--max-motion-length", "200",
    "--batch-size", "8",
    "--lr", "3e-4",
    "--epochs", "5",
    "--num-workers", "0",
    "--max-samples", "500",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- smoke train config (coexistence mode) ---"
Write-Host ("  d_model=384  n_layers=6  batch=8  samples=500  epochs=5")
Write-Host ("  lr=3e-4 (peak), OneCycle warmup -> peak -> anneal")
Write-Host ("  est. wall time: ~30-45 min (laptop stays usable)")
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

$p.Id | Out-File -Encoding utf8 smoke_train.pid
Write-Host "Launched PID=$($p.Id)"
Write-Host ""
Write-Host "Tail metrics with:"
Write-Host "  Get-Content $err -Wait -Tail 20"
Write-Host "Kill with:"
Write-Host "  Stop-Process -Id (Get-Content smoke_train.pid)"
