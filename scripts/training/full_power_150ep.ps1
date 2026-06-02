# Full-power Inter-X-only training, 150 epochs.
#
# Sized to the VRAM ceiling of the laptop (4 GB):
#   d_model=256, n_layers=4 with cross-actor attention -> ~2.7 GB at batch=8.
#   d=384/n=6 OOM'd with the same data load.
#
# Data: --max-samples 1500 -> ~1200 paired interactions after quality filter
#       -> 960 train pairs (vs the prior 800-sample cap that gave 494 train).
#       ~2x the prior training set.
#
# Schedule: OneCycleLR with --lr 1e-4 peak (not 3e-4) because the peak is at
# epoch ~45 of a 150-epoch run, and at that lr scale + extended duration we
# want a gentler peak to avoid the gradient-overflow class of NaN failures.
#
# Losses (no L4, it caused two consecutive NaN crashes under AMP fp16):
#   separation=1.0  -- inter-actor distance matches GT
#   velocity=0.5    -- frame-to-frame continuity (fixes the "alzheimer" look)
#
# Wall-time estimate at 3.3 sec/step:
#   ~120 batches/epoch -> ~7 min/epoch -> 150 epochs ~ 17 hours.
#   Early-stop=10 may cut it earlier if val_ce plateaus.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "fullpower_$ts.out"
$err = "fullpower_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "interx_paired",
    "--num-actors", "2",
    "--cross-actor-attention",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "256", "--d-state", "64", "--n-layers", "4",
    "--max-motion-length", "200",
    "--batch-size", "8",
    "--lr", "1e-4",
    "--epochs", "150",
    "--num-workers", "0",
    "--max-samples", "1500",
    "--weight-decay", "0.1",
    "--separation-loss-weight", "1.0",
    "--velocity-loss-weight", "0.5",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- Full-power 150-epoch Inter-X-only ---"
Write-Host "  d_model=256  n_layers=4  K=2  cross-attn=ON  (proven to fit VRAM)"
Write-Host "  max_samples=1500 (~960 train pairs)  batch=8"
Write-Host "  separation=1.0  velocity=0.5  NO L4  weight_decay=0.1"
Write-Host "  lr=1e-4 peak  150 epochs (early-stop=10)"
Write-Host "  RVQ: original best_model.pt (val_recon=0.0946)"
Write-Host "  estimated wall time ~ 17 hours"

$p = Start-Process -FilePath python `
    -ArgumentList $argList `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru

$p.Id | Out-File -Encoding utf8 local_train.pid

Write-Host ""
Write-Host "Launched PID=$($p.Id)  stderr=$err"
