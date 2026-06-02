# Real training run for the V3 architecture (numActors=2 + cross-actor attention)
# on the Inter-X paired dataset. This is the thesis-grade run that produces the
# headline numbers and visualization checkpoints.
#
# Combines V3 architectural novelty with the quickwin anti-overfit knobs:
#   - K=2 streams + actor embeddings + cross-actor attention each layer
#   - Smaller d_model=256, n_layers=4 (anti-overfit for the small dataset)
#   - 800 samples (max RAM budget without rebuild)
#   - 30 epochs at lr=3e-4 peak (early-stop=10 will likely cap it)
#   - weight_decay=0.1 (heavier regularisation than default 0.01)
#   - batch=8 + grad-ckpt
#
# Wall-time estimate at 4.0 sec/step:
#   train ~ 640 pairs / 8 = 80 batches/epoch -> ~5.3 min/epoch
#   30 epochs = ~2.7 hours (early-stop will likely cut it shorter)
#
# Expected outcome relative to the prior dual-person run (val_ce=5.37 on 238
# samples, no actor embeddings, no cross-attention):
#   - val_ce should land in the 4.5-4.9 range (better but still small data)
#   - top1 on TEST should reach 0.10-0.15
#   - Generated videos should show inter-actor reactivity (one arm raises,
#     the other moves toward it), unlike V1 which had independent streams

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "v3_run_$ts.out"
$err = "v3_run_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "interx_paired",
    "--num-actors", "2",
    "--cross-actor-attention",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "256", "--d-state", "64", "--n-layers", "4",
    "--max-motion-length", "200",
    "--batch-size", "8",
    "--lr", "3e-4",
    "--epochs", "30",
    "--num-workers", "0",
    "--max-samples", "800",
    "--weight-decay", "0.1",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- V3 real run (cross-actor attention, K=2) ---"
Write-Host "  d_model=256  n_layers=4  K=2  cross-attn=ON"
Write-Host "  max_samples=800  epochs=30  batch=8  lr=3e-4  weight_decay=0.1"
Write-Host "  estimated wall time ~ 2.7 hours (early-stop may cap shorter)"
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

$p.Id | Out-File -Encoding utf8 local_train.pid

Write-Host "Launched PID=$($p.Id)"
Write-Host ""
Write-Host "Check status: powershell -ExecutionPolicy Bypass -File scripts\training\check_local_train.ps1"
Write-Host "Kill:         Stop-Process -Id (Get-Content local_train.pid)"
