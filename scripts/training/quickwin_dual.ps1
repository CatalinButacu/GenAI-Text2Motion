# Quick-win batch for dual-person Inter-X: combines four anti-overfit changes
# vs the 100-epoch run that hit val_ce=5.37 on 238 samples.
#
# Changes from the prior run:
#   A. Constant LR (3e-4) -- stops wasting epochs on OneCycleLR warmup.
#      Train_motion_ssm.py uses OneCycleLR with max_lr from --lr; setting --lr
#      to 3e-4 means peak is reached around epoch 9 (30% of 30) instead of
#      epoch 30 (30% of 100). Net effect: much more useful training time.
#   B. Smaller model -- d_model=256, n_layers=4 (~1.4M params) vs prior 3M.
#      Less capacity means less memorization of the small Inter-X subset.
#   D. Heavier weight decay -- 0.01 -> 0.1 directly fights overfit.
#   E. More samples -- 300 -> 800 (matches RAM budget; ~6x train data).
#
# Wall time estimate:
#   train split ~ 640 pairs / 8 batch = 80 batches/epoch
#   each batch ~3.0 sec (smaller model = ~20% faster) -> 4 min/epoch
#   30 epochs -> ~2 hours, early-stop=10 may cap it earlier
#
# Expected: val_ce ~ 4.5-4.8, top1 ~ 12-15% on TEST set.
# If this beats 5.37 / 0.072 on bigger data with less overfit, validation passes
# and we move to the spatial-coherence novelty experiment next.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "quickwin_dual_$ts.out"
$err = "quickwin_dual_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "interx_paired",
    "--dual-person",
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

Write-Host "--- quickwin dual-person batch ---"
Write-Host "  d_model=256  n_layers=4  batch=8"
Write-Host "  max_samples=800  epochs=30  lr=3e-4  weight_decay=0.1"
Write-Host "  estimated wall time ~ 2 hours (early-stop=10 may cap it)"
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
Write-Host "Check status:  powershell -ExecutionPolicy Bypass -File scripts\training\check_local_train.ps1"
Write-Host "Kill:          Stop-Process -Id (Get-Content local_train.pid)"
