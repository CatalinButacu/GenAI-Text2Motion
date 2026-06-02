# Path C: resume the prior V3+separation checkpoint (val_ce=5.6564) and
# fine-tune for 15 epochs with the new velocity-match loss added.
#
# Reasoning:
#   - Prior V3-sep (anchored data + cross-attn + separation only) reached
#     val_ce=5.66 cleanly without NaN.
#   - Path 1 (bigger model + L4 + velocity + separation) blew up at peak LR
#     because L4's pow(4) overflows in fp16 AMP.
#   - C keeps everything that worked, adds JUST the velocity-match loss
#     (which is L2, safe in fp16), at a lower LR for fine-tuning.
#
# Architecture: identical to the source checkpoint (d_model=256, n_layers=4,
# K=2, cross-actor attention). --warm-start loads weights only, so the new
# --lr is honoured by a fresh OneCycleLR schedule over 15 epochs.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "pathc_run_$ts.out"
$err = "pathc_run_$ts.err"

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
    "--epochs", "15",
    "--num-workers", "0",
    "--max-samples", "800",
    "--weight-decay", "0.1",
    "--separation-loss-weight", "1.0",
    "--velocity-loss-weight", "0.5",
    "--resume", "checkpoints/motion_ssm/20260515-092902/best_model.pt",
    "--warm-start",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- Path C: resume V3-sep + add velocity loss ---"
Write-Host "  resume from: checkpoints/motion_ssm/20260515-092902/best_model.pt (val_ce=5.66)"
Write-Host "  d_model=256  n_layers=4  K=2  cross-attn=ON  (matches source)"
Write-Host "  separation=1.0  velocity=0.5  NO L4 (avoid NaN)"
Write-Host "  lr=1e-4 peak (fine-tune scale)  epochs=15  --warm-start"
Write-Host "  estimated wall time ~ 1.25 hours"
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
