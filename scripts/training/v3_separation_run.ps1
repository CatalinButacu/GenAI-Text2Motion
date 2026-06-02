# Real V3 run with inter-actor distance loss.
#
# Anchored Inter-X paired data + cross-actor attention + separation loss.
# This is the "fix the overlap" run — same as v3_real_run.ps1 but with
# --separation-loss-weight 1.0 so the model gets explicit gradient toward
# matching GT inter-actor distance.
#
# Separation weight chosen from the smoke (5.0 was too dominant: sep took
# ~58% of total loss budget; 1.0 keeps it at ~10-15%, comparable to the
# length-loss contribution).

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "v3sep_run_$ts.out"
$err = "v3sep_run_$ts.err"

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
    "--separation-loss-weight", "1.0",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- V3 + separation run (anchored data, K=2, cross-attn, sep loss=1.0) ---"
Write-Host "  d_model=256  n_layers=4  K=2"
Write-Host "  max_samples=800  epochs=30  separation=1.0"
Write-Host "  estimated wall time ~ 2 hours"
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
