# Smoke test for the new --text-keywords filter + --pose-loss-weight L4 loss.
#
# Goal: verify the wiring works end-to-end in ~6 min before committing to a
# multi-hour run. Should print these in the log:
#   [unified] text filter keywords=['walk', 'run'] -> kept N / M samples
#   text-filter keywords: walk,run
#   aux pose-decode L4 loss weight: 0.300
# And the per-step tqdm postfix should include pose_l4=... alongside tok_ce.
#
# If it NaN's or the loss explodes, we kill and lower poseLossWeight.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_wrl4_$ts.out"
$err = "smoke_wrl4_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "amass", "humanml3d",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "384", "--d-state", "64", "--n-layers", "6",
    "--max-motion-length", "200",
    "--batch-size", "8",
    "--lr", "3e-4",
    "--epochs", "2",
    "--num-workers", "0",
    "--max-samples", "2000",
    "--text-keywords", "walk,run",
    "--pose-loss-weight", "0.3",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- smoke walk+run + L4 aux loss ---"
Write-Host "  max_samples=2000 (pre-filter), text keywords=walk,run"
Write-Host "  expect filter to keep ~300-700 samples"
Write-Host "  batch=8, epochs=2, pose-loss-weight=0.3"
Write-Host "  estimated wall time: ~6 min"
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

$p.Id | Out-File -Encoding utf8 smoke_wrl4.pid

Write-Host "Launched PID=$($p.Id)"
Write-Host ""
Write-Host "Watch with: Get-Content $err -Wait -Tail 5"
Write-Host "Kill with:  Stop-Process -Id (Get-Content smoke_wrl4.pid)"
