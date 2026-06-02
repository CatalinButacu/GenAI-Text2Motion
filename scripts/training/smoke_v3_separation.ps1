# Smoke test for V3 + inter-actor separation losses.
#
# Validates:
#   - --separation-loss-weight and --overlap-penalty-weight CLI flags propagate
#   - softDecodePose works on logitsList[0] and logitsList[1] without OOM
#   - interactorDistLoss returns finite values
#   - per-step postfix shows `sep=` and `overlap=` alongside `tok_ce=`
#   - no NaN through the soft-decode + L2 distance gradient

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_v3sep_$ts.out"
$err = "smoke_v3sep_$ts.err"

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
    "--epochs", "2",
    "--num-workers", "0",
    "--max-samples", "300",
    "--separation-loss-weight", "5.0",
    "--overlap-penalty-weight", "2.0",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- smoke V3 + separation losses ---"
Write-Host "  separation=5.0  overlap_penalty=2.0  min_dist=0.3"
Write-Host "  300 samples / 2 epochs ~ 10-15 min"
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

$p.Id | Out-File -Encoding utf8 smoke_v3sep.pid

Write-Host "Launched PID=$($p.Id)"
