# Smoke test for dual-person Inter-X training.
#
# Validates the new paired-loader + dual-decoder-SSM end-to-end:
#   - data: --sources interx_paired loads paired (P1, P2) motions
#   - model: --dual-person grows a second RVQ decoder head
#   - loss: 0.5 * (CE_p1 + CE_p2) on tokens
#   - eval: top-1 averaged over both streams
#
# Should print:
#   [UnifiedDataset] InterX paired: N / M pairs kept
#   dual-person mode: ON (model will grow a 2nd RVQ decoder head)
# Per-step postfix should show tok_ce ~ 6.5 initially (random over 1024 codes).
# val_ce should drop monotonically over the 2 epochs.
#
# Sized small so we can verify wiring in ~10-15 min before committing to a real run.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_dual_$ts.out"
$err = "smoke_dual_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "interx_paired",
    "--dual-person",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "384", "--d-state", "64", "--n-layers", "6",
    "--max-motion-length", "200",
    "--batch-size", "8",
    "--lr", "3e-4",
    "--epochs", "2",
    "--num-workers", "0",
    "--max-samples", "300",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- smoke dual-person Inter-X ---"
Write-Host "  --sources interx_paired --dual-person"
Write-Host "  max_samples=300 pairs, 2 epochs, batch=8"
Write-Host "  estimated wall time: ~10-15 min"
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

$p.Id | Out-File -Encoding utf8 smoke_dual.pid

Write-Host "Launched PID=$($p.Id)"
Write-Host ""
Write-Host "Watch:   Get-Content $err -Wait -Tail 5"
Write-Host "Kill:    Stop-Process -Id (Get-Content smoke_dual.pid)"
