# Smoke test for the V3 architecture: numActors=2 + cross-actor attention.
#
# Validates the full refactor end-to-end:
#   - data loader emits `motion` (actor 0) + `motion_actor_1` (actor 1)
#   - TextToMotionSSM builds K=2 streams with actor embeddings
#   - Cross-actor attention layer fires between every SSM layer
#   - Forward returns tuple of length 2 (one logits per actor)
#   - Trainer iterates over both actors, mean CE loss
#   - Eval averages metrics over both actors
#
# Watch for in the log:
#   multi-actor mode: K=2 streams  cross_actor_attention=True
#   InterX paired: N pairs kept
#   Epoch 1: ... loss=... tok_ce=...
# If model construction or forward fails -> the refactor has a bug we need
# to find before the real run.

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_v3_$ts.out"
$err = "smoke_v3_$ts.err"

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
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- smoke V3 (numActors=2 + cross-actor attention) ---"
Write-Host "  d_model=256  n_layers=4  K=2  cross-attn=ON"
Write-Host "  max_samples=300  epochs=2  batch=8"
Write-Host "  estimated wall time: ~12-15 min"
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

$p.Id | Out-File -Encoding utf8 smoke_v3.pid

Write-Host "Launched PID=$($p.Id)"
Write-Host ""
Write-Host "Watch:   Get-Content $err -Wait -Tail 5"
Write-Host "Kill:    Stop-Process -Id (Get-Content smoke_v3.pid)"
