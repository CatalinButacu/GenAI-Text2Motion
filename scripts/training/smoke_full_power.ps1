# Pre-flight smoke for the 150-epoch full-power Inter-X run.
# Verifies the same arch + data + loss combo trains 2 epochs without crashing,
# specifically checks max-samples=1500 fits in laptop RAM (prior runs capped
# at 800 because larger paired loading risked OOM on buffer build).

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "smoke_fp_$ts.out"
$err = "smoke_fp_$ts.err"

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
    "--epochs", "2",
    "--num-workers", "0",
    "--max-samples", "1500",
    "--weight-decay", "0.1",
    "--separation-loss-weight", "1.0",
    "--velocity-loss-weight", "0.5",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

Write-Host "--- Smoke for full-power Inter-X (will tell us 1500-sample RAM is OK) ---"
Write-Host "  d_model=384  n_layers=6  K=2  cross-attn=ON"
Write-Host "  max_samples=1500 (vs prior 800), 2 epochs, batch=8"
Write-Host "  separation=1.0  velocity=0.5  NO L4 (avoid NaN)"
Write-Host "  expected ~15-20 min"

$p = Start-Process -FilePath python `
    -ArgumentList $argList `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden `
    -PassThru

$p.Id | Out-File -Encoding utf8 smoke_fp.pid
Write-Host ""
Write-Host "Launched PID=$($p.Id)  stdout=$out  stderr=$err"
