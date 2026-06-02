# Local training run targeted at walking + running motions, with the new
# L4 pose-decode auxiliary loss.
#
# Defaults match the smoke result that confirmed end-to-end wiring:
#   - text keywords: walk + run -> filter keeps ~2000-3000 samples from the
#     21K-sample unified cache
#   - pose_loss_weight: 0.1 (lower than the smoke's 0.3 -- let token CE drive
#     early epochs, L4 polish later)
#   - 15 epochs, batch=8, d_model=384 / n_layers=6 (proven local config)
#   - no --max-samples cap so the filter sees the full dataset
#
# Expected: with ~2000+ walk/run samples and 15 epochs, val_ce should land
# below the 4.1465 baseline (which was on 5000 mixed-domain samples). Target
# val_ce ~ 3.8-4.0 on this narrow domain.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\training\walkrun_train_local.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\training\walkrun_train_local.ps1 -PoseLossWeight 0.2 -Epochs 20

param(
    [int]$BatchSize        = 8,
    [int]$Epochs           = 15,
    [double]$Lr            = 3e-4,
    [string]$Keywords      = "walk,run",
    [double]$PoseLossWeight = 0.1
)

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "walkrun_train_$ts.out"
$err = "walkrun_train_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "amass", "humanml3d",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "384", "--d-state", "64", "--n-layers", "6",
    "--max-motion-length", "200",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--epochs", "$Epochs",
    "--num-workers", "0",
    "--text-keywords", $Keywords,
    "--pose-loss-weight", "$PoseLossWeight",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

# Rough wall-time estimate from the smoke: 3.4 sec/step.
# Assume filter keeps ~2500 walk/run samples; train split ~80%.
$estSamples    = 2500
$stepsPerEpoch = [math]::Ceiling(($estSamples * 0.8) / $BatchSize)
$walltimeHrs   = ($stepsPerEpoch * 3.4 * $Epochs) / 3600.0

Write-Host "--- walk+run local train ---"
Write-Host ("  keywords        = $Keywords")
Write-Host ("  batch           = $BatchSize")
Write-Host ("  epochs          = $Epochs")
Write-Host ("  poseLossWeight  = $PoseLossWeight")
Write-Host ("  lr (peak)       = $Lr")
Write-Host ("  est steps/epoch ~ $stepsPerEpoch (assuming ~$estSamples walk/run samples)")
Write-Host ("  est wall time   ~ {0:F1} hours" -f $walltimeHrs)
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
