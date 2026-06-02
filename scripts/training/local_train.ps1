# Local MotionSSM training launcher with conservative, coexistence-friendly
# defaults. Parameterized so you can run smoke / medium / big without editing.
#
# Architecture is fixed to the proven d_model=384 / n_layers=6 / BiMamba+FiLM
# configuration that reached val_ce=4.6038 on the cloud. Only data size and
# epoch count change between runs.
#
# Defaults: 5000 samples, 10 epochs, batch=8. Expected: ~4-5h on RTX 3050
# Laptop in coexistence mode (laptop usable for normal work during training).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\training\local_train.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\training\local_train.ps1 -MaxSamples 10000 -Epochs 15
#   powershell -ExecutionPolicy Bypass -File scripts\training\local_train.ps1 -BatchSize 16
#
# Pidfile: local_train.pid (used by check_local_train.ps1)
# Logs: local_train_<timestamp>.{out,err} in project root.

param(
    [int]$BatchSize    = 8,
    [int]$MaxSamples   = 5000,
    [int]$Epochs       = 10,
    [double]$Lr        = 3e-4,
    [int]$NumWorkers   = 0
)

$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "local_train_$ts.out"
$err = "local_train_$ts.err"

$argList = @(
    "scripts/training/train_motion_ssm.py",
    "--data-source", "unified", "--sources", "amass", "humanml3d",
    "--use-sbert", "--bidirectional", "--use-film", "--gradient-checkpointing",
    "--d-model", "384", "--d-state", "64", "--n-layers", "6",
    "--max-motion-length", "200",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--epochs", "$Epochs",
    "--num-workers", "$NumWorkers",
    "--max-samples", "$MaxSamples",
    "--checkpoint-dir", "checkpoints/motion_ssm",
    "--rvq-checkpoint", "checkpoints/rvq_tokenizer/best_model.pt",
    "--device", "cuda"
)

# Rough wall-time forecast based on smoke-test measured 3.1 sec/step.
# Train split is ~80% of max_samples.
$stepsPerEpoch = [math]::Ceiling(($MaxSamples * 0.8) / $BatchSize)
$secPerStep    = 3.1
$walltimeHrs   = ($stepsPerEpoch * $secPerStep * $Epochs) / 3600.0

Write-Host "--- local train config ---"
Write-Host ("  d_model=384  n_layers=6  batch={0}  samples={1}  epochs={2}" `
            -f $BatchSize, $MaxSamples, $Epochs)
Write-Host ("  lr={0:G2} peak, OneCycle warmup -> peak -> anneal" -f $Lr)
Write-Host ("  steps/epoch ~ {0}" -f $stepsPerEpoch)
Write-Host ("  estimated wall time ~ {0:F1} hours" -f $walltimeHrs)
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
Write-Host "Check status with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\training\check_local_train.ps1"
Write-Host "Kill with:"
Write-Host "  Stop-Process -Id (Get-Content local_train.pid)"
