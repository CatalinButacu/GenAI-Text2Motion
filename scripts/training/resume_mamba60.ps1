param([double]$BudgetHoursPerRun = 72, [int]$StaleMinutes = 90)
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
while ($root -and -not (Test-Path (Join-Path $root "pyproject.toml"))) {
    $root = Split-Path $root -Parent
}
if (-not $root) {
    throw "repo root (pyproject.toml) not found above $PSScriptRoot"
}
Set-Location $root
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date
$runDir = "artifacts/logs/train/20260921T161630Z_generator_mamba"
$label = "resume_mamba60"
$logFile = "artifacts/outputs/$label.log"

function Hrs {
    return [math]::Round(((Get-Date) - $start).TotalHours, 1)
}

if (-not (Test-Path $runDir)) {
    throw "run dir not found: $runDir"
}

Write-Output "=== resume mamba 60-epoch twin from $runDir  start $(Get-Date -Format u) ==="

$before = Get-Date
$p = Start-Process -FilePath $py -ArgumentList (
    "-u -m text2motion.app.cli train-generator " +
    "--config configs/generator/gen_pilot_fsq8x1024.yaml " +
    "--backbone mamba " +
    "--tokenizer_ckpt artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt " +
    "--overfit_gate artifacts/outputs/gates/mamba_34m_overfit.json " +
    "--init_ckpt artifacts/checkpoints/generator/generator_mamba_pretrained.pt " +
    "--ckpt_name generator_mamba_bs8_e60.pt " +
    "--epochs 60 --batch_size 8 --num_workers 4 --eval_every 6 --cfg_scale 6.0 --temperature 1.0 " +
    "--resume --resume_run_dir $runDir"
) -PassThru -NoNewWindow -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"

Start-Sleep -Seconds 45
$heartbeat = Join-Path $runDir "heartbeat"

while (-not $p.HasExited) {
    Start-Sleep -Seconds 120
    $el = ((Get-Date) - $before).TotalHours
    if ($el -ge $BudgetHoursPerRun) {
        $r = "max-hours ($BudgetHoursPerRun h)"
        break
    }
    if (Test-Path $heartbeat) {
        $age = ((Get-Date) - (Get-Item $heartbeat).LastWriteTime).TotalMinutes
        if ($age -ge $StaleMinutes) {
            $r = "stall (${age}min stale heartbeat)"
            break
        }
    }
}

if (-not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
    Start-Sleep 5
    "$((Get-Date).ToString('u'))  $label KILLED: $r" | Out-File -Append artifacts/outputs/GUARD_KILL.txt -Encoding utf8
    Write-Output "GUARD KILLED ${label}: $r  (best-by-val checkpoint is safe; rerun this script to resume again)"
} else {
    Write-Output "$label exited on its own (code $($p.ExitCode)) at $(Hrs)h -- verify via checkpoint, exit code is unreliable on this machine"
}
