# Quick-look status for the smoke training run.
# Shows: process state, last log lines, GPU usage, target outputs.

$pidFile = "smoke_train.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "No smoke_train.pid - not launched."
    exit 0
}

$procId = [int](Get-Content $pidFile)
$proc   = Get-Process -Id $procId -ErrorAction SilentlyContinue

if ($proc) {
    $cpuSec = [math]::Round($proc.CPU, 1)
    $rssMb  = [math]::Round($proc.WS / 1MB, 0)
    Write-Host "Smoke PID $procId ALIVE  CPU=${cpuSec}s  RSS=${rssMb}MB"
} else {
    Write-Host "Smoke PID $procId is DONE / DEAD"
}

# Find the latest smoke_train_*.err file
$errFile = Get-ChildItem smoke_train_*.err -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($errFile) {
    Write-Host ""
    Write-Host "--- $($errFile.Name) tail ---"
    Get-Content $errFile.FullName -Tail 12
}

Write-Host ""
Write-Host "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader

Write-Host ""
Write-Host "--- trainer run dirs (newest first) ---"
Get-ChildItem checkpoints\motion_ssm -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 Name, LastWriteTime
