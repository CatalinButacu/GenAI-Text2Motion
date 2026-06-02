# Status helper for the active local training run.
# Shows: process state, last per-epoch metrics, GPU temperature/util, run dir.

$pidFile = "local_train.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "No local_train.pid - nothing launched in this session."
    exit 0
}

$procId = [int](Get-Content $pidFile)
$proc   = Get-Process -Id $procId -ErrorAction SilentlyContinue

if ($proc) {
    $cpuSec  = [math]::Round($proc.CPU, 1)
    $rssMb   = [math]::Round($proc.WS / 1MB, 0)
    Write-Host "Train PID $procId ALIVE  CPU=${cpuSec}s  RSS=${rssMb}MB"
} else {
    Write-Host "Train PID $procId is DONE / DEAD"
}

# Find the most-recently-written training .err file. Matches launcher
# patterns: local_train_*.err, walkrun_train_*.err, smoke_*.err.
$patterns = @(
    "local_train_*.err",
    "walkrun_train_*.err",
    "smoke_*.err",
    "v3_run_*.err",
    "v3sep_run_*.err",
    "path1_run_*.err",
    "pathc_run_*.err",
    "fullpower_*.err",
    "quickwin_*.err"
)
$errFile  = $patterns |
    ForEach-Object { Get-ChildItem $_ -File -ErrorAction SilentlyContinue } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $errFile) {
    Write-Host "No training .err file in cwd (looked for local_train_*, walkrun_train_*, smoke_*)."
    exit 0
}

Write-Host ""
Write-Host "--- per-epoch summary (from $($errFile.Name)) ---"
$epochs = Select-String -Path $errFile.FullName -Pattern 'INFO\s+epoch=\d+/\d+' |
          Select-Object -Last 12 -ExpandProperty Line

if ($epochs) {
    $epochs
} else {
    Write-Host "no epoch= lines yet (still in setup or first epoch)"
}

Write-Host ""
Write-Host "--- last tqdm progress (current step) ---"
Get-Content $errFile.FullName -Tail 2

Write-Host ""
Write-Host "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw `
    --format=csv,noheader

Write-Host ""
Write-Host "--- newest run dirs ---"
Get-ChildItem checkpoints\motion_ssm -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 Name, LastWriteTime
