# Check the status of the local prebuild_unified_cache.py background process
# launched earlier in this session.
#
# Prints: process status, last log lines, and whether the target joblib exists.

param(
    [string]$ExpectedHash = "8eb88293994e"
)

$pidFile = "prebuild.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "No prebuild.pid found - no active prebuild in this session."
    exit 0
}

$procId = [int](Get-Content $pidFile)
$proc = Get-Process -Id $procId -ErrorAction SilentlyContinue

if ($proc) {
    $cpuSec = [math]::Round($proc.CPU, 1)
    $rssMb  = [math]::Round($proc.WS / 1MB, 0)
    Write-Host "Prebuild PID $procId ALIVE  CPU=${cpuSec}s  RSS=${rssMb}MB"
} else {
    Write-Host "Prebuild PID $procId has exited."
}

Write-Host ""
Write-Host "--- last log lines (prebuild_cache.err) ---"
if (Test-Path prebuild_cache.err) {
    Get-Content prebuild_cache.err -Tail 12
} else {
    Write-Host "no log yet"
}

Write-Host ""
Write-Host "--- target cache file (hash $ExpectedHash) ---"
$cachePath = "data/.cache/unified_buf_$ExpectedHash.joblib"

if (Test-Path $cachePath) {
    $size = [math]::Round((Get-Item $cachePath).Length / 1MB, 1)
    Write-Host "EXISTS: $cachePath  size=${size}MB  ready to upload"
} else {
    Write-Host "NOT YET: $cachePath"
}
