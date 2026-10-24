# powershell -ExecutionPolicy Bypass -File scripts/training/local_guard.ps1 -TargetPid 1234 -HeartbeatPath artifacts/logs/train/RUN/heartbeat
param(
    [Parameter(Mandatory = $true)][int]$TargetPid,
    [string]$HeartbeatPath = "",
    [int]$StaleMinutes = 30,
    [double]$MaxHours = 12,
    [int]$PollSeconds = 120
)
$started = Get-Date
$reason = ""
while ($true) {
    Start-Sleep -Seconds $PollSeconds
    $proc = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Output "guard: pid $TargetPid exited on its own - guard done"
        exit 0
    }
    $elapsed = ((Get-Date) - $started).TotalHours
    if ($elapsed -ge $MaxHours) {
        $reason = "max-hours budget ($MaxHours h) reached"
        break
    }
    if ($StaleMinutes -gt 0 -and $HeartbeatPath) {
        $lastSign = if (Test-Path $HeartbeatPath) { (Get-Item $HeartbeatPath).LastWriteTime } else { $started }
        $age = ((Get-Date) - $lastSign).TotalMinutes
        if ($age -ge $StaleMinutes) {
            $reason = "no heartbeat for $([math]::Round($age)) min (frozen run)"
            break
        }
    }
}
Write-Output "guard: KILLING pid $TargetPid - $reason"
Stop-Process -Id $TargetPid -Force -Confirm:$false
$marker = "artifacts/outputs/GUARD_KILL.txt"
"$((Get-Date).ToUniversalTime().ToString('u'))  pid $TargetPid  $reason" | Out-File -Append $marker -Encoding utf8
Write-Output "guard: logged to $marker - inspect before relaunching (fail loud, do not ignore)"
exit 1
