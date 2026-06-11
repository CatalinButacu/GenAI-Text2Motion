# Local training guardrail — the laptop twin of the cloud cost guards (hard lifetime + idle
# watchdog). Born from the 2026-06-10 incident: a tokenizer run froze mid-epoch and burned 10.6 h
# of CPU unnoticed. Kills the target process when EITHER trigger fires and logs the reason loudly.
#
#   guard 1 (stall):  no write to -HeartbeatPath for -StaleMinutes (run_log's metrics.jsonl is
#                     unbuffered, one line per epoch/eval -> a frozen run stops touching it)
#   guard 2 (budget): -MaxHours wall-clock elapsed (the "I need my laptop back" guarantee)
#
#   powershell scripts/local_guard.ps1 -TargetPid 1234 -HeartbeatPath outputs/runs/<run>/metrics.jsonl `
#       -StaleMinutes 30 -MaxHours 8
#
# Pass -StaleMinutes 0 to disable the stall guard (bounded jobs like eval that only log at the end).
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
    if (-not $proc) { Write-Output "guard: pid $TargetPid exited on its own - guard done"; exit 0 }

    $elapsed = ((Get-Date) - $started).TotalHours
    if ($elapsed -ge $MaxHours) { $reason = "max-hours budget ($MaxHours h) reached"; break }

    if ($StaleMinutes -gt 0 -and $HeartbeatPath -and (Test-Path $HeartbeatPath)) {
        $age = ((Get-Date) - (Get-Item $HeartbeatPath).LastWriteTime).TotalMinutes
        if ($age -ge $StaleMinutes) { $reason = "heartbeat stale for $([math]::Round($age)) min (frozen run)"; break }
    }
}

Write-Output "guard: KILLING pid $TargetPid - $reason"
Stop-Process -Id $TargetPid -Force -Confirm:$false
$marker = "outputs/GUARD_KILL.txt"
"$((Get-Date).ToUniversalTime().ToString('u'))  pid $TargetPid  $reason" | Out-File -Append $marker -Encoding utf8
Write-Output "guard: logged to $marker - inspect before relaunching (fail loud, do not ignore)"
exit 1
