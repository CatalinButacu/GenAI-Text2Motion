# powershell -NoExit -ExecutionPolicy Bypass -File scripts/training/watch_training.ps1
param(
    [string]$LogDir = (Join-Path $PSScriptRoot "..\..\runs\outputs"),
    [int]$PollSeconds = 5,
    [int]$TailBytes = 3000
)
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "training logs"

function __OpenFollower {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    $null = $stream.Seek([math]::Max(0, $stream.Length - $TailBytes), [IO.SeekOrigin]::Begin)
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
    if ($stream.Position -gt 0) {
        $null = $reader.ReadLine()
    }
    return $reader
}

function __NewestRunLog {
    return Get-ChildItem $LogDir -Filter "*.log" -File |
        Where-Object { $_.Name -notlike "*_driver.log" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

$current = ""
$followers = @()
while ($true) {
    $newest = __NewestRunLog
    if ($newest -and $newest.FullName -ne $current) {
        foreach ($follower in $followers) {
            $follower.Reader.Dispose()
        }
        $current = $newest.FullName
        Write-Host ""
        Write-Host "=== $($newest.Name)  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
        Get-Content $current -TotalCount 1 | Write-Host -ForegroundColor Yellow
        $followers = @()
        foreach ($pair in @(@($current, "Gray"), @("$current.err", "Red"))) {
            $reader = __OpenFollower -Path $pair[0]
            if ($reader) {
                $followers += [pscustomobject]@{ Reader = $reader; Color = $pair[1] }
            }
        }
    }
    foreach ($follower in $followers) {
        $text = $follower.Reader.ReadToEnd()
        if ($text) {
            Write-Host -NoNewline $text -ForegroundColor $follower.Color
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
