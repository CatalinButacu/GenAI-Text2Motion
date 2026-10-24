# Import-Module (Join-Path $PSScriptRoot "TrainingRun.psm1") -Force
Set-StrictMode -Version 3.0

function Enter-RepoRoot {
    param([Parameter(Mandatory = $true)][string]$StartDirectory)
    $root = $StartDirectory
    while ($root -and -not (Test-Path (Join-Path $root "pyproject.toml"))) {
        $root = Split-Path $root -Parent
    }
    if (-not $root) {
        throw "repo root (pyproject.toml) not found above $StartDirectory"
    }
    Set-Location $root
    $env:PYTHONPATH = "src"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    New-Item -ItemType Directory -Force "artifacts/outputs" | Out-Null
    return $root
}

function Get-ResumeFlag {
    param([Parameter(Mandatory = $true)][string]$ResumePath)
    if (Test-Path $ResumePath) {
        return "--resume"
    }
    return ""
}

function Test-RunComplete {
    param(
        [Parameter(Mandatory = $true)][string]$ResumePath,
        [Parameter(Mandatory = $true)][int]$Epochs,
        [string]$Python = ".\.venv\Scripts\python.exe"
    )
    if (-not (Test-Path $ResumePath)) {
        return $false
    }
    $read = "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)['epoch'] + 1)"
    $reached = & $Python -c $read $ResumePath
    if ($LASTEXITCODE -ne 0) {
        throw "cannot read the epoch from $ResumePath"
    }
    return ([int]$reached -ge $Epochs)
}

function Join-QuotedArguments {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Arguments)
    $quoted = foreach ($argument in ($Arguments | Where-Object { $_ })) {
        if ($argument -match '[\s"]') {
            '"' + ($argument -replace '"', '\"') + '"'
        }
        else {
            $argument
        }
    }
    return ($quoted -join " ")
}

function Get-NewestHeartbeat {
    param([Parameter(Mandatory = $true)][datetime]$Since)
    return Get-ChildItem "artifacts/logs/train" -Recurse -Filter "heartbeat" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $Since } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Invoke-TrainingStep {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogFile,
        [double]$MaxHours = 72,
        [int]$StaleMinutes = 0,
        [string]$Python = ".\.venv\Scripts\python.exe",
        [int]$PollSeconds = 60
    )
    $argumentLine = Join-QuotedArguments -Arguments (@("-u") + $Arguments)
    Write-Host "RUN ${Tag}: $Python $argumentLine"
    $started = Get-Date
    $process = Start-Process -FilePath $Python -ArgumentList $argumentLine -PassThru -NoNewWindow `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err"
    $null = $process.Handle
    $reason = ""
    while (-not $process.WaitForExit($PollSeconds * 1000)) {
        $hours = ((Get-Date) - $started).TotalHours
        if ($hours -ge $MaxHours) {
            $reason = "time budget of $MaxHours h reached"
            break
        }
        if ($StaleMinutes -le 0) {
            continue
        }
        $heartbeat = Get-NewestHeartbeat -Since $started
        $lastSign = if ($heartbeat) { $heartbeat.LastWriteTime } else { $started }
        $age = ((Get-Date) - $lastSign).TotalMinutes
        if ($age -ge $StaleMinutes) {
            $reason = "no heartbeat for $([math]::Round($age)) min"
            break
        }
    }
    if ($reason) {
        Stop-Process -Id $process.Id -Force
        $stamp = (Get-Date).ToUniversalTime().ToString("u")
        "$stamp  $Tag KILLED: $reason" | Out-File -Append "artifacts/outputs/GUARD_KILL.txt" -Encoding utf8
        Write-Host "KILLED ${Tag}: $reason"
        return 124
    }
    $code = $process.ExitCode
    $elapsed = [math]::Round(((Get-Date) - $started).TotalHours, 2)
    Write-Host "EXIT ${Tag}: code $code after $elapsed h"
    return $code
}

function Write-DoneMarker {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int[]]$ExitCodes
    )
    $failed = @($ExitCodes | Where-Object { $_ -ne 0 })
    if ($failed.Count -gt 0) {
        Write-Host "NOT DONE: $($failed.Count) step(s) failed; no marker at $Path"
        return $false
    }
    "done $((Get-Date).ToUniversalTime().ToString('u'))" | Out-File $Path -Encoding utf8
    Write-Host "DONE: marker written to $Path"
    return $true
}

Export-ModuleMember -Function Enter-RepoRoot, Get-ResumeFlag, Test-RunComplete, Invoke-TrainingStep, Write-DoneMarker
