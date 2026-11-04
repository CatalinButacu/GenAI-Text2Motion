# powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research/paper_watch_nightly.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$watch = Join-Path $repo "doc\notes\paper-watch"
$day = Get-Date -Format "yyyy-MM-dd"
$report = Join-Path $watch "$day.md"
$marks = Join-Path $watch "$day-marks.md"
$brief = Join-Path $PSScriptRoot "paper_watch_marks.md"
$copilot = Join-Path $env:APPDATA "npm\copilot.cmd"
$log = Join-Path $watch "nightly.log"

& (Join-Path $repo ".venv\Scripts\python.exe") -u (Join-Path $PSScriptRoot "paper_watch.py") --out_dir $watch *>> $log
if ($LASTEXITCODE -ne 0) { throw "paper_watch.py failed with exit code $LASTEXITCODE, see $log" }

$prompt = "Read the file $brief completely and do the task it describes. The report file is $report."
& $copilot -C $repo --agent research-scout -p $prompt -s --model gpt-6-luna --allow-all-tools --deny-tool shell --deny-tool write > $marks 2>> $log
if ($LASTEXITCODE -ne 0) { throw "copilot failed with exit code $LASTEXITCODE, see $log" }
"$(Get-Date -Format s) done: $report, $marks" | Add-Content $log
