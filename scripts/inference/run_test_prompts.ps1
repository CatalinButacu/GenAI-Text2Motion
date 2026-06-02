# Run a list of test prompts through main.py to generate motion videos.
# Captures per-prompt exit code + log tail + output file size so we can spot
# failures (e.g. NaN in motion -> render crash, OOM, etc.).
#
# Each prompt runs synchronously so we can attribute errors to the right one.
#
# Usage from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\inference\run_test_prompts.ps1
#
# To add prompts, edit the $prompts array below.

$prompts = @(
    @{ Prompt = "a person sits down";       Name = "test_sit";   Duration = 4 },
    @{ Prompt = "a person kicks a ball";    Name = "test_kick";  Duration = 5 },
    @{ Prompt = "a person runs forward";    Name = "test_run";   Duration = 4 }
)

$ts = Get-Date -Format 'yyyyMMdd-HHmmss'
$results = @()

foreach ($p in $prompts) {
    Write-Host ""
    Write-Host "============================================="
    Write-Host " $($p.Name)  =>  '$($p.Prompt)'"
    Write-Host "============================================="

    $outFile = "inference_$($p.Name)_$ts.out"
    $errFile = "inference_$($p.Name)_$ts.err"

    $argList = @(
        "main.py",
        "`"$($p.Prompt)`"",
        "--name", $p.Name,
        "--duration", "$($p.Duration)",
        "--fps", "30",
        "--device", "cuda"
    )

    $start = Get-Date

    $proc = Start-Process -FilePath python `
        -ArgumentList $argList `
        -RedirectStandardOutput $outFile `
        -RedirectStandardError $errFile `
        -WindowStyle Hidden `
        -PassThru `
        -Wait

    $elapsedSec = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)

    # Renderer saves <name>_0.mp4 (clip-index suffix), not <name>.mp4.
    $videoPath = Get-ChildItem "outputs/videos" -Filter "$($p.Name)_*.mp4" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending |
                 Select-Object -First 1
    $videoExists = $null -ne $videoPath
    $videoMb = if ($videoExists) {
        [math]::Round($videoPath.Length / 1MB, 2)
    } else { 0 }

    $results += [pscustomobject]@{
        Name      = $p.Name
        Prompt    = $p.Prompt
        ExitCode  = $proc.ExitCode
        Seconds   = $elapsedSec
        VideoMB   = $videoMb
        VideoOk   = $videoExists
    }

    if ($proc.ExitCode -eq 0) {
        Write-Host "  exit=$($proc.ExitCode)  ${elapsedSec}s  video=${videoMb}MB at $videoPath"
        # Show interesting tail line ("video ->")
        Get-Content $outFile -Tail 5 | Where-Object { $_ -match "video|entities|actions" }
    } else {
        Write-Host "  exit=$($proc.ExitCode)  ${elapsedSec}s  FAILED"
        Write-Host "  err tail:"
        Get-Content $errFile -Tail 10
    }
}

Write-Host ""
Write-Host "============================================="
Write-Host " Summary"
Write-Host "============================================="

$results | Format-Table -AutoSize

# List all the produced videos
Write-Host ""
Write-Host "--- outputs/videos/ ---"
Get-ChildItem outputs/videos -Filter "test_*.mp4" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name,
                  @{n='MB';e={[math]::Round($_.Length / 1MB, 2)}},
                  LastWriteTime |
    Format-Table -AutoSize
