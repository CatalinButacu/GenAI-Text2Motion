# Three-step "ship the new best_model" workflow:
#   1. Copy <run_dir>/best_model.pt over checkpoints/motion_ssm/best_model.pt
#      (with a timestamped backup of whatever was there before)
#   2. Try inference with python main.py
#   3. If inference fails on the SBERT key mismatch, run fix_sbert_keys.py
#      in-place on the promoted file, then retry inference
#
# Usage from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\maintenance\promote_and_test.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\maintenance\promote_and_test.ps1 -RunId 20260513-174038
#   powershell -ExecutionPolicy Bypass -File scripts\maintenance\promote_and_test.ps1 -Prompt "a person sits down"

param(
    [string]$RunId  = "20260513-174038",
    [string]$Prompt = "a person walks forward",
    [string]$Name   = "test_local_best",
    [int]$Duration  = 4,
    [int]$Fps       = 30
)

$ErrorActionPreference = "Stop"

$src = "checkpoints/motion_ssm/$RunId/best_model.pt"
$dst = "checkpoints/motion_ssm/best_model.pt"

if (-not (Test-Path $src)) {
    Write-Error "Source not found: $src"
    exit 1
}

# Step 1: promote (with backup)
$ts        = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupDst = "checkpoints/motion_ssm/best_model_pre_$ts.pt"

Write-Host "--- Step 1: promote ---"

if (Test-Path $dst) {
    Write-Host "  backup: $dst -> $backupDst"
    Move-Item $dst $backupDst -Force
}

Write-Host "  copy:   $src -> $dst"
Copy-Item $src $dst -Force

$srcMb = [math]::Round((Get-Item $src).Length / 1MB, 1)
$dstMb = [math]::Round((Get-Item $dst).Length / 1MB, 1)
Write-Host "  sizes:  src=${srcMb}MB  dst=${dstMb}MB  (match: $($srcMb -eq $dstMb))"

# Step 2 & 3: inference, with auto-fix on SBERT key mismatch
function Invoke-Inference {
    param([string]$attemptLabel)
    Write-Host ""
    Write-Host "--- $attemptLabel ---"

    $outFile = "inference_$ts.out"
    $errFile = "inference_$ts.err"

    $argList = @(
        "main.py",
        "`"$Prompt`"",
        "--name", $Name,
        "--duration", "$Duration",
        "--fps", "$Fps",
        "--device", "cuda"
    )

    # Run synchronously this time (not background) so we get the exit code.
    $proc = Start-Process -FilePath python `
        -ArgumentList $argList `
        -RedirectStandardOutput $outFile `
        -RedirectStandardError $errFile `
        -WindowStyle Hidden `
        -PassThru `
        -Wait

    Write-Host "  exit code: $($proc.ExitCode)"
    return $proc.ExitCode
}

$rc = Invoke-Inference -attemptLabel "Step 2: first inference attempt"

if ($rc -ne 0) {
    # Check if it was the SBERT key error
    $errContent = Get-Content "inference_$ts.err" -Raw -ErrorAction SilentlyContinue

    if ($errContent -match "text_encoder\.sbert\.0\.model\." -or `
        $errContent -match "text_encoder\.sbert\.0\.auto_model\.") {

        Write-Host ""
        Write-Host "--- Step 3: SBERT key mismatch detected, running migration ---"

        python scripts/maintenance/fix_sbert_keys.py $dst --inplace

        $rc = Invoke-Inference -attemptLabel "Step 3b: retry after key fix"
    } else {
        Write-Host "ERROR: inference failed with a non-SBERT issue. See inference_$ts.err"
        Get-Content "inference_$ts.err" -Tail 25
        exit 1
    }
}

if ($rc -eq 0) {
    Write-Host ""
    Write-Host "--- Done ---"
    Get-Content "inference_$ts.out" -Tail 5
    Write-Host ""
    Write-Host "Look in outputs/ for: $Name.mp4 (or similar)"
    Get-ChildItem outputs -Filter "$Name*" -ErrorAction SilentlyContinue |
        Select-Object Name, @{n='KB';e={[math]::Round($_.Length / 1KB, 1)}}, LastWriteTime |
        Format-Table -AutoSize
} else {
    Write-Host "Inference still failing after fix. See inference_$ts.err"
    Get-Content "inference_$ts.err" -Tail 25
    exit 1
}
