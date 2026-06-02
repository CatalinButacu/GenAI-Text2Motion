# Upload locally-built unified buffer joblibs to S3 so the EC2 trainer can
# reuse them instead of needing raw AMASS .npz files on the cloud.
#
# What it does:
#   1. Lists data/.cache/unified_buf_*.joblib produced by prebuild_unified_cache.py
#   2. aws s3 sync of that directory into s3://<bucket>/cache/
#   3. Confirms the target hash file is now in S3
#
# Usage from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\cloud\upload_unified_cache.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\cloud\upload_unified_cache.ps1 -ExpectedHash 8eb88293994e

param(
    [string]$Bucket       = "dissertation-motion-cache-91340264",
    [string]$ExpectedHash = ""    # if set, verify this specific joblib is present locally + in S3 after upload
)

$cacheDir = "data/.cache"

Write-Host "--- Local cache joblibs ---"
Get-ChildItem "$cacheDir/unified_buf_*.joblib" |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name,
                  @{n='MB';e={[math]::Round($_.Length / 1MB, 1)}},
                  LastWriteTime |
    Format-Table -AutoSize

if ($ExpectedHash) {
    $expectedPath = "$cacheDir/unified_buf_$ExpectedHash.joblib"

    if (-not (Test-Path $expectedPath)) {
        Write-Error "Expected cache file missing locally: $expectedPath"
        exit 1
    }
}

Write-Host ""
Write-Host "--- Uploading $cacheDir/*.joblib to s3://$Bucket/cache/ ---"
aws s3 sync $cacheDir "s3://$Bucket/cache/" `
    --exclude "*" `
    --include "*.joblib"

Write-Host ""
Write-Host "--- S3 cache contents after upload ---"
aws s3 ls "s3://$Bucket/cache/" --human-readable

if ($ExpectedHash) {
    Write-Host ""
    Write-Host "--- Verifying $ExpectedHash exists in S3 ---"
    aws s3 ls "s3://$Bucket/cache/unified_buf_$ExpectedHash.joblib" | Out-Host

    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK: cache hash $ExpectedHash is now in S3"
    } else {
        Write-Error "FAIL: cache hash $ExpectedHash not found in S3 after upload"
        exit 1
    }
}
