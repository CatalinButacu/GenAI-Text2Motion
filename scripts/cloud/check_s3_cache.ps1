# Show what is currently in the S3 cache prefix, sorted by size.
# Useful for diagnosing accidental over-uploads.

param(
    [string]$Bucket = "dissertation-motion-cache-91340264"
)

Write-Host "--- s3://$Bucket/cache/ contents ---"
aws s3 ls "s3://$Bucket/cache/" --human-readable

Write-Host ""
Write-Host "--- total size ---"
aws s3 ls "s3://$Bucket/cache/" --recursive --summarize --human-readable |
    Select-Object -Last 4
