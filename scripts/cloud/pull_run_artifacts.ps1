# Pull the best_model + training.log + config from a finished S3 run dir into
# the local checkpoints tree so we have a record. Skips per-epoch checkpoints
# and wandb noise to keep the download small.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\cloud\pull_run_artifacts.ps1 -RunId 20260512-160903

param(
    [Parameter(Mandatory=$true)][string]$RunId,
    [string]$Bucket = "dissertation-motion-cache-91340264"
)

$srcPrefix = "s3://$Bucket/checkpoints/motion_ssm/$RunId"
$dstDir    = "checkpoints/motion_ssm/$RunId"

Write-Host "Pulling $srcPrefix -> $dstDir"
Write-Host "(excluding per-epoch checkpoints + wandb)"
Write-Host ""

aws s3 sync $srcPrefix $dstDir `
    --exclude "wandb/*" `
    --exclude "checkpoint_epoch*.pt"

Write-Host ""
Write-Host "--- local copy ---"
Get-ChildItem $dstDir -File |
    Select-Object Name,
                  @{n='MB';e={[math]::Round($_.Length / 1MB, 1)}},
                  LastWriteTime |
    Format-Table -AutoSize
