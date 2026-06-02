# Hard-kill the active training EC2 RIGHT NOW (saves any partial best_model first).
#
# Steps:
#   1. SSH in and force the trainer to flush its current best_model to S3
#      (the cleanupOnExit trap should do this anyway on terminate, but we save
#      a snapshot first as a belt-and-braces measure).
#   2. aws ec2 terminate-instances on the tracked instance.
#   3. Print final cost.

$ErrorActionPreference = "Continue"
$keyPath = Join-Path $env:USERPROFILE ".ssh/dissertation-eu-west-1.pem"

Push-Location D:\Facultate\dissertation\scripts\cloud\aws
$publicIp   = terraform output -raw public_ip 2>$null
$instanceId = terraform output -raw instance_id 2>$null
Pop-Location

if (-not $instanceId) {
    Write-Host "No instance to kill (terraform has none in state)."
    exit 0
}

Write-Host "Target: $instanceId @ $publicIp"
Write-Host ""

# Get launch time for final cost
$launchStr = aws ec2 describe-instances `
    --instance-ids $instanceId `
    --region eu-west-1 `
    --query "Reservations[].Instances[].LaunchTime" `
    --output text 2>$null

if ($launchStr) {
    $launch = [datetime]::Parse($launchStr).ToUniversalTime()
    $uptimeHrs = ([datetime]::UtcNow - $launch).TotalHours
    $finalCost = $uptimeHrs * 0.526

    Write-Host ("Uptime:    {0:F2} hours" -f $uptimeHrs)
    Write-Host ("Final cost ~$ {0:F2}" -f $finalCost)
    Write-Host ""
}

# Try to flush current best to S3 via SSH first (best-effort, ignore failure)
if ($publicIp) {
    Write-Host "--- Flushing any current checkpoints to S3 first ---"

    $sshOpts = @(
        "-i", $keyPath,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes"
    )

    ssh @sshOpts ubuntu@$publicIp `
        "aws s3 sync /home/ubuntu/repo/checkpoints s3://dissertation-motion-cache-91340264/checkpoints --storage-class STANDARD_IA 2>&1 | tail -5"
}

Write-Host ""
Write-Host "--- Terminating $instanceId ---"
aws ec2 terminate-instances `
    --instance-ids $instanceId `
    --region eu-west-1 `
    --query "TerminatingInstances[].{id:InstanceId,prev:PreviousState.Name,now:CurrentState.Name}" `
    --output table

Write-Host ""
Write-Host "Done. Run scripts\cloud\destroy_terraform.ps1 to clean up IAM/SG too."
