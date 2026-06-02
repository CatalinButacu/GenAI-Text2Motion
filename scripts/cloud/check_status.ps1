# All-in-one status check for the active training EC2:
#   - if instance is gone     -> "training finished (clean self-terminate)" or "killed"
#   - if instance is up:
#       - in setup phase      -> tail /var/log/user-data.log (cloud-init progress)
#       - in training phase   -> grep epoch= lines from training.log + GPU stats
#
# Reads the public IP from terraform output so it auto-targets the latest instance.

$keyPath = Join-Path $env:USERPROFILE ".ssh/dissertation-eu-west-1.pem"

Push-Location D:\Facultate\dissertation\scripts\cloud\aws
$publicIp   = terraform output -raw public_ip 2>$null
$instanceId = terraform output -raw instance_id 2>$null
Pop-Location

if (-not $publicIp) {
    Write-Host "No public_ip in terraform output. Check 'aws ec2 describe-instances'."
    exit 0
}

Write-Host "Instance: $instanceId @ $publicIp"

# Check from AWS whether the instance still exists
$state = aws ec2 describe-instances `
    --instance-ids $instanceId `
    --region eu-west-1 `
    --query "Reservations[].Instances[].State.Name" `
    --output text 2>$null

if (-not $state -or $state -eq "terminated" -or $state -eq "shutting-down") {
    Write-Host "AWS reports state: $state (or unknown)"
    Write-Host "Look for final artifacts in S3:"
    Write-Host "  aws s3 ls s3://dissertation-motion-cache-91340264/checkpoints/motion_ssm/"
    exit 0
}

Write-Host "AWS state: $state"
Write-Host ""

$sshOpts = @(
    "-i", $keyPath,
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes"
)

# One remote command that prints both phases (setup + training)
$remoteCmd = @"
# Phase detection: is training.log present yet?
RUN_ID=`$(ls -t /home/ubuntu/repo/checkpoints/motion_ssm/ 2>/dev/null | head -1)
TLOG=/home/ubuntu/repo/checkpoints/motion_ssm/`$RUN_ID/training.log

if [ -n "`$RUN_ID" ] && [ -f "`$TLOG" ] && grep -q 'epoch=' "`$TLOG" 2>/dev/null; then
    PHASE="training"
else
    PHASE="setup"
fi

echo "=== phase: `$PHASE ==="

if [ "`$PHASE" = "setup" ]; then
    echo '--- user-data.log tail ---'
    sudo tail -n 25 /var/log/user-data.log 2>/dev/null || tail -n 25 /var/log/user-data.log
else
    echo '--- run_id ---'
    echo `$RUN_ID
    echo '--- epoch summary ---'
    grep -E 'INFO\s+epoch=' "`$TLOG" | tail -n 10
    echo '--- GPU ---'
    nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo '--- tqdm tail ---'
    sudo tail -n 3 /var/log/user-data.log 2>/dev/null || tail -n 3 /var/log/user-data.log
fi
"@

ssh @sshOpts ubuntu@$publicIp $remoteCmd
