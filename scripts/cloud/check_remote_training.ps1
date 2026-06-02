# SSH into the live EC2 training instance and print:
#   - latest per-epoch metrics from training.log
#   - current GPU state (memory, temperature, power)
#   - the actively-updating tqdm tail (last few step lines)
#
# Usage from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\cloud\check_remote_training.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\cloud\check_remote_training.ps1 -RunId 20260512-150000
#
# Reads the public IP from `terraform output` so it auto-updates per EC2.

param(
    [string]$RunId = "",   # if empty, picks the latest motion_ssm/<ts>/ dir on the remote
    [int]$Epochs  = 8      # how many recent epoch= lines to show
)

$keyPath = Join-Path $env:USERPROFILE ".ssh/dissertation-eu-west-1.pem"

Push-Location D:\Facultate\dissertation\scripts\cloud\aws
$publicIp = terraform output -raw public_ip 2>$null
Pop-Location

if (-not $publicIp) {
    Write-Error "No public_ip in terraform output. Is the EC2 still running?"
    exit 1
}

Write-Host "Target EC2: $publicIp"

$sshOpts = @(
    "-i", $keyPath,
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes"
)

# Build the remote command (single bash -c so it's one SSH round-trip)
$remoteCmd = @"
set -e
RUN_ID='$RunId'
if [ -z "`$RUN_ID" ]; then
    RUN_ID=`$(ls -t /home/ubuntu/repo/checkpoints/motion_ssm/ 2>/dev/null | head -1)
fi
LOG=/home/ubuntu/repo/checkpoints/motion_ssm/`$RUN_ID/training.log

echo '=== run_id ==='
echo `$RUN_ID

echo '=== latest epochs ==='
if [ -f "`$LOG" ]; then
    grep -E 'INFO\s+epoch=' "`$LOG" | tail -n $Epochs
else
    echo 'no training.log yet (still in setup phase)'
fi

echo '=== GPU ==='
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo '=== tqdm tail (current step) ==='
tail -n 4 /var/log/user-data.log
"@

ssh @sshOpts ubuntu@$publicIp $remoteCmd
