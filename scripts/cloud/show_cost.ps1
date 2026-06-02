# Compute current AWS spend for the active training run, based on instance
# uptime x on-demand hourly price. Also breaks down EBS + S3 estimates.
#
# Pricing as of mid-2026 (eu-west-1):
#   g4dn.xlarge on-demand : $0.526/hr
#   gp3 EBS storage       : $0.0832/GB-month
#   S3 STANDARD storage   : $0.0245/GB-month

param(
    [double]$EC2HourlyUsd = 0.526,
    [int]$EbsGB           = 80,
    [string]$Bucket       = "dissertation-motion-cache-91340264"
)

Push-Location D:\Facultate\dissertation\scripts\cloud\aws
$instanceId = terraform output -raw instance_id 2>$null
Pop-Location

if (-not $instanceId) {
    Write-Host "No instance_id in terraform output."
    exit 0
}

# Get launch time + state from AWS
$row = aws ec2 describe-instances `
    --instance-ids $instanceId `
    --region eu-west-1 `
    --query "Reservations[].Instances[].{state:State.Name,launch:LaunchTime}" `
    --output text 2>$null

# row is two tab-separated values: launch_time state
$parts = $row -split "`t"
if ($parts.Count -lt 2) {
    Write-Host "Failed to parse describe-instances output: $row"
    exit 1
}

$launchTimeStr = $parts[0]
$state         = $parts[1]

$launch    = [datetime]::Parse($launchTimeStr).ToUniversalTime()
$nowUtc    = [datetime]::UtcNow
$uptimeHrs = ($nowUtc - $launch).TotalHours

$ec2Cost      = $uptimeHrs * $EC2HourlyUsd
$ebsCostSoFar = $EbsGB * 0.0832 * ($uptimeHrs / (24.0 * 30))

Write-Host "Instance:    $instanceId"
Write-Host "State:       $state"
Write-Host "Launched:    $launch (UTC)"
Write-Host "Now:         $nowUtc (UTC)"
Write-Host ("Uptime:      {0:F2} hours" -f $uptimeHrs)
Write-Host ""
Write-Host "--- Cost so far (approximate) ---"
Write-Host ("EC2 compute:   $ {0:F2}  (${1:F3}/hr x {2:F1}h)" -f $ec2Cost, $EC2HourlyUsd, $uptimeHrs)
Write-Host ("EBS 80 GB:     $ {0:F2}" -f $ebsCostSoFar)
Write-Host ("Total so far:  $ {0:F2}" -f ($ec2Cost + $ebsCostSoFar))
Write-Host ""

# S3 size (one-shot lookup)
$s3Out = aws s3 ls "s3://$Bucket/" --recursive --summarize 2>$null |
    Select-String "Total Size:"
Write-Host "--- S3 storage (running cost ~ \$0.025/GB/month) ---"
Write-Host $s3Out
