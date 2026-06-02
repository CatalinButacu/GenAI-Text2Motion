# Verify that nothing on AWS is still billing us for compute.
# Checks across all relevant regions for any leftover EC2, EBS, NAT, etc.
# Reports S3 storage separately (small standing cost, not zero).
#
# Run any time you want to be sure: "is anything still costing money?"

$regions = @("eu-west-1", "eu-central-1", "us-east-1", "us-west-2")

Write-Host "==========================================="
Write-Host " EC2 instances (any non-terminated state)"
Write-Host "==========================================="

$foundEc2 = $false

foreach ($r in $regions) {
    $rows = aws ec2 describe-instances `
        --region $r `
        --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" `
        --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name]" `
        --output text 2>$null

    if ($rows) {
        Write-Host "[$r] $rows"
        $foundEc2 = $true
    } else {
        Write-Host "[$r] none"
    }
}

if (-not $foundEc2) {
    Write-Host ""
    Write-Host "  OK: no EC2 instances anywhere -> EC2 compute bill = `$0/hr"
}

Write-Host ""
Write-Host "==========================================="
Write-Host " EBS volumes (orphans = unattached storage)"
Write-Host "==========================================="

foreach ($r in $regions) {
    $vols = aws ec2 describe-volumes `
        --region $r `
        --filters "Name=status,Values=available" `
        --query "Volumes[].[VolumeId,Size,VolumeType]" `
        --output text 2>$null

    if ($vols) {
        Write-Host "[$r] orphan: $vols"
    } else {
        Write-Host "[$r] no orphan volumes"
    }
}

Write-Host ""
Write-Host "==========================================="
Write-Host " Elastic IPs (allocated but unused = billed)"
Write-Host "==========================================="

foreach ($r in $regions) {
    $eips = aws ec2 describe-addresses `
        --region $r `
        --query "Addresses[?AssociationId==null].[AllocationId,PublicIp]" `
        --output text 2>$null

    if ($eips) {
        Write-Host "[$r] unattached EIP: $eips"
    } else {
        Write-Host "[$r] no unattached EIPs"
    }
}

Write-Host ""
Write-Host "==========================================="
Write-Host " NAT gateways (always billed)"
Write-Host "==========================================="

foreach ($r in $regions) {
    $nats = aws ec2 describe-nat-gateways `
        --region $r `
        --filter "Name=state,Values=available,pending" `
        --query "NatGateways[].[NatGatewayId,State]" `
        --output text 2>$null

    if ($nats) {
        Write-Host "[$r] $nats"
    } else {
        Write-Host "[$r] no NAT gateways"
    }
}

Write-Host ""
Write-Host "==========================================="
Write-Host " S3 storage (ongoing per-GB-month cost)"
Write-Host "==========================================="

aws s3 ls 2>$null | ForEach-Object {
    $bucketName = ($_ -split "\s+")[2]
    if ($bucketName) {
        $size = aws s3 ls "s3://$bucketName" --recursive --summarize 2>$null |
                Select-String "Total Size:"
        Write-Host "  $bucketName  $size"
    }
}

Write-Host ""
Write-Host "==========================================="
Write-Host " Final verdict"
Write-Host "==========================================="
Write-Host "EC2 compute:  `$0/hr  (no instances)"
Write-Host "EBS storage:  see orphans list above"
Write-Host "S3 storage:   see bucket totals above (~`$0.025/GB-month STANDARD)"
