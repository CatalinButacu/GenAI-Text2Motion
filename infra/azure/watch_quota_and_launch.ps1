param(
    [string[]]$Regions = @(
        "germanywestcentral", "swedencentral", "polandcentral",
        "francecentral", "italynorth", "uksouth", "switzerlandnorth"
    ),
    [int]$PollMinutes = 15,
    [int]$MaxHours = 8,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$logDir = "artifacts/logs/azure"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "quota_watch.log"

function Say {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}

function Get-Quota {
    param([string]$Location, [string]$NameContains)
    $val = az vm list-usage --location $Location --query "[?contains(name.value, '$NameContains')].limit | [0]" --output tsv 2>$null
    if ([string]::IsNullOrWhiteSpace($val)) { return -1 }
    return [int]$val
}

function Get-CoresQuota {
    param([string]$Location)
    $val = az vm list-usage --location $Location --query "[?name.value=='cores'].limit | [0]" --output tsv 2>$null
    if ([string]::IsNullOrWhiteSpace($val)) { return -1 }
    return [int]$val
}

Say "quota watcher started -- polling every $PollMinutes min across: $($Regions -join ', ')"

if ($WhatIf) {
    foreach ($region in $Regions) {
        $family = Get-Quota -Location $region -NameContains "NCADSA100v4"
        $cores = Get-CoresQuota -Location $region
        Say "[WhatIf] $region : family=$family cores=$cores"
    }
    Say "[WhatIf] would loop every $PollMinutes min and launch train_on_demand.ps1 -MaxHours $MaxHours on the first region where family>=24 and cores>=24"
    exit 0
}

while ($true) {
    foreach ($region in $Regions) {
        $family = Get-Quota -Location $region -NameContains "NCADSA100v4"
        $cores = Get-CoresQuota -Location $region
        Say "$region : family=$family cores=$cores"
        if ($family -ge 24 -and $cores -ge 24) {
            Say "QUOTA GRANTED in $region -- launching train_on_demand.ps1 (MaxHours=$MaxHours)"
            & .\infra\azure\train_on_demand.ps1 -Location $region -MaxHours $MaxHours
            Say "training run returned -- watcher stopping"
            exit 0
        }
    }
    Start-Sleep -Seconds ($PollMinutes * 60)
}
