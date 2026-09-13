param(
    [string]$ResourceGroup = "rg-thesis-t2m",
    [string]$Location = "swedencentral",
    [string]$VmName = "vm-thesis-trainer",
    [string]$VmSize = "Standard_NC24ads_A100_v4",
    [string]$StorageAccount = "stthesist2m01",
    [string]$AdminUser = "azureuser",
    [int]$MaxHours = 8,
    [int]$BackstopHours = 12,
    [switch]$KeepVm,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$IMAGE = "microsoft-dsvm:ubuntu-hpc:2204:latest"
$vmCreated = $false

function Say {
    param([string]$Message)
    Write-Output "[$(Get-Date -Format HH:mm:ss)] $Message"
}

function Rescue-Artifacts {
    Say "rescuing artifacts before teardown"
    $script = @"
cd ~/thessis 2>/dev/null || exit 0
azcopy sync artifacts/checkpoints https://$StorageAccount.blob.core.windows.net/checkpoints --recursive 2>&1 | tail -2
azcopy sync artifacts/outputs https://$StorageAccount.blob.core.windows.net/runs --recursive 2>&1 | tail -2
"@
    try {
        az vm run-command invoke --resource-group $ResourceGroup --name $VmName `
            --command-id RunShellScript --scripts $script --output none 2>$null
        Say "artifact sync issued"
    }
    catch {
        Say "WARNING: artifact rescue failed -- VM may already be unreachable"
    }
}

function Teardown {
    if (-not $vmCreated) {
        return
    }
    if ($KeepVm) {
        Say "KeepVm set -- leaving $VmName ALLOCATED. It is billing. Deallocate with:"
        Say "  az vm deallocate --resource-group $ResourceGroup --name $VmName"
        return
    }
    Rescue-Artifacts
    Say "deallocating $VmName (stops compute billing)"
    az vm deallocate --resource-group $ResourceGroup --name $VmName --output none
    Say "deallocated"
}

if ($WhatIf) {
    Write-Output "=== WhatIf ==="
    Write-Output "would create : $VmName ($VmSize) in $ResourceGroup / $Location"
    Write-Output "image        : $IMAGE"
    Write-Output "identity     : id-thesis-trainer (container-scoped RBAC, no account access)"
    Write-Output "max hours    : $MaxHours (script budget)"
    Write-Output "backstop     : auto-shutdown after $BackstopHours h, survives laptop death"
    Write-Output "teardown     : rescue artifacts then deallocate on EVERY exit path"
    exit 0
}

$quota = az vm list-usage --location $Location --query "[?contains(name.value, 'NCADSA100v4')].limit | [0]" --output tsv
if ([int]$quota -lt 24) {
    Say "ABORT: A100 quota in $Location is $quota vCPUs, need 24."
    Say "File the request first: infra/azure/request_gpu_quota.ps1"
    exit 1
}

$identityId = az identity show --name id-thesis-trainer --resource-group $ResourceGroup --query id --output tsv

try {
    Say "creating $VmName ($VmSize)"
    az vm create --resource-group $ResourceGroup --name $VmName --location $Location `
        --size $VmSize --image $IMAGE --admin-username $AdminUser `
        --generate-ssh-keys --assign-identity $identityId `
        --public-ip-sku Standard --nic-delete-option Delete --os-disk-delete-option Delete `
        --tags project=thesis stage=train managed-by=cli --output none
    $vmCreated = $true
    Say "VM created"

    Say "arming auto-shutdown backstop at +$BackstopHours h"
    $shutdownAt = (Get-Date).AddHours($BackstopHours).ToString("HHmm")
    az vm auto-shutdown --resource-group $ResourceGroup --name $VmName --time $shutdownAt --output none
    Say "backstop armed for $shutdownAt UTC -- fires even if this laptop dies"

    Say "starting training (MAX_HOURS=$MaxHours per stage)"
    $train = @"
set -euo pipefail
cd ~/thessis
bash infra/kernel_canary.sh
MAX_HOURS=$MaxHours BLOB=https://$StorageAccount.blob.core.windows.net bash infra/azure/run_twins.sh
"@
    az vm run-command invoke --resource-group $ResourceGroup --name $VmName `
        --command-id RunShellScript --scripts $train --output table

    Say "training returned"
}
catch {
    Say "FAILED: $($_.Exception.Message)"
    throw
}
finally {
    Teardown
}

Say "done. Artifacts in https://$StorageAccount.blob.core.windows.net/checkpoints"
