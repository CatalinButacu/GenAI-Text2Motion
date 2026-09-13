param(
    [string]$ResourceGroup = "rg-thesis-t2m",
    [string]$Location = "swedencentral",
    [string]$StorageAccount = "stthesist2m01",
    [string]$KeyVault = "kv-thesis-t2m",
    [string]$AlertEmail = "",
    [int]$MonthlyBudgetEur = 50,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$TAGS = @("project=thesis", "purpose=text2motion", "managed-by=cli")

function Step {
    param([string]$Describe, [scriptblock]$Action)
    Write-Output ""
    Write-Output "--- $Describe"
    if ($WhatIf) {
        Write-Output "    (WhatIf: skipped)"
        return
    }
    & $Action
}

$subId = az account show --query id --output tsv
$rgScope = "/subscriptions/$subId/resourceGroups/$ResourceGroup"
$saScope = "$rgScope/providers/Microsoft.Storage/storageAccounts/$StorageAccount"

Write-Output "subscription   : $subId"
Write-Output "resource group : $ResourceGroup ($Location)"
Write-Output "storage        : $StorageAccount"

Step "blob containers: data (read by trainer), checkpoints (written by trainer), runs (audit trail)" {
    foreach ($c in @("data", "checkpoints", "runs")) {
        az storage container create --name $c --account-name $StorageAccount --auth-mode login --output none
        Write-Output "    container $c ready"
    }
}

Step "user-assigned managed identities: one per job, no secrets to rotate" {
    foreach ($id in @("id-thesis-trainer", "id-thesis-inference", "id-thesis-deployer")) {
        az identity create --name $id --resource-group $ResourceGroup --location $Location --tags @TAGS --output none
        Write-Output "    identity $id ready"
    }
}

Step "RBAC: least privilege, scoped to individual containers not the account" {
    $trainer = az identity show --name id-thesis-trainer --resource-group $ResourceGroup --query principalId --output tsv
    $inference = az identity show --name id-thesis-inference --resource-group $ResourceGroup --query principalId --output tsv

    az role assignment create --assignee-object-id $trainer --assignee-principal-type ServicePrincipal `
        --role "Storage Blob Data Reader" --scope "$saScope/blobServices/default/containers/data" --output none
    Write-Output "    trainer   -> data         : read"

    az role assignment create --assignee-object-id $trainer --assignee-principal-type ServicePrincipal `
        --role "Storage Blob Data Contributor" --scope "$saScope/blobServices/default/containers/checkpoints" --output none
    Write-Output "    trainer   -> checkpoints  : write"

    az role assignment create --assignee-object-id $trainer --assignee-principal-type ServicePrincipal `
        --role "Storage Blob Data Contributor" --scope "$saScope/blobServices/default/containers/runs" --output none
    Write-Output "    trainer   -> runs         : write"

    az role assignment create --assignee-object-id $inference --assignee-principal-type ServicePrincipal `
        --role "Storage Blob Data Reader" --scope "$saScope/blobServices/default/containers/checkpoints" --output none
    Write-Output "    inference -> checkpoints  : read only"
}

Step "key vault: holds endpoint keys so nothing lands in git" {
    az keyvault create --name $KeyVault --resource-group $ResourceGroup --location $Location `
        --enable-rbac-authorization true --sku standard --tags @TAGS --output none
    $inference = az identity show --name id-thesis-inference --resource-group $ResourceGroup --query principalId --output tsv
    az role assignment create --assignee-object-id $inference --assignee-principal-type ServicePrincipal `
        --role "Key Vault Secrets User" `
        --scope "$rgScope/providers/Microsoft.KeyVault/vaults/$KeyVault" --output none
    Write-Output "    $KeyVault ready; inference identity may read secrets"
}

if ($AlertEmail -eq "") {
    Write-Output ""
    Write-Output "--- alerts and budget: SKIPPED (pass -AlertEmail to enable)"
}
else {
    Step "action group: one notification target reused by budget and failure alerts" {
        az monitor action-group create --name ag-thesis-alerts --resource-group $ResourceGroup `
            --short-name thesis --action email primary $AlertEmail --output none
        Write-Output "    ag-thesis-alerts -> $AlertEmail"
    }

    Step "budget: $MonthlyBudgetEur EUR/month, alerts at 50/80/100 percent of forecast and actual" {
        $start = (Get-Date -Day 1).ToString("yyyy-MM-dd")
        $end = (Get-Date -Day 1).AddYears(2).ToString("yyyy-MM-dd")
        az consumption budget create --budget-name budget-thesis --amount $MonthlyBudgetEur `
            --category cost --time-grain Monthly --time-period-start $start --time-period-end $end `
            --output none
        Write-Output "    budget-thesis created at $MonthlyBudgetEur EUR/month"
        Write-Output "    NOTE: attach the action group in the portal, or via the budgets REST API"
    }
}

Write-Output ""
Write-Output "=== provisioning complete ==="
Write-Output "verify with:"
Write-Output "  az resource list --resource-group $ResourceGroup --output table"
Write-Output "  az role assignment list --scope $saScope --output table"
