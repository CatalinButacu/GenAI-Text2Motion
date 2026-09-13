param(
    [Parameter(Mandatory = $true)][string]$FirstName,
    [Parameter(Mandatory = $true)][string]$LastName,
    [Parameter(Mandatory = $true)][string]$Email,
    [string]$Region = "germanywestcentral",
    [int]$Cores = 24,
    [string]$Country = "ROU",
    [string]$Language = "en-us",
    [string]$TimeZone = "W. Europe Standard Time",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

$QUOTA_SERVICE = "06bfd9d3-516b-d5c6-5802-169c800dec89"
$QUOTA_PROBLEM = "e12e3d1d-7fa0-af33-c6d0-3c50df9658a3"

$subId = az account show --query id --output tsv
$ticketName = "gpu-quota-a100-$Region-$(Get-Date -Format yyyyMMddHHmm)"

$description = @"
Request: raise two Compute limits in $Region so a single Standard_NC24ads_A100_v4 VM can start.

1. Standard NCADS_A100_v4 Family vCPUs: 0 -> $Cores
2. Total Regional vCPUs: 10 -> $Cores

Both limits currently block the deployment. The regional cap of 10 is below the 24 vCPUs the SKU
requires, so raising the family quota alone is not sufficient.

Entitlement is already in place: az vm list-skus reports Standard_NC24ads_A100_v4 with no
restrictions in $Region (and in 15 other regions), so this is a quota grant, not an entitlement
change. The self-service path was attempted first and refused:

  az quota update --resource-name standardNCADSA100v4Family \
    --scope /subscriptions/$subId/providers/Microsoft.Compute/locations/$Region \
    --limit-object value=$Cores --resource-type dedicated
  -> (QuotaNotAvailableForResource) Request failed.

The same call was refused in swedencentral, so this needs a human-reviewed grant.

Purpose: a master's dissertation experiment (text-to-motion generation). One VM, one region,
time-boxed. The training script enforces a max-hours budget, a heartbeat stall-kill, and
deallocates the VM on every exit path including failure. A subscription budget alert is configured.
Subscription is Pay-As-You-Go with the spending limit off.

If capacity is unavailable in $Region, any of these are acceptable substitutes:
swedencentral, polandcentral, francecentral, italynorth, uksouth, switzerlandnorth.
"@

$problemClassification = "/providers/Microsoft.Support/services/$QUOTA_SERVICE/problemClassifications/$QUOTA_PROBLEM"

if ($WhatIf) {
    Write-Output "=== WhatIf: no ticket will be filed ==="
    Write-Output "subscription : $subId"
    Write-Output "ticket name  : $ticketName"
    Write-Output "region       : $Region"
    Write-Output "cores        : $Cores"
    Write-Output "contact      : $FirstName $LastName <$Email>"
    Write-Output ""
    Write-Output $description
    exit 0
}

$flatDescription = ($description -replace "`r`n", " " -replace "`n", " ") -replace "\\", "" -replace "->", "to" -replace "\s+", " "

$ticketArgs = @(
    "--ticket-name", $ticketName,
    "--title", "Increase NCADSA100v4 family and Total Regional vCPUs to $Cores in $Region",
    "--description", $flatDescription,
    "--problem-classification", $problemClassification,
    "--severity", "minimal",
    "--advanced-diagnostic-consent", "No",
    "--contact-first-name", $FirstName,
    "--contact-last-name", $LastName,
    "--contact-method", "email",
    "--contact-email", $Email,
    "--contact-country", $Country,
    "--contact-language", $Language,
    "--contact-timezone", $TimeZone,
    "--output", "table"
)
az support in-subscription tickets create @ticketArgs

Write-Output ""
Write-Output "Track it with:"
Write-Output "  az support in-subscription tickets show --ticket-name $ticketName --output table"
