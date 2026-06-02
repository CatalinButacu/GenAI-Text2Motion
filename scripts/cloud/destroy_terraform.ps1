# Tear down ALL terraform-managed AWS resources (IAM role, instance profile,
# security group). Free to keep but cleaner to remove when not actively training.
#
# Safe to re-run later: `terraform apply` will recreate everything if needed.
# S3 bucket and stored objects are NOT touched (managed manually, not via tf).

$ErrorActionPreference = "Stop"

Push-Location D:\Facultate\dissertation\scripts\cloud\aws

try {
    Write-Host "--- terraform state before destroy ---"
    terraform state list

    Write-Host ""
    Write-Host "--- terraform destroy -auto-approve ---"
    terraform destroy -auto-approve -no-color

    Write-Host ""
    Write-Host "--- state after ---"
    terraform state list 2>&1
}
finally {
    Pop-Location
}
