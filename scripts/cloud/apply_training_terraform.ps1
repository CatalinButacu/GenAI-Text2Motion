# Spin up the training EC2 by calling terraform apply.
#
# What it does:
#   1. cd into scripts/cloud/aws/ where the .tf files live
#   2. taint any tracked aws_instance.training_vm so apply re-creates with the
#      latest startup.sh (terraform won't re-render user_data without a taint)
#   3. terraform apply -auto-approve
#   4. Print the outputs (instance_id, public_ip, ssh_command)
#
# Costs ~$0.53/hour on g4dn.xlarge until either training self-terminates or
# you run scripts\cloud\destroy_training_terraform.ps1.

$ErrorActionPreference = "Stop"

Push-Location D:\Facultate\dissertation\scripts\cloud\aws

try {
    Write-Host "--- terraform state list (before) ---"
    terraform state list

    # Re-taint instance if it's in state, so apply re-creates with current startup.sh.
    $haveInstance = terraform state list | Where-Object { $_ -eq "aws_instance.training_vm" }

    if ($haveInstance) {
        Write-Host ""
        Write-Host "--- tainting aws_instance.training_vm to force fresh user_data ---"
        terraform taint aws_instance.training_vm
    } else {
        Write-Host "(no instance in state; apply will create fresh)"
    }

    Write-Host ""
    Write-Host "--- terraform apply -auto-approve ---"
    terraform apply -auto-approve -no-color

    Write-Host ""
    Write-Host "--- key outputs ---"
    terraform output
}
finally {
    Pop-Location
}
