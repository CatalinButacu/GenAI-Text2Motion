output "public_ip" {
  value = azurerm_public_ip.thesis.ip_address
}

output "ssh" {
  value = "ssh ${var.admin_username}@${azurerm_public_ip.thesis.ip_address}"
}

output "storage_account" {
  value = azurerm_storage_account.thesis.name
}

output "checkpoint_container_url" {
  value = "https://${azurerm_storage_account.thesis.name}.blob.core.windows.net/${azurerm_storage_container.checkpoints.name}"
}
