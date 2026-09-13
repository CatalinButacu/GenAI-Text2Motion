terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

resource "azurerm_resource_group" "thesis" {
  name     = "${var.prefix}-rg"
  location = var.location
}

resource "azurerm_virtual_network" "thesis" {
  name                = "${var.prefix}-vnet"
  resource_group_name = azurerm_resource_group.thesis.name
  location            = azurerm_resource_group.thesis.location
  address_space       = ["10.42.0.0/16"]
}

resource "azurerm_subnet" "thesis" {
  name                 = "${var.prefix}-subnet"
  resource_group_name  = azurerm_resource_group.thesis.name
  virtual_network_name = azurerm_virtual_network.thesis.name
  address_prefixes     = ["10.42.1.0/24"]
}

resource "azurerm_network_security_group" "thesis" {
  name                = "${var.prefix}-nsg"
  resource_group_name = azurerm_resource_group.thesis.name
  location            = azurerm_resource_group.thesis.location

  security_rule {
    name                       = "ssh-from-operator"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.allowed_ssh_cidr
    destination_address_prefix = "*"
  }
}

resource "azurerm_public_ip" "thesis" {
  name                = "${var.prefix}-ip"
  resource_group_name = azurerm_resource_group.thesis.name
  location            = azurerm_resource_group.thesis.location
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_interface" "thesis" {
  name                = "${var.prefix}-nic"
  resource_group_name = azurerm_resource_group.thesis.name
  location            = azurerm_resource_group.thesis.location

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.thesis.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.thesis.id
  }
}

resource "azurerm_network_interface_security_group_association" "thesis" {
  network_interface_id      = azurerm_network_interface.thesis.id
  network_security_group_id = azurerm_network_security_group.thesis.id
}

resource "azurerm_storage_account" "thesis" {
  name                     = "${replace(var.prefix, "-", "")}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.thesis.name
  location                 = azurerm_resource_group.thesis.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "checkpoints" {
  name               = "checkpoints"
  storage_account_id = azurerm_storage_account.thesis.id
}

resource "azurerm_linux_virtual_machine" "gpu" {
  name                  = "${var.prefix}-vm"
  resource_group_name   = azurerm_resource_group.thesis.name
  location              = azurerm_resource_group.thesis.location
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.thesis.id]
  priority              = "Spot"
  eviction_policy       = "Deallocate"
  max_bid_price         = var.max_bid_price

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(var.ssh_public_key_path)
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = var.os_disk_gb
  }

  source_image_reference {
    publisher = "canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_role_assignment" "vm_writes_checkpoints" {
  scope                = azurerm_storage_account.thesis.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_linux_virtual_machine.gpu.identity[0].principal_id
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "gpu" {
  virtual_machine_id    = azurerm_linux_virtual_machine.gpu.id
  location              = azurerm_resource_group.thesis.location
  enabled               = true
  daily_recurrence_time = var.shutdown_time
  timezone              = var.shutdown_timezone

  notification_settings {
    enabled = false
  }
}
