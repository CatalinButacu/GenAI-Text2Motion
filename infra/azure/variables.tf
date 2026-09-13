variable "subscription_id" {
  type = string
}

variable "prefix" {
  type    = string
  default = "thesis-gpu"
}

variable "location" {
  type    = string
  default = "germanywestcentral"
}

variable "vm_size" {
  type    = string
  default = "Standard_NC24ads_A100_v4"
}

variable "admin_username" {
  type    = string
  default = "azureuser"
}

variable "ssh_public_key_path" {
  type = string
}

variable "allowed_ssh_cidr" {
  type = string
}

variable "max_bid_price" {
  type    = number
  default = -1
}

variable "os_disk_gb" {
  type    = number
  default = 128
}

variable "shutdown_time" {
  type    = string
  default = "2300"
}

variable "shutdown_timezone" {
  type    = string
  default = "GTB Standard Time"
}
