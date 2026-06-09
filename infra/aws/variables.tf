variable "project" {
  type    = string
  default = "thesis-t2m"
}

variable "region" {
  type        = string
  description = "AWS region to launch in (use the one where you'll keep data/snapshots)."
  default     = "eu-central-1"
}

variable "instance_type" {
  type        = string
  description = "g4dn.xlarge = T4 16GB (cheapest decent GPU); g5.xlarge = A10G 24GB."
  default     = "g4dn.xlarge"
}

variable "ami_id" {
  type        = string
  description = "Override the auto-selected Deep Learning AMI; empty = pick latest for the region."
  default     = ""
}

variable "public_key_path" {
  type        = string
  description = "Public half of dissertation_v2.pem (ssh-keygen -y > this file)."
  default     = "dissertation_v2.pub"
}

variable "allowed_cidr" {
  type        = string
  description = "Your public IP /32 to allow SSH (IPv4 fallback). IPv6-only? use 0.0.0.0/0 (key-only) and reach the box via SSM instead."
  default     = "0.0.0.0/0"
}

variable "bucket" {
  type        = string
  description = "Existing S3 bucket name for the data+results transfer (aws s3 mb s3://<name>)."
}

variable "use_spot" {
  type    = bool
  default = true
}

variable "spot_max_price" {
  type        = string
  description = "Max $/hr for spot; leave at on-demand price as the cap."
  default     = "0.30"
}

variable "volume_gb" {
  type    = number
  default = 125
}

variable "max_hours" {
  type        = number
  description = "Hard self-terminate after this many hours, no matter what (forgotten-instance guard)."
  default     = 36
}

variable "idle_shutdown_minutes" {
  type        = number
  description = "Terminate after the GPU has been idle this long (i.e. training finished)."
  default     = 20
}
