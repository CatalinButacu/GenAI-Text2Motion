variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "key_pair_name" {
  description = "EC2 Key Pair name (EC2 Console -> Key Pairs)"
  type        = string
  default     = "dissertation-eu-west-1"
}

variable "use_spot" {
  description = "Use spot instance (~70% cheaper, can be interrupted)"
  type        = bool
  default     = true
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type. Spot prices vary by region:
      g4dn.xlarge  T4   16GB  ~$0.12-0.28/hr  (cheapest, slowest, available everywhere)
      g6.xlarge    L4   24GB  ~$0.24-0.39/hr  (best perf/$, NOT in eu-west-1)
      g5.xlarge    A10G 24GB  ~$0.45-0.74/hr  (fastest, also pricier)
  EOT
  type    = string
  default = "g4dn.xlarge"
}

variable "s3_bucket" {
  description = "S3 bucket for data cache and checkpoint upload"
  type        = string
}

variable "s3_cache_uri" {
  description = "s3:// URI of the preprocessed AMASS joblib cache (.pkl)"
  type        = string
}

variable "repo_url" {
  description = "HTTPS URL of the dissertation git repo to clone on EC2"
  type        = string
}

variable "data_source" {
  description = "Training data source: amass | humanml3d | unified | mega"
  type        = string
  default     = "unified"

  validation {
    condition     = contains(["amass", "humanml3d", "unified", "mega"], var.data_source)
    error_message = "data_source must be amass | humanml3d | unified | mega."
  }
}

variable "epochs_rvq" {
  description = "RVQ tokenizer training epochs (runs first)"
  type        = number
  default     = 50
}

variable "epochs_ssm" {
  description = "MotionSSM training epochs (runs after RVQ)"
  type        = number
  default     = 200
}

variable "batch_size" {
  description = "Batch size (g4dn.xlarge T4 16GB handles 64 safely)"
  type        = number
  default     = 64
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH. Lock to your IP: -var='allowed_ssh_cidr=YOUR.IP/32'"
  type        = string
  default     = "0.0.0.0/0"
}
