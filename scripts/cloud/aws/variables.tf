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

variable "wandb_api_key" {
  description = <<-EOT
    Optional W&B API key for live online loss curves at https://wandb.ai.
    Get yours at https://wandb.ai/authorize. Leave empty to keep offline mode
    (logs to disk only). Marked sensitive so terraform never prints it.
  EOT
  type      = string
  sensitive = true
  default   = ""
}

# --- Text-encoder choice (see doc/planning/10_TEXT_ENCODER_AND_CFG.md) ---
variable "text_encoder" {
  description = <<-EOT
    Which text encoder to use (snake-case alias, passed to --text-encoder).
    Empty string keeps the script's default (currently sbert-small).
      sbert-small  -> all-MiniLM-L6-v2  (384-d, 22M params, historical baseline)
      sbert-mpnet  -> all-mpnet-base-v2 (768-d, 110M)
      clip-b       -> clip-ViT-B-32     (512-d, OpenAI CLIP -- best motion verbs)
      clip-l       -> clip-ViT-L-14     (768-d, larger CLIP)
  EOT
  type        = string
  default     = ""

  validation {
    condition     = contains(["", "sbert-small", "sbert-mpnet", "clip-b", "clip-l"], var.text_encoder)
    error_message = "text_encoder must be one of sbert-small|sbert-mpnet|clip-b|clip-l or empty."
  }
}

# --- Decoder architecture (see doc/planning/10_TEXT_ENCODER_AND_CFG.md) ---
variable "ar_k_head" {
  description = <<-EOT
    Use the autoregressive K-codebook head (ResidualKHead). Forces training
    from scratch -- the legacy independent-head checkpoints cannot be loaded
    onto this layer set. Expected +0.1-0.3 nats on token CE per the 2026-05
    technique audit (MoMask/Mogo reading).
  EOT
  type        = bool
  default     = false
}

# --- Speed flag ---
variable "compile_model" {
  description = <<-EOT
    Wrap the model in torch.compile(mode='reduce-overhead', dynamic=False).
    Expected 3-5x training throughput on GPU; first batch compiles for 30-90s.
    Ignored on CPU. NOT YET BENCHMARKED at scale on g4dn.xlarge / T4.
  EOT
  type        = bool
  default     = false
}
