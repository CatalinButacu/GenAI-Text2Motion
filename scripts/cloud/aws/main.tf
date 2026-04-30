# =============================================================================
# AWS GPU Training VM — MotionSSM (RVQ + SSM pipeline)
# =============================================================================
# Quickstart:
#   1. Edit terraform.tfvars with your values
#   2. terraform init && terraform apply
#   3. Monitor: ssh ... 'tail -f /var/log/user-data.log'
#   4. Checkpoints auto-upload to S3 and instance self-terminates
#   5. Download: aws s3 sync s3://<bucket>/checkpoints/rvq_tokenizer ./checkpoints/rvq_tokenizer
#                aws s3 sync s3://<bucket>/checkpoints/motion_ssm    ./checkpoints/motion_ssm
# =============================================================================

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

# -----------------------------------------------------------------------------
# AMI — AWS Deep Learning AMI (Ubuntu 22.04, PyTorch + CUDA pre-installed)
# -----------------------------------------------------------------------------
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# -----------------------------------------------------------------------------
# Security Group — SSH only
# -----------------------------------------------------------------------------
resource "aws_security_group" "ssh_only" {
  name        = "dissertation-ssh-only"
  description = "Allow SSH inbound, all outbound"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -----------------------------------------------------------------------------
# IAM — EC2 role: S3 read+write + self-terminate
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ec2_role" {
  name = "dissertation-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ec2_policy" {
  name = "dissertation-ec2-policy"
  role = aws_iam_role.ec2_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Access"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      },
      {
        Sid      = "SelfTerminate"
        Effect   = "Allow"
        Action   = ["ec2:TerminateInstances"]
        Resource = "*"
        Condition = {
          StringEquals = { "ec2:ResourceTag/Project" = "dissertation" }
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "dissertation-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# -----------------------------------------------------------------------------
# EC2 — g4dn.xlarge: T4 GPU 16 GB VRAM, 4 vCPU, 16 GB RAM
# Spot: ~$0.16/hr | On-demand: ~$0.53/hr
# -----------------------------------------------------------------------------
resource "aws_instance" "training_vm" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = "g4dn.xlarge"
  key_name               = var.key_pair_name
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.ssh_only.id]

  root_block_device {
    volume_size           = 80    # GB: OS (DLAMI needs ≥75 GB) + repo + checkpoints
    volume_type           = "gp3"
    delete_on_termination = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
      }
    }
  }

  user_data = templatefile("${path.module}/startup.sh", {
    s3_bucket    = var.s3_bucket
    s3_cache_uri = var.s3_cache_uri
    region       = var.region
    repo_url     = var.repo_url
    data_source  = var.data_source
    epochs_rvq   = var.epochs_rvq
    epochs_ssm   = var.epochs_ssm
    batch_size   = var.batch_size
  })

  tags = {
    Name    = "dissertation-training"
    Project = "dissertation"
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
output "instance_id"   { value = aws_instance.training_vm.id }
output "public_ip"     { value = aws_instance.training_vm.public_ip }

output "ssh_command" {
  value = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_instance.training_vm.public_ip}"
}

output "watch_logs" {
  value = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_instance.training_vm.public_ip} 'tail -f /var/log/user-data.log'"
}

output "download_checkpoints" {
  value = "aws s3 sync s3://${var.s3_bucket}/checkpoints ./checkpoints"
}

output "cost_estimate" {
  value = var.use_spot ? "~$0.16/hr spot (T4 16GB)" : "~$0.53/hr on-demand (T4 16GB)"
}
