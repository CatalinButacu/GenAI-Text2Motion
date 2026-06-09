terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Latest AWS Deep Learning base AMI (CUDA + Nvidia driver preinstalled) for the region.
data "aws_ami" "dlami" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.dlami[0].id
}

# Recreate the key pair in AWS from the public half of dissertation_v2.pem (conflict-free name).
resource "aws_key_pair" "thesis" {
  key_name   = "${var.project}-key"
  public_key = file(var.public_key_path)
}

resource "aws_security_group" "ssh" {
  name        = "${var.project}-ssh"
  description = "SSH from a single allowed CIDR"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Project = var.project }
}

# Reach the box via SSM Session Manager (works over IPv6, no inbound SSH) + let it read/write the
# S3 transfer bucket. This is the IPv6-only-client path; the SSH SG stays as an IPv4 fallback.
resource "aws_iam_role" "trainer" {
  name = "${var.project}-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.trainer.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "s3" {
  role = aws_iam_role.trainer.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = ["arn:aws:s3:::${var.bucket}", "arn:aws:s3:::${var.bucket}/*"]
    }]
  })
}

resource "aws_iam_instance_profile" "trainer" {
  name = "${var.project}-profile"
  role = aws_iam_role.trainer.name
}

resource "aws_instance" "trainer" {
  ami                                  = local.ami_id
  instance_type                        = var.instance_type
  key_name                             = aws_key_pair.thesis.key_name
  vpc_security_group_ids               = [aws_security_group.ssh.id]
  iam_instance_profile                 = aws_iam_instance_profile.trainer.name
  instance_initiated_shutdown_behavior = "terminate" # any OS shutdown -> terminate (releases EBS)

  # Spot: cheapest, and our --resume checkpoints make an interruption a non-event.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  root_block_device {
    volume_size = var.volume_gb
    volume_type = "gp3"
  }

  # Cost guards: hard max-lifetime + idle-GPU watchdog (terminates after training ends).
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    max_minutes  = var.max_hours * 60
    idle_minutes = var.idle_shutdown_minutes
  })

  tags = { Project = var.project, Name = "${var.project}-trainer" }
}
