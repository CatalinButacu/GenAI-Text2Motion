output "public_ip" {
  value = aws_instance.trainer.public_ip
}

output "ssh" {
  value = "ssh -i dissertation_v2.pem ubuntu@${aws_instance.trainer.public_ip}"
}

output "instance_id" {
  value = aws_instance.trainer.id
}

output "cost_guards" {
  value = "hard-terminate in ${var.max_hours}h; idle-GPU terminate after ${var.idle_shutdown_minutes}m; or 'terraform destroy'."
}
