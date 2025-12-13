output "instance_ids" {
  description = "IDs of the EC2 instances"
  value       = aws_autoscaling_group.app.id
}

output "instance_public_ips" {
  description = "Public IPs of the EC2 instances"
  value       = aws_lb.app.dns_name
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.app.dns_name
}

output "asg_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = try(aws_autoscaling_group.main[0].arn, "")
}

output "instance_arns" {
  description = "ARNs of EC2 instances"
  value       = []
}
