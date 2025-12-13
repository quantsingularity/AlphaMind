output "app_security_group_id" {
  description = "ID of the application security group"
  value       = try(aws_security_group.app[0].id, "")
}

output "db_security_group_id" {
  description = "ID of the database security group"
  value       = try(aws_security_group.db[0].id, "")
}

output "monitoring_security_group_id" {
  description = "ID of the monitoring security group"
  value       = try(aws_security_group.monitoring[0].id, "")
}
