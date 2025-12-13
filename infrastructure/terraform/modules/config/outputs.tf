output "config_recorder_id" {
  description = "ID of the Config recorder"
  value       = aws_config_configuration_recorder.main.id
}

output "config_bucket_name" {
  description = "Name of the Config S3 bucket"
  value       = aws_s3_bucket.config.bucket
}
