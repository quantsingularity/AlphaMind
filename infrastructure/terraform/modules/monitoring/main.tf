resource "aws_cloudwatch_log_group" "main" {
  name              = "/aws/${var.app_name}/${var.environment}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_id
}

resource "aws_sns_topic" "alerts" {
  name              = "${var.app_name}-${var.environment}-alerts"
  kms_master_key_id = var.kms_key_id
}

resource "aws_sns_topic_subscription" "email_alerts" {
  count     = length(var.alert_email_addresses)
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email_addresses[count.index]
}
