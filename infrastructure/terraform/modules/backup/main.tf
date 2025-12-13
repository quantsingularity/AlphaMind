resource "aws_backup_vault" "main" {
  name        = "${var.app_name}-${var.environment}-vault"
  kms_key_arn = var.kms_key_id
}

resource "aws_backup_plan" "main" {
  name = "${var.app_name}-${var.environment}-plan"

  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = var.backup_schedule

    lifecycle {
      delete_after = var.backup_retention_days
    }
  }
}

resource "aws_iam_role" "backup" {
  name = "${var.app_name}-${var.environment}-backup-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "backup.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

resource "aws_backup_selection" "main" {
  name         = "${var.app_name}-${var.environment}-selection"
  plan_id      = aws_backup_plan.main.id
  iam_role_arn = aws_iam_role.backup.arn

  resources = concat(
    var.ec2_instance_arns != null ? var.ec2_instance_arns : [],
    var.rds_instance_arn != null ? [var.rds_instance_arn] : []
  )
}
