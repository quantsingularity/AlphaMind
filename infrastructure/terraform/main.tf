terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    # Backend configuration can be provided via -backend-config flag
    # For local development, you can skip backend initialization
    # Enhanced backend configuration for compliance
    encrypt        = true
    kms_key_id     = "alias/terraform-state-key"
    dynamodb_table = "terraform-state-lock"
    # Will be configured per environment
  }
}

# Configure AWS Provider with enhanced security
provider "aws" {
  region = var.aws_region

  # Default tags for all resources - compliance requirement
  default_tags {
    tags = merge(var.default_tags, {
      "Compliance:PCI-DSS"   = "true"
      "Compliance:GDPR"      = "true"
      "Compliance:SOX"       = "true"
      "Compliance:NIST-CSF"  = "true"
      "Compliance:ISO-27001" = "true"
      "DataClassification"   = "Financial"
      "BackupRequired"       = "true"
      "MonitoringRequired"   = "true"
      "EncryptionRequired"   = "true"
      "AuditRequired"        = "true"
      "ManagedBy"            = "Terraform"
      "LastUpdated"          = timestamp()
    })
  }

  # Assume role for cross-account access if needed
  dynamic "assume_role" {
    for_each = var.assume_role_arn != null ? [1] : []
    content {
      role_arn     = var.assume_role_arn
      session_name = "terraform-${var.environment}"
    }
  }
}

# Data sources for compliance and security
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# KMS Key for encryption at rest - PCI DSS requirement
resource "aws_kms_key" "main" {
  description             = "KMS key for AlphaMind ${var.environment} encryption"
  deletion_window_in_days = var.kms_deletion_window
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow CloudTrail to encrypt logs"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name       = "${var.app_name}-${var.environment}-kms-key"
    Purpose    = "Encryption at rest for financial data"
    Compliance = "PCI-DSS,GDPR,SOX"
  }
}

resource "aws_kms_alias" "main" {
  name          = "alias/${var.app_name}-${var.environment}"
  target_key_id = aws_kms_key.main.key_id
}

# CloudTrail for audit logging - SOX compliance requirement
module "cloudtrail" {
  source = "./modules/cloudtrail"

  environment                   = var.environment
  app_name                      = var.app_name
  kms_key_id                    = aws_kms_key.main.arn
  cloudtrail_bucket_name        = "${var.app_name}-${var.environment}-cloudtrail-${random_id.bucket_suffix.hex}"
  enable_log_file_validation    = true
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_logging                = true

  depends_on = [aws_kms_key.main]
}

# Config for compliance monitoring
module "config" {
  source = "./modules/config"

  environment = var.environment
  app_name    = var.app_name
  kms_key_id  = aws_kms_key.main.arn

  depends_on = [aws_kms_key.main]
}

# GuardDuty for threat detection - NIST CSF requirement
module "guardduty" {
  source = "./modules/guardduty"

  environment = var.environment
  app_name    = var.app_name

  enable_s3_protection         = true
  enable_malware_protection    = true
  enable_kubernetes_protection = true
}

# Security Hub for centralized security findings
module "security_hub" {
  source = "./modules/security_hub"

  environment = var.environment
  app_name    = var.app_name

  enable_aws_foundational_standard = true
  enable_pci_dss_standard          = true
  enable_cis_standard              = true
}

# Network module with enhanced security
module "network" {
  source = "./modules/network"

  environment           = var.environment
  app_name              = var.app_name
  vpc_cidr              = var.vpc_cidr
  availability_zones    = var.availability_zones
  public_subnet_cidrs   = var.public_subnet_cidrs
  private_subnet_cidrs  = var.private_subnet_cidrs
  database_subnet_cidrs = var.database_subnet_cidrs
  kms_key_id            = aws_kms_key.main.arn

  # Enhanced security features
  enable_vpc_flow_logs = true
  enable_dns_hostnames = true
  enable_dns_support   = true
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev" ? true : false

  depends_on = [aws_kms_key.main]
}

# Security module with comprehensive controls
module "security" {
  source = "./modules/security"

  environment = var.environment
  app_name    = var.app_name
  vpc_id      = module.network.vpc_id
  kms_key_id  = aws_kms_key.main.arn

  # Security group configurations
  allowed_cidr_blocks = var.allowed_cidr_blocks
  database_port       = var.database_port
  application_port    = var.application_port

  # WAF configuration
  enable_waf     = true
  waf_rate_limit = var.waf_rate_limit

  depends_on = [module.network, aws_kms_key.main]
}

# Compute module with security hardening
module "compute" {
  source = "./modules/compute"

  environment        = var.environment
  app_name           = var.app_name
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  public_subnet_ids  = module.network.public_subnet_ids
  instance_type      = var.instance_type
  key_name           = var.key_name
  kms_key_id         = aws_kms_key.main.arn

  # Security configurations
  security_group_ids = [
    module.security.app_security_group_id,
    module.security.monitoring_security_group_id
  ]

  # Auto Scaling configuration
  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_desired_capacity

  # Enhanced monitoring
  enable_detailed_monitoring = true

  depends_on = [module.network, module.security, aws_kms_key.main]
}

# Database module with encryption and backup
module "database" {
  source = "./modules/database"

  environment         = var.environment
  app_name            = var.app_name
  vpc_id              = module.network.vpc_id
  database_subnet_ids = module.network.database_subnet_ids
  db_instance_class   = var.db_instance_class
  db_name             = var.db_name
  db_username         = var.db_username
  db_password         = var.db_password
  kms_key_id          = aws_kms_key.main.arn

  # Security configurations
  security_group_ids = [module.security.db_security_group_id]

  # Backup and compliance settings
  backup_retention_period = var.db_backup_retention_period
  backup_window           = var.db_backup_window
  maintenance_window      = var.db_maintenance_window

  # Encryption settings
  storage_encrypted = true

  # Enhanced monitoring
  monitoring_interval          = 60
  performance_insights_enabled = true

  depends_on = [module.network, module.security, aws_kms_key.main]
}

# Storage module with encryption and lifecycle policies
module "storage" {
  source = "./modules/storage"

  environment = var.environment
  app_name    = var.app_name
  kms_key_id  = aws_kms_key.main.arn

  # Compliance settings
  enable_versioning = true
  enable_logging    = true
  lifecycle_rules   = var.s3_lifecycle_rules

  depends_on = [aws_kms_key.main]
}

# Monitoring and alerting module
module "monitoring" {
  source = "./modules/monitoring"

  environment = var.environment
  app_name    = var.app_name
  kms_key_id  = aws_kms_key.main.arn

  # CloudWatch configuration
  log_retention_days = var.log_retention_days

  # SNS topic for alerts
  alert_email_addresses = var.alert_email_addresses

  # Auto Scaling Group ARN for monitoring
  asg_arn = module.compute.asg_arn

  # Database identifier for monitoring
  db_instance_identifier = module.database.db_instance_identifier

  depends_on = [module.compute, module.database, aws_kms_key.main]
}

# Backup module for disaster recovery
module "backup" {
  source = "./modules/backup"

  environment = var.environment
  app_name    = var.app_name
  kms_key_id  = aws_kms_key.main.arn

  # Backup configuration
  backup_schedule       = var.backup_schedule
  backup_retention_days = var.backup_retention_days

  # Resources to backup
  ec2_instance_arns = module.compute.instance_arns
  rds_instance_arn  = module.database.db_instance_arn

  depends_on = [module.compute, module.database, aws_kms_key.main]
}

# Random ID for unique resource naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}
