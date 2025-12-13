resource "aws_db_subnet_group" "main" {
  name       = "${var.db_name}-${var.environment}-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name        = "${var.db_name}-${var.environment}-subnet-group"
    Environment = var.environment
  }
}

resource "aws_db_instance" "main" {
  identifier              = "${var.db_name}-${var.environment}"
  allocated_storage       = 20
  storage_type            = "gp2"
  engine                  = "mysql"
  engine_version          = "8.0"
  instance_class          = var.db_instance_class
  db_name                 = var.db_name
  username                = var.db_username
  password                = var.db_password
  parameter_group_name    = "default.mysql8.0"
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = var.security_group_ids
  skip_final_snapshot     = true
  multi_az                = var.environment == "prod" ? true : false
  backup_retention_period = var.environment == "prod" ? 7 : 1

  tags = {
    Name        = "${var.db_name}-${var.environment}"
    Environment = var.environment
  }
}
