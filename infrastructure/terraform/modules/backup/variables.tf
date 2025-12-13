variable "environment" {
  type = string
}
variable "app_name" {
  type = string
}
variable "kms_key_id" {
  type = string
}
variable "backup_schedule" {
  type = string
}
variable "backup_retention_days" {
  type = number
}
variable "ec2_instance_arns" {
  type    = list(string)
  default = []
}
variable "rds_instance_arn" {
  type    = string
  default = null
}
