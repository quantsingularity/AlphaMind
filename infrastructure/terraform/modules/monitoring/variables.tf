variable "environment" {
  type = string
}
variable "app_name" {
  type = string
}
variable "kms_key_id" {
  type = string
}
variable "log_retention_days" {
  type = number
}
variable "alert_email_addresses" {
  type = list(string)
}
variable "asg_arn" {
  type    = string
  default = ""
}
variable "db_instance_identifier" {
  type    = string
  default = ""
}
