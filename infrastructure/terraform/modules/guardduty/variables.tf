variable "environment" {
  type = string
}
variable "app_name" {
  type = string
}
variable "enable_s3_protection" {
  type    = bool
  default = true
}
variable "enable_malware_protection" {
  type    = bool
  default = true
}
variable "enable_kubernetes_protection" {
  type    = bool
  default = true
}
