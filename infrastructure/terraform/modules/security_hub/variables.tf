variable "environment" {
  type = string
}
variable "app_name" {
  type = string
}
variable "enable_aws_foundational_standard" {
  type    = bool
  default = true
}
variable "enable_pci_dss_standard" {
  type    = bool
  default = true
}
variable "enable_cis_standard" {
  type    = bool
  default = true
}
