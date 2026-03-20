# AlphaMind Infrastructure

## Overview

This infrastructure directory contains comprehensive security and compliance configurations for the AlphaMind quantitative trading system. The infrastructure has been specifically designed to meet financial industry standards including PCI DSS, GDPR, SOX, NIST Cybersecurity Framework, and ISO 27001 requirements.

## Directory Structure

```
infrastructure/
├── README.md                          # This comprehensive guide
├── ansible/                           # Configuration management
│   ├── inventory/
│   │   └── hosts.yml                  # Environment-specific hosts
│   ├── playbooks/
│   │   └── main.yml                   # Main deployment playbook
│   └── roles/
│       ├── common/                    # Common system configurations
│       ├── database/                  # Database security hardening
│       ├── webserver/                 # Web server security
│       └── security/                  # Comprehensive security role
│           ├── tasks/main.yml         # Security tasks implementation
│           ├── handlers/main.yml      # Security service handlers
│           ├── defaults/main.yml      # Security default variables
│           └── templates/             # Security configuration templates
├── kubernetes/                        # Container orchestration
│   ├── base/                          # Base Kubernetes manifests
│   │   ├── app-configmap.yaml         # application configuration
│   │   ├── app-secrets.yaml           # Encrypted secrets management
│   │   ├── backend-deployment.yaml    # Backend service deployment
│   │   ├── database-service.yaml      # Database service configuration
│   │   ├── frontend-deployment.yaml   # Frontend service deployment
│   │   ├── ingress.yaml               # Ingress configuration
│   │   ├── network-policies.yaml      # Network microsegmentation
│   │   ├── pod-security-policy.yaml   # Pod security policies
│   │   ├── rbac.yaml                  # Role-based access control
│   │   └── redis-*.yaml               # Redis configuration files
│   └── environments/                  # Environment-specific configs
│       ├── dev/values.yaml
│       ├── staging/values.yaml
│       └── prod/values.yaml
└── terraform/                         # Infrastructure as Code
    ├── main.tf                        # main configuration
    ├── variables.tf                   # Comprehensive variable definitions
    ├── outputs.tf                     # Infrastructure outputs
    ├── environments/                  # Environment-specific variables
    │   ├── dev/terraform.tfvars
    │   ├── staging/terraform.tfvars
    │   └── prod/terraform.tfvars
    └── modules/                       # Reusable infrastructure modules
        ├── backup/                    # Disaster recovery and backup
        ├── cloudtrail/                # Audit logging (SOX compliance)
        ├── compute/                   # EC2 instances and Auto Scaling
        ├── config/                    # AWS Config for compliance monitoring
        ├── database/                  # RDS with encryption and backup
        ├── guardduty/                 # Threat detection
        ├── monitoring/                # CloudWatch monitoring and alerting
        ├── network/                   # VPC, subnets, and network security
        ├── security/                  # Security groups, WAF, and IAM
        ├── security_hub/              # Centralized security findings
        └── storage/                   # S3 with encryption and lifecycle
```

## Security

### Network Security

- **Network Segmentation**: Separate subnets for web, application, and database tiers
- **Network Policies**: Kubernetes network policies for microsegmentation
- **VPC Flow Logs**: Network traffic monitoring and analysis
- **WAF Protection**: Web Application Firewall with rate limiting
- **Security Groups**: Least-privilege network access controls

### Data Protection

- **Encryption at Rest**: AES-256 encryption for all data storage
- **Encryption in Transit**: TLS 1.2+ for all communications
- **Key Management**: AWS KMS with automatic key rotation
- **Database Encryption**: RDS encryption with performance insights
- **Backup Encryption**: Encrypted backups with 7-year retention

### Access Control

- **Multi-Factor Authentication**: MFA required for all access
- **Role-Based Access Control**: RBAC implementation across all systems
- **Least Privilege**: Minimal required permissions for all accounts
- **Session Management**: Automatic session timeout and monitoring
- **API Security**: API key management and rate limiting

### Monitoring and Logging

- **Comprehensive Audit Trails**: All actions logged and monitored
- **Real-time Alerting**: Immediate notification of security events
- **Log Retention**: 7-year log retention for financial compliance
- **SIEM Integration**: Security Information and Event Management ready
- **Compliance Reporting**: Automated compliance status reporting

### Vulnerability Management

- **Automated Scanning**: Regular vulnerability assessments
- **Patch Management**: Automated security updates
- **Container Security**: Image scanning and runtime protection
- **Configuration Compliance**: Continuous configuration monitoring
- **Penetration Testing**: Regular security testing framework

## Deployment Instructions

### Prerequisites

- Terraform >= 1.0.0
- Ansible >= 2.9
- kubectl >= 1.20
- AWS CLI configured with appropriate permissions
- Docker for container operations

### Environment Setup

1. **Configure AWS Credentials**

   ```bash
   aws configure
   # Ensure your AWS credentials have necessary permissions for:
   # - EC2, RDS, S3, IAM, KMS, CloudTrail, GuardDuty, Config, Security Hub
   ```

2. **Set Environment Variables**
   ```bash
   export ENVIRONMENT=prod  # or dev, staging
   export AWS_REGION=us-west-2
   export APP_NAME=alphamind
   ```

### Terraform Deployment

1. **Initialize Terraform**

   ```bash
   cd terraform/environments/prod
   terraform init
   ```

2. **Plan Infrastructure**

   ```bash
   terraform plan -var-file="terraform.tfvars" -out=tfplan
   ```

3. **Apply Infrastructure**
   ```bash
   terraform apply tfplan
   ```

### Ansible Configuration

1. **Update Inventory**

   ```bash
   # Edit ansible/inventory/hosts.yml with your server IPs
   ```

2. **Deploy Security Configuration**
   ```bash
   cd ansible
   ansible-playbook -i inventory/hosts.yml playbooks/main.yml
   ```

### Kubernetes Deployment

1. **Apply Base Configuration**

   ```bash
   cd kubernetes
   kubectl apply -f base/
   ```

2. **Environment-Specific Configuration**
   ```bash
   # For production
   kubectl apply -f environments/prod/
   ```

---

## Quick Start Deployment Guide

### Prerequisites

Install the following tools:

```bash
# Terraform >= 1.6.0
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# kubectl >= 1.28.0
curl -LO "https://dl.k8s.io/release/v1.28.0/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Ansible >= 2.15.0
pip install ansible ansible-lint yamllint

# AWS CLI
pip install awscli
aws configure
```

### Local Development Setup

1. **Terraform Local Validation**

```bash
cd infrastructure/terraform

# Create your variables file from example
cp environments/dev/terraform.tfvars.example environments/dev/terraform.tfvars
# Edit terraform.tfvars with your values (DO NOT commit)

# Format code
terraform fmt -recursive

# Initialize (local state for development)
terraform init -backend=false

# Validate configuration
terraform validate

# Plan (with your variables)
terraform plan -var-file=environments/dev/terraform.tfvars
```

2. **Kubernetes Local Validation**

```bash
cd infrastructure/kubernetes

# Create secrets from example
cp base/app-secrets.example.yaml base/app-secrets.yaml
# Edit app-secrets.yaml with base64 encoded values (DO NOT commit)

# Validate YAML
yamllint base/

# Dry-run apply
kubectl apply --dry-run=client -f base/

# Apply to cluster (when ready)
kubectl apply -f base/
```

3. **Ansible Local Validation**

```bash
cd infrastructure/ansible

# Install required collections
ansible-galaxy collection install -r requirements.yml

# Create inventory from example
cp inventory/hosts.example.yml inventory/hosts.yml
# Edit with your server IPs (DO NOT commit)

# Validate syntax
ansible-playbook --syntax-check playbooks/main.yml

# Run with check mode (dry-run)
ansible-playbook -i inventory/hosts.yml --check playbooks/main.yml

# Run for real (when ready)
ansible-playbook -i inventory/hosts.yml playbooks/main.yml
```

### Running All Validations

```bash
cd infrastructure
./validate.sh
```

This will check:

- ✓ Required tools installed
- ✓ Terraform format, init, validate
- ✓ Kubernetes YAML lint and dry-run
- ✓ Ansible lint and syntax check
- ✓ No committed secrets

### Tool Versions Used

- Terraform: 1.6.0
- kubectl: 1.28.0
- Ansible: 2.15.0
- Python: 3.11
- yamllint: 1.33.0
- ansible-lint: 6.20.0
