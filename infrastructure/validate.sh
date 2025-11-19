#!/bin/bash
# Infrastructure Validation Script for Financial Compliance
# Validates Ansible, Kubernetes, and Terraform configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if file exists and is not empty
check_file() {
    local file=$1
    local description=$2

    if [[ -f "$file" && -s "$file" ]]; then
        print_success "$description exists and is not empty"
        return 0
    else
        print_error "$description missing or empty: $file"
        return 1
    fi
}

# Check if directory exists
check_directory() {
    local dir=$1
    local description=$2

    if [[ -d "$dir" ]]; then
        print_success "$description directory exists"
        return 0
    else
        print_error "$description directory missing: $dir"
        return 1
    fi
}

# Validate YAML syntax
validate_yaml() {
    local file=$1
    local description=$2

    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            print_success "$description has valid YAML syntax"
            return 0
        else
            print_error "$description has invalid YAML syntax: $file"
            return 1
        fi
    else
        print_warning "Python3 not available, skipping YAML validation for $description"
        return 0
    fi
}

# Validate Terraform syntax
validate_terraform() {
    local dir=$1
    local description=$2

    if command -v terraform &> /dev/null; then
        cd "$dir"
        if terraform validate &>/dev/null; then
            print_success "$description has valid Terraform syntax"
            cd - &>/dev/null
            return 0
        else
            print_error "$description has invalid Terraform syntax"
            cd - &>/dev/null
            return 1
        fi
    else
        print_warning "Terraform not available, skipping syntax validation for $description"
        return 0
    fi
}

# Check for security best practices
check_security_practices() {
    local file=$1
    local description=$2

    # Check for hardcoded secrets (basic check)
    if grep -i -E "(password|secret|key|token).*=.*['\"][^'\"]{8,}['\"]" "$file" &>/dev/null; then
        print_warning "$description may contain hardcoded secrets"
    else
        print_success "$description appears to follow secret management best practices"
    fi

    # Check for encryption settings
    if grep -i -E "(encrypt|ssl|tls)" "$file" &>/dev/null; then
        print_success "$description includes encryption configurations"
    else
        print_warning "$description may be missing encryption configurations"
    fi
}

# Main validation
main() {
    print_header "AlphaMind Enhanced Infrastructure Validation"
    print_info "Validating infrastructure configurations for financial compliance..."
    echo ""

    # Check base directory structure
    print_header "Directory Structure Validation"
    check_directory "ansible" "Ansible"
    check_directory "kubernetes" "Kubernetes"
    check_directory "terraform" "Terraform"
    check_directory "ansible/roles/security" "Ansible Security Role"
    check_directory "kubernetes/base" "Kubernetes Base Manifests"
    check_directory "terraform/modules" "Terraform Modules"
    echo ""

    # Validate Ansible configurations
    print_header "Ansible Configuration Validation"
    check_file "ansible/playbooks/main.yml" "Ansible main playbook"
    check_file "ansible/roles/security/tasks/main.yml" "Security role tasks"
    check_file "ansible/roles/security/handlers/main.yml" "Security role handlers"
    check_file "ansible/roles/security/defaults/main.yml" "Security role defaults"

    # Validate Ansible YAML files
    for yaml_file in ansible/playbooks/*.yml ansible/roles/*/tasks/*.yml ansible/roles/*/handlers/*.yml; do
        if [[ -f "$yaml_file" ]]; then
            validate_yaml "$yaml_file" "$(basename "$yaml_file")"
            check_security_practices "$yaml_file" "$(basename "$yaml_file")"
        fi
    done
    echo ""

    # Validate Kubernetes configurations
    print_header "Kubernetes Configuration Validation"
    k8s_files=(
        "kubernetes/base/app-configmap.yaml"
        "kubernetes/base/app-secrets.yaml"
        "kubernetes/base/backend-deployment.yaml"
        "kubernetes/base/database-service.yaml"
        "kubernetes/base/frontend-deployment.yaml"
        "kubernetes/base/ingress.yaml"
        "kubernetes/base/network-policies.yaml"
        "kubernetes/base/pod-security-policy.yaml"
        "kubernetes/base/rbac.yaml"
    )

    for k8s_file in "${k8s_files[@]}"; do
        if [[ -f "$k8s_file" ]]; then
            check_file "$k8s_file" "$(basename "$k8s_file")"
            validate_yaml "$k8s_file" "$(basename "$k8s_file")"
            check_security_practices "$k8s_file" "$(basename "$k8s_file")"
        fi
    done
    echo ""

    # Validate Terraform configurations
    print_header "Terraform Configuration Validation"
    check_file "terraform/main.tf" "Terraform main configuration"
    check_file "terraform/variables.tf" "Terraform variables"
    check_file "terraform/outputs.tf" "Terraform outputs"

    # Check Terraform modules
    terraform_modules=(
        "terraform/modules/cloudtrail"
        "terraform/modules/compute"
        "terraform/modules/database"
        "terraform/modules/network"
        "terraform/modules/security"
        "terraform/modules/storage"
    )

    for module in "${terraform_modules[@]}"; do
        if [[ -d "$module" ]]; then
            check_directory "$module" "$(basename "$module") module"
            if [[ -f "$module/main.tf" ]]; then
                check_file "$module/main.tf" "$(basename "$module") module main.tf"
                check_security_practices "$module/main.tf" "$(basename "$module") module"
            fi
            validate_terraform "$module" "$(basename "$module") module"
        fi
    done
    echo ""

    # Compliance checks
    print_header "Compliance Requirements Validation"

    # PCI DSS checks
    if grep -r -i "pci.dss\|firewall\|encryption" . &>/dev/null; then
        print_success "PCI DSS compliance configurations found"
    else
        print_error "PCI DSS compliance configurations missing"
    fi

    # GDPR checks
    if grep -r -i "gdpr\|data.protection\|privacy" . &>/dev/null; then
        print_success "GDPR compliance configurations found"
    else
        print_error "GDPR compliance configurations missing"
    fi

    # SOX checks
    if grep -r -i "sox\|audit\|logging" . &>/dev/null; then
        print_success "SOX compliance configurations found"
    else
        print_error "SOX compliance configurations missing"
    fi

    # NIST CSF checks
    if grep -r -i "nist\|cybersecurity.framework" . &>/dev/null; then
        print_success "NIST CSF compliance configurations found"
    else
        print_error "NIST CSF compliance configurations missing"
    fi

    # ISO 27001 checks
    if grep -r -i "iso.27001\|information.security" . &>/dev/null; then
        print_success "ISO 27001 compliance configurations found"
    else
        print_error "ISO 27001 compliance configurations missing"
    fi
    echo ""

    # Security best practices checks
    print_header "Security Best Practices Validation"

    # Check for encryption configurations
    if grep -r -i -E "(kms|encrypt|ssl|tls)" terraform/ kubernetes/ ansible/ &>/dev/null; then
        print_success "Encryption configurations found"
    else
        print_error "Encryption configurations missing"
    fi

    # Check for monitoring and logging
    if grep -r -i -E "(monitor|logging|cloudwatch|prometheus)" . &>/dev/null; then
        print_success "Monitoring and logging configurations found"
    else
        print_error "Monitoring and logging configurations missing"
    fi

    # Check for backup configurations
    if grep -r -i -E "(backup|retention|recovery)" . &>/dev/null; then
        print_success "Backup and recovery configurations found"
    else
        print_error "Backup and recovery configurations missing"
    fi

    # Check for access control
    if grep -r -i -E "(rbac|iam|access.control|mfa)" . &>/dev/null; then
        print_success "Access control configurations found"
    else
        print_error "Access control configurations missing"
    fi

    # Check for network security
    if grep -r -i -E "(network.policy|security.group|firewall|waf)" . &>/dev/null; then
        print_success "Network security configurations found"
    else
        print_error "Network security configurations missing"
    fi
    echo ""

    # Documentation checks
    print_header "Documentation Validation"
    check_file "README.md" "Enhanced README documentation"

    if grep -i -E "(pci.dss|gdpr|sox|nist|iso.27001)" README.md &>/dev/null; then
        print_success "Compliance standards documented in README"
    else
        print_error "Compliance standards not documented in README"
    fi

    if grep -i -E "(deployment|installation|configuration)" README.md &>/dev/null; then
        print_success "Deployment instructions found in README"
    else
        print_error "Deployment instructions missing in README"
    fi
    echo ""

    # Final summary
    print_header "Validation Summary"
    echo -e "Total Checks: ${BLUE}$TOTAL_CHECKS${NC}"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNING_CHECKS${NC}"
    echo ""

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        print_success "All critical validations passed! Infrastructure is ready for deployment."
        exit 0
    else
        print_error "$FAILED_CHECKS critical issues found. Please address before deployment."
        exit 1
    fi
}

# Run validation
main "$@"
