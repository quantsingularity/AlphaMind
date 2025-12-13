#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "AlphaMind Infrastructure Validation"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

# Track overall status
OVERALL_STATUS=0

# ============================================================================
# 1. Check Required Tools
# ============================================================================
echo "[1/5] Checking required tools..."

REQUIRED_TOOLS=("terraform" "kubectl" "ansible" "yamllint" "ansible-lint")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if command_exists "$tool"; then
        version=$($tool --version 2>&1 | head -n1)
        print_status 0 "$tool installed: $version"
    else
        print_status 1 "$tool NOT installed"
        OVERALL_STATUS=1
    fi
done
echo ""

# ============================================================================
# 2. Validate Terraform
# ============================================================================
echo "[2/5] Validating Terraform..."

cd terraform

# Format check
if terraform fmt -check -recursive > /dev/null 2>&1; then
    print_status 0 "Terraform format check passed"
else
    print_status 1 "Terraform format check failed"
    OVERALL_STATUS=1
fi

# Init without backend
if terraform init -backend=false > /tmp/tf_init.log 2>&1; then
    print_status 0 "Terraform init successful"
else
    print_status 1 "Terraform init failed (see /tmp/tf_init.log)"
    OVERALL_STATUS=1
fi

# Validate
if terraform validate > /tmp/tf_validate.log 2>&1; then
    print_status 0 "Terraform validate successful"
else
    print_status 1 "Terraform validate failed (see /tmp/tf_validate.log)"
    OVERALL_STATUS=1
    cat /tmp/tf_validate.log
fi

cd ..
echo ""

# ============================================================================
# 3. Validate Kubernetes
# ============================================================================
echo "[3/5] Validating Kubernetes..."

cd kubernetes

# YAML lint
if yamllint base/ > /tmp/k8s_yamllint.log 2>&1; then
    print_status 0 "Kubernetes yamllint passed"
else
    print_status 1 "Kubernetes yamllint found issues (see /tmp/k8s_yamllint.log)"
    # Don't fail on yamllint warnings
fi

# Dry-run
if kubectl apply --dry-run=client -f base/ > /tmp/k8s_dryrun.log 2>&1; then
    print_status 0 "Kubernetes dry-run successful"
else
    print_status 1 "Kubernetes dry-run failed (see /tmp/k8s_dryrun.log)"
    OVERALL_STATUS=1
fi

cd ..
echo ""

# ============================================================================
# 4. Validate Ansible
# ============================================================================
echo "[4/5] Validating Ansible..."

cd ansible

# Install collections
if [ -f "requirements.yml" ]; then
    ansible-galaxy collection install -r requirements.yml > /tmp/ansible_collections.log 2>&1
fi

# Ansible lint
if ansible-lint . > /tmp/ansible_lint.log 2>&1; then
    print_status 0 "ansible-lint passed"
else
    print_status 1 "ansible-lint found issues (see /tmp/ansible_lint.log)"
    # Don't fail on ansible-lint warnings for now
fi

# Syntax check
if ansible-playbook --syntax-check playbooks/main.yml > /tmp/ansible_syntax.log 2>&1; then
    print_status 0 "Ansible syntax check passed"
else
    print_status 1 "Ansible syntax check failed (see /tmp/ansible_syntax.log)"
    OVERALL_STATUS=1
fi

cd ..
echo ""

# ============================================================================
# 5. Check for Secrets
# ============================================================================
echo "[5/5] Checking for committed secrets..."

# Check for common secret patterns
SECRET_PATTERNS=(
    "password.*=.*['\"][^'\"]{8,}['\"]"
    "secret.*=.*['\"][^'\"]{8,}['\"]"
    "api[_-]key.*=.*['\"][^'\"]{8,}['\"]"
)

SECRETS_FOUND=0
for pattern in "${SECRET_PATTERNS[@]}"; do
    if grep -r -E "$pattern" terraform/ kubernetes/ ansible/ 2>/dev/null | grep -v ".example" | grep -v "#"; then
        SECRETS_FOUND=1
    fi
done

if [ $SECRETS_FOUND -eq 0 ]; then
    print_status 0 "No obvious secrets found in code"
else
    print_status 1 "Potential secrets found in code"
    OVERALL_STATUS=1
fi

echo ""
echo "================================================"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}All validations passed!${NC}"
else
    echo -e "${RED}Some validations failed. Please review the logs.${NC}"
fi
echo "================================================"

exit $OVERALL_STATUS
