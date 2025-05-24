#!/bin/bash

# AlphaMind Project - Release Orchestration Script

# Define colors for output
COLOR_RESET=\e[0m
COLOR_GREEN=\e[32m
COLOR_RED=\e[31m
COLOR_YELLOW=\e[33m
COLOR_BLUE=\e[34m
COLOR_CYAN=\e[36m

# --- Configuration --- 
# Optional: Set to true to enable automatic deployment after successful build
AUTO_DEPLOY=false 
# Optional: Set to true to enable automatic Git tagging after successful build
AUTO_TAG=false
# Optional: Specify the Git remote name for pushing tags
GIT_REMOTE_NAME="origin"

# --- Helper Functions --- 

# Function to check if a command exists locally
command_exists_local() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print section headers
print_header() {
    echo -e "\n${COLOR_BLUE}==================================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE} $1 ${COLOR_RESET}"
    echo -e "${COLOR_BLUE}==================================================${COLOR_RESET}"
}

# Function to print success messages
print_success() {
    echo -e "${COLOR_GREEN}[SUCCESS] $1${COLOR_RESET}"
}

# Function to print error messages
print_error() {
    echo -e "${COLOR_RED}[ERROR] $1${COLOR_RESET}" >&2
}

# Function to print warning messages
print_warning() {
    echo -e "${COLOR_YELLOW}[WARNING] $1${COLOR_RESET}"
}

# Function to print info messages
print_info() {
    echo -e "${COLOR_CYAN}[INFO] $1${COLOR_RESET}"
}

# Function to run a script and check its exit code
run_script() {
    local script_path="$1"
    local script_name="$(basename "$script_path")"
    
    if [ ! -f "$script_path" ]; then
        print_error "Script 
\"$script_name\" not found at 
\"$script_path\"."
        exit 1
    fi
    
    if [ ! -x "$script_path" ]; then
        print_warning "Script 
\"$script_name\" is not executable. Attempting to make it executable..."
        chmod +x "$script_path"
        if [ $? -ne 0 ]; then
            print_error "Failed to make script 
\"$script_name\" executable."
            exit 1
        fi
    fi
    
    print_info "Executing script: $script_name..."
    # Execute the script
    "$script_path"
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        print_error "Script 
\"$script_name\" failed with exit code $exit_code."
        exit 1 # Stop the release process if any script fails
    else
        print_success "Script 
\"$script_name\" executed successfully."
    fi
}

# --- Initialization --- 

# Exit immediately if a command exits with a non-zero status.
set -e

# Define project root directory (assuming the script is in the project root)
PROJECT_ROOT="$(pwd)"

# Define paths to the other scripts
TEST_SCRIPT="$PROJECT_ROOT/run_all_tests.sh"
LINT_SCRIPT="$PROJECT_ROOT/lint_format_enhanced.sh"
BUILD_SCRIPT="$PROJECT_ROOT/build.sh"
DEPLOY_SCRIPT="$PROJECT_ROOT/deploy.sh"

# --- Prerequisite Checks --- 
print_header "Performing Prerequisite Checks for Release Script"

CHECKS_PASSED=1
# Check if dependent scripts exist
if [ ! -f "$TEST_SCRIPT" ]; then
    print_error "Test script 
\"run_all_tests.sh\" not found."
    CHECKS_PASSED=0
fi
if [ ! -f "$LINT_SCRIPT" ]; then
    print_error "Lint script 
\"lint_format_enhanced.sh\" not found."
    CHECKS_PASSED=0
fi
if [ ! -f "$BUILD_SCRIPT" ]; then
    print_error "Build script 
\"build.sh\" not found."
    CHECKS_PASSED=0
fi
if [ "$AUTO_DEPLOY" = true ] && [ ! -f "$DEPLOY_SCRIPT" ]; then
    print_error "Deployment script 
\"deploy.sh\" not found, but AUTO_DEPLOY is set to true."
    CHECKS_PASSED=0
fi
# Check for Git if tagging is enabled
if [ "$AUTO_TAG" = true ] && ! command_exists_local git; then
    print_error "Git command not found locally, but AUTO_TAG is set to true."
    CHECKS_PASSED=0
fi

if [ $CHECKS_PASSED -eq 0 ]; then
    print_error "Prerequisite checks failed. Please ensure all required scripts exist and tools are installed."
    exit 1
else
    print_success "Prerequisite checks passed."
fi

# --- Release Workflow --- 
print_header "Starting AlphaMind Release Process"

# 1. Run Tests
print_header "Step 1: Running All Tests"
run_script "$TEST_SCRIPT"

# 2. Check Code Quality
print_header "Step 2: Checking Code Quality"
run_script "$LINT_SCRIPT"

# 3. Build Project
print_header "Step 3: Building Project Artifacts"
run_script "$BUILD_SCRIPT"

# 4. Git Tagging (Optional)
if [ "$AUTO_TAG" = true ]; then
    print_header "Step 4: Tagging Release in Git"
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_error "There are uncommitted changes in the working directory. Please commit or stash them before tagging."
        exit 1
    fi
    
    # Prompt for tag version
    read -p "Enter the tag version (e.g., v1.0.0): " TAG_VERSION
    if [ -z "$TAG_VERSION" ]; then
        print_error "Tag version cannot be empty."
        exit 1
    fi
    
    # Check if tag already exists
    if git rev-parse "$TAG_VERSION" >/dev/null 2>&1; then
        print_error "Tag 
\"$TAG_VERSION\" already exists."
        exit 1
    fi
    
    print_info "Creating Git tag: $TAG_VERSION"
    git tag -a "$TAG_VERSION" -m "Release $TAG_VERSION"
    
    print_info "Pushing tag to remote 
\"$GIT_REMOTE_NAME\"..."
    git push "$GIT_REMOTE_NAME" "$TAG_VERSION"
    
    print_success "Git tag 
\"$TAG_VERSION\" created and pushed successfully."
else
    print_info "Step 4: Git Tagging skipped (AUTO_TAG is false)."
fi

# 5. Deployment (Optional)
if [ "$AUTO_DEPLOY" = true ]; then
    print_header "Step 5: Deploying Release"
    print_warning "Ensure the deployment script 
\"$DEPLOY_SCRIPT\" is correctly configured for the target environment!"
    # Add a confirmation step?
    read -p "Proceed with deployment? (y/N): " CONFIRM_DEPLOY
    if [[ "$CONFIRM_DEPLOY" =~ ^[Yy]$ ]]; then
        run_script "$DEPLOY_SCRIPT"
    else
        print_warning "Deployment aborted by user."
    fi
else
    print_info "Step 5: Deployment skipped (AUTO_DEPLOY is false)."
fi

# --- Final Summary --- 
print_header "AlphaMind Release Process Summary"
print_success "Release process completed successfully!"
if [ "$AUTO_TAG" = true ]; then
    print_success "Release tagged as: $TAG_VERSION"
fi
if [ "$AUTO_DEPLOY" = true ] && [[ "$CONFIRM_DEPLOY" =~ ^[Yy]$ ]]; then
    print_success "Release deployed."
elif [ "$AUTO_DEPLOY" = true ]; then
     print_warning "Deployment was configured but aborted by user."
fi

exit 0

