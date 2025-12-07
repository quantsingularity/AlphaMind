#!/bin/bash
# AlphaMind - Database Migration Script

# Define colors for output
COLOR_RESET="\e[0m"
COLOR_GREEN="\e[32m"
COLOR_RED="\e[31m"
COLOR_YELLOW="\e[33m"
COLOR_BLUE="\e[34m"
COLOR_CYAN="\e[36m"

# --- Helper Functions ---

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to print section headers
print_header() {
  echo -e "\n${COLOR_BLUE}===========================================================${COLOR_RESET}"
  echo -e "${COLOR_BLUE} $1 ${COLOR_RESET}"
  echo -e "${COLOR_BLUE}===========================================================${COLOR_RESET}"
}

# Function to print success messages
print_success() {
  echo -e "${COLOR_GREEN}[SUCCESS] $1${COLOR_RESET}"
}

# Function to print error messages
print_error() {
  echo -e "${COLOR_RED}[ERROR] $1${COLOR_RESET}" >&2
}

# Function to print info messages
print_info() {
  echo -e "${COLOR_CYAN}[INFO] $1${COLOR_RESET}"
}

# --- Initialization ---

# Exit immediately if a command exits with a non-zero status, treat unset variables as an error, and fail if any command in a pipeline fails
set -euo pipefail

# Define project root directory
PROJECT_ROOT="$(pwd)"

# Parse command line arguments
ACTION=""
MESSAGE=""
ENVIRONMENT="development"

while [[ $# -gt 0 ]]; do
  case $1 in
    --action)
      ACTION="$2"
      shift 2
      ;;
    --message)
      MESSAGE="$2"
      shift 2
      ;;
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./db_migrate.sh --action [init|migrate|upgrade|downgrade] [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --action ACTION        Migration action: init, migrate, upgrade, downgrade"
      echo "  --message MESSAGE      Migration message (required for 'migrate')"
      echo "  --env ENVIRONMENT      Environment to run migration against (default: development)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  print_error "Migration action is required. Use --help for usage."
  exit 1
fi

# --- Environment Setup ---

print_header "Setting Up Database Migration Environment"

# Check for Python virtual environment and activate
if [[ -d "venv" ]]; then
  print_info "Activating Python virtual environment..."
  source venv/bin/activate
else
  print_error "Python virtual environment not found. Please run setup_environment.sh first."
  exit 1
fi

# Load environment variables for the specified environment
ENV_FILE="$PROJECT_ROOT/.env.$ENVIRONMENT"
if [[ -f "$ENV_FILE" ]]; then
  print_info "Loading environment variables from $ENV_FILE"
  # Simple way to load variables, assumes key=value format
  while IFS='=' read -r key value; do
    if [[ ! -z "$key" && "$key" != \#* ]]; then
      export "$key"="$value"
    fi
  done < "$ENV_FILE"
else
  print_warning "Environment file $ENV_FILE not found. Using default environment."
fi

# Check for Alembic (or other migration tool)
if ! command_exists alembic; then
  print_error "Alembic not found. Please install it in your virtual environment."
  exit 1
fi

# Change to backend directory where alembic.ini is located
if [[ ! -d "backend" ]]; then
  print_error "Backend directory not found."
  exit 1
fi
cd backend

# --- Run Migration Action ---

print_header "Running Database Migration Action: $ACTION"

case "$ACTION" in
  init)
    print_info "Initializing Alembic environment..."
    alembic init migrations
    print_success "Alembic environment initialized. Review migrations/alembic.ini and migrations/env.py."
    ;;
  migrate)
    if [[ -z "$MESSAGE" ]]; then
      print_error "Migration message is required for 'migrate' action. Use --message."
      exit 1
    fi
    print_info "Creating new migration with message: $MESSAGE"
    alembic revision --autogenerate -m "$MESSAGE"
    print_success "New migration created."
    ;;
  upgrade)
    print_info "Upgrading database to latest revision..."
    alembic upgrade head
    print_success "Database upgrade complete."
    ;;
  downgrade)
    print_info "Downgrading database by one revision..."
    alembic downgrade -1
    print_success "Database downgrade complete."
    ;;
  *)
    print_error "Invalid action: $ACTION. Supported actions: init, migrate, upgrade, downgrade."
    exit 1
    ;;
esac

# Deactivate virtual environment
deactivate

print_success "Database migration script finished."
