#!/bin/bash
# AlphaMind - Development Server Startup Script

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

# --- Environment Setup ---

print_header "Setting Up Development Environment"

# Check for Python virtual environment and activate
if [[ -d "venv" ]]; then
  print_info "Activating Python virtual environment..."
  source venv/bin/activate
else
  print_error "Python virtual environment not found. Please run setup_environment.sh first."
  exit 1
fi

# --- Start Backend Server ---

start_backend() {
  print_header "Starting Backend Development Server"

  if [[ ! -d "backend" ]]; then
    print_error "Backend directory not found, skipping backend start."
    return 1
  fi

  cd backend

  # Check for uvicorn or similar server
  if command_exists uvicorn; then
    print_info "Starting Uvicorn server..."
    # Assuming the main application is in backend/app/main.py and the app object is named 'app'
    # Use reload for development
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  elif command_exists flask; then
    print_info "Starting Flask development server..."
    # Assuming a Flask app
    export FLASK_APP=app.main
    flask run --host 0.0.0.0 --port 8000
  else
    print_error "No suitable backend server (uvicorn, flask) found. Please install one."
    return 1
  fi

  cd "$PROJECT_ROOT"
}

# --- Start Web Frontend Server ---

start_web_frontend() {
  print_header "Starting Web Frontend Development Server"

  if [[ ! -d "web-frontend" ]]; then
    print_error "Web frontend directory not found, skipping frontend start."
    return 1
  fi

  cd web-frontend

  # Check for package.json and start script
  if [[ -f "package.json" ]]; then
    if grep -q "\"start\"" package.json; then
      print_info "Starting web frontend with 'npm start' or 'yarn start'..."
      if [[ -f "yarn.lock" ]]; then
        yarn start
      else
        npm start
      fi
    else
      print_error "No 'start' script found in web-frontend/package.json."
      return 1
    fi
  else
    print_error "No package.json found in web-frontend directory."
    return 1
  fi

  cd "$PROJECT_ROOT"
}

# --- Main Execution ---

print_header "Starting AlphaMind Development Environment"

# Start backend in a background process
start_backend &
BACKEND_PID=$!
print_info "Backend server started with PID: $BACKEND_PID"

# Start web frontend in the foreground (blocking)
start_web_frontend

# Cleanup: Kill the backend process when the frontend process is stopped (e.g., with Ctrl+C)
print_info "Stopping backend server (PID: $BACKEND_PID)..."
kill $BACKEND_PID

print_success "Development environment stopped."
