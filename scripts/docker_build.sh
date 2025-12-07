#!/bin/bash
# AlphaMind - Docker Image Build Script

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
IMAGE_NAME="alphamind"
TAG="latest"
BUILD_ARGS=""
NO_CACHE=false
PUSH=false
COMPONENT="all"

while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --build-arg)
      BUILD_ARGS="$BUILD_ARGS --build-arg $2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=true
      shift
      ;;
    --push)
      PUSH=true
      shift
      ;;
    --component)
      COMPONENT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./docker_build.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --name NAME            Base image name (default: alphamind)"
      echo "  --tag TAG              Image tag (default: latest)"
      echo "  --build-arg ARG        Pass a build-arg to Docker (can be used multiple times)"
      echo "  --no-cache             Do not use cache when building the image"
      echo "  --push                 Push the image to the registry after building"
      echo "  --component COMPONENT  Component to build: backend, web-frontend, or all (default: all)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Prerequisite Check ---

print_header "Checking Prerequisites"

if ! command_exists docker; then
  print_error "Docker command not found. Please install Docker."
  exit 1
fi

# --- Build Component Images ---

build_backend_image() {
  print_header "Building Backend Docker Image"

  if [[ ! -f "backend/Dockerfile" ]]; then
    print_warning "backend/Dockerfile not found, skipping backend image build."
    return 0
  fi

  local full_tag="$IMAGE_NAME-backend:$TAG"
  local cache_option=""
  if [[ "$NO_CACHE" == "true" ]]; then
    cache_option="--no-cache"
  fi

  print_info "Building image: $full_tag"
  docker build -f backend/Dockerfile -t "$full_tag" $cache_option $BUILD_ARGS .

  if [[ "$PUSH" == "true" ]]; then
    print_info "Pushing image: $full_tag"
    docker push "$full_tag"
  fi

  print_success "Backend image $full_tag built and optionally pushed."
}

build_web_frontend_image() {
  print_header "Building Web Frontend Docker Image"

  if [[ ! -f "web-frontend/Dockerfile" ]]; then
    print_warning "web-frontend/Dockerfile not found, skipping web frontend image build."
    return 0
  fi

  local full_tag="$IMAGE_NAME-web:$TAG"
  local cache_option=""
  if [[ "$NO_CACHE" == "true" ]]; then
    cache_option="--no-cache"
  fi

  print_info "Building image: $full_tag"
  docker build -f web-frontend/Dockerfile -t "$full_tag" $cache_option $BUILD_ARGS .

  if [[ "$PUSH" == "true" ]]; then
    print_info "Pushing image: $full_tag"
    docker push "$full_tag"
  fi

  print_success "Web frontend image $full_tag built and optionally pushed."
}

# --- Main Execution ---

print_header "Starting Docker Build Process"

if [[ "$COMPONENT" == "all" || "$COMPONENT" == "backend" ]]; then
  build_backend_image
fi

if [[ "$COMPONENT" == "all" || "$COMPONENT" == "web-frontend" ]]; then
  build_web_frontend_image
fi

print_success "Docker build process complete."
