# AlphaMind Project: Automation Scripts Documentation

This document provides details on the various automation scripts created for the AlphaMind project. These scripts help streamline development workflows, including environment setup, testing, code quality checks, building, deployment, and release orchestration.

## Table of Contents

1.  [Environment Setup (`setup_environment.sh`)](#1-environment-setup-setup_environmentsh)
2.  [Test Runner (`run_all_tests.sh`)](#2-test-runner-run_all_testssh)
3.  [Code Quality (`lint_format_enhanced.sh`)](#3-code-quality-lint_format_enhancedsh)
4.  [Build Script (`build.sh`)](#4-build-script-buildsh)
5.  [Deployment Script (`deploy.sh`)](#5-deployment-script-deploysh)
6.  [Release Orchestration (`release.sh`)](#6-release-orchestration-releasesh)

---

_(Documentation for each script will be added below)_

## 1. Environment Setup (`setup_environment.sh`)

**Purpose:**
This script automates the setup of the development environment for all components of the AlphaMind project. It ensures that all necessary dependencies are installed for the backend, web frontend, and mobile frontend.

**Prerequisites:**

- Python 3.11 (with `pip`)
- Node.js (with `npm`)
- Yarn (can be installed globally via `npm install -g yarn`)

**Functionality:**

1.  **Prerequisite Checks:** Verifies that Python 3.11, pip, npm, and yarn are installed and accessible in the system's PATH.
2.  **Backend Setup:**
    - Creates a Python virtual environment named `venv` in the project root if it doesn't exist.
    - Activates the virtual environment.
    - Installs/updates Python dependencies listed in `backend/requirements.txt` using pip.
    - Deactivates the virtual environment.
3.  **Web Frontend Setup:**
    - Navigates to the `web-frontend` directory.
    - Installs Node.js dependencies listed in `package.json` using `npm install`.
4.  **Mobile Frontend Setup:**
    - Navigates to the `mobile-frontend` directory.
    - Installs Node.js dependencies listed in `package.json` using `yarn install`.

**Usage:**

1.  Place the script in the root directory of the AlphaMind project.
2.  Make the script executable: `chmod +x setup_environment.sh`
3.  Run the script from the project root: `./setup_environment.sh`

**Notes:**

- The script uses color-coded output for better readability.
- It will exit immediately if any prerequisite check fails or if a critical step (like virtual environment creation or dependency installation) encounters an error.
- It assumes the backend virtual environment should be located at the project root (`./venv`).

---

## 2. Test Runner (`run_all_tests.sh`)

**Purpose:**
This script executes the automated tests for all components of the AlphaMind project: the Python backend, the web frontend, and the mobile frontend.

**Prerequisites:**

- Python 3.11 (with `pip`)
- Node.js (with `npm`)
- Yarn
- Project environment should ideally be set up using `setup_environment.sh` first, although this script attempts to install dependencies if `node_modules` or `requirements.txt` are missing/outdated.
- Test frameworks configured: `pytest` for backend, `jest` (or similar, via `npm test`/`yarn test`) for frontends.

**Functionality:**

1.  **Prerequisite Checks:** Verifies that Python 3.11, npm, and yarn are installed.
2.  **Backend Tests:**
    - Activates the backend virtual environment (`./venv`).
    - Installs/updates Python dependencies from `backend/requirements.txt`.
    - Runs `pytest` within the `backend/tests` directory.
    - Deactivates the virtual environment.
3.  **Web Frontend Tests:**
    - Navigates to the `web-frontend` directory.
    - Installs Node.js dependencies using `npm install` if `node_modules` is missing.
    - Runs the test script defined in `package.json` (`npm test`). If the script doesn't exist, it attempts to run `jest` directly if found locally.
4.  **Mobile Frontend Tests:**
    - Navigates to the `mobile-frontend` directory.
    - Installs Node.js dependencies using `yarn install` if `node_modules` is missing.
    - Runs the test script defined in `package.json` (`yarn test`). If the script doesn't exist, it attempts to run `jest` directly if found locally.
5.  **Reporting:** Provides color-coded output for readability and reports the pass/fail status for each component suite and an overall summary.

**Usage:**

1.  Place the script in the root directory of the AlphaMind project.
2.  Make the script executable: `chmod +x run_all_tests.sh`
3.  Run the script from the project root: `./run_all_tests.sh`

**Notes:**

- The script will exit immediately if a prerequisite check fails or if a critical setup step fails.
- It attempts to handle missing dependencies, but running `setup_environment.sh` first is recommended for a clean state.
- Test failures within a component suite will be reported, and the script will exit with a non-zero status if any suite fails.

---

## 3. Code Quality (`lint_format_enhanced.sh`)

**Purpose:**
This script checks the code quality of the AlphaMind project by running linters and formatters for both the Python backend and the JavaScript/TypeScript frontends. It helps maintain code consistency and catch potential issues.

**Prerequisites:**

- Python 3.11 (with `pip`)
- Node.js (with `npm`)
- Yarn
- Project environment should ideally be set up using `setup_environment.sh` first.
- Code quality tools installed or installable:
  - Python: `black`, `isort`, `flake8` (script attempts to install them in the venv if missing).
  - JS/TS: `eslint`, `prettier` (expected to be dev dependencies in `package.json` for frontends).

**Functionality:**

1.  **Prerequisite Checks:** Verifies that Python 3.11, npm, and yarn are installed.
2.  **Backend Checks (Python):**
    - Activates the backend virtual environment (`./venv`).
    - Checks for and attempts to install `black`, `isort`, `flake8` if missing.
    - Runs `black --check` to identify formatting issues.
    - Runs `isort --check-only` to identify import sorting issues.
    - Runs `flake8` to identify linting issues.
    - Deactivates the virtual environment.
3.  **Web Frontend Checks (JS/TS):**
    - Navigates to the `web-frontend` directory.
    - Installs dependencies (`npm install`) if `node_modules` is missing.
    - Attempts to run `npm run lint` and `npm run format:check` scripts if they exist in `package.json`.
    - If standard scripts don't exist, it attempts to run `eslint` and `prettier --check` directly using locally installed binaries (`./node_modules/.bin/...`).
4.  **Mobile Frontend Checks (JS/TS):**
    - Navigates to the `mobile-frontend` directory.
    - Installs dependencies (`yarn install`) if `node_modules` is missing.
    - Attempts to run `yarn run lint` and `yarn run format:check` scripts if they exist in `package.json`.
    - If standard scripts don't exist, it attempts to run `eslint` and `prettier --check` directly using locally installed binaries.
5.  **Reporting:** Provides color-coded output and a final summary indicating the pass/fail/skipped status for each component.

**Usage:**

1.  Place the script in the root directory of the AlphaMind project.
2.  Make the script executable: `chmod +x lint_format_enhanced.sh`
3.  Run the script from the project root: `./lint_format_enhanced.sh`

**Notes:**

- This script performs **checks** only; it does **not** automatically fix formatting or linting issues. You need to run the formatters (e.g., `black .`, `isort .`, `npm run format`, `yarn run format`) or fix linting errors manually based on the script's output.
- The script attempts to install missing Python tools but assumes frontend tools are project dependencies.
- It will exit with a non-zero status if any checks fail.

---

## 4. Build Script (`build.sh`)

**Purpose:**
This script automates the process of creating production-ready builds or packages for the different components of the AlphaMind project. The output artifacts are placed in a central `build` directory within the project root.

**Prerequisites:**

- Python 3.11
- Node.js (with `npm`)
- Yarn
- `npx` (usually installed with npm)
- Project environment should ideally be set up using `setup_environment.sh` first.
- Web frontend should have a `build` script in `package.json` (e.g., `npm run build`).
- Mobile frontend should be an Expo project capable of web export (`expo export`).

**Functionality:**

1.  **Prerequisite Checks:** Verifies that Python 3.11, npm, and yarn are installed.
2.  **Prepare Build Directory:** Removes any existing `build` directory in the project root and creates a fresh one.
3.  **Web Frontend Build:**
    - Navigates to the `web-frontend` directory.
    - Installs dependencies (`npm install`) if needed.
    - Runs `npm run build` if the script exists.
    - Copies the contents of the build output directory (assumed to be `dist` or `build`) into `./build/web`.
4.  **Mobile Frontend Build (Web Export):**
    - Navigates to the `mobile-frontend` directory.
    - Installs dependencies (`yarn install`) if needed.
    - Runs `npx expo export --platform web --output-dir web-build` to generate a static web version.
    - Copies the contents of the `web-build` directory into `./build/mobile`.
5.  **Backend Packaging:**
    - Copies relevant backend source directories (`ai_models`, `alpha_research`, etc.) and `requirements.txt` into `./build/backend`.
    - Excludes tests, cache files (`__pycache__`, `*.pyc`), and the virtual environment.
    - (Note: Does not create a Python wheel package by default, but copies source files for deployment).
6.  **Reporting:** Provides color-coded output for readability and indicates the success or failure of each step.

**Usage:**

1.  Place the script in the root directory of the AlphaMind project.
2.  Make the script executable: `chmod +x build.sh`
3.  Run the script from the project root: `./build.sh`

**Output:**

- The script creates a `build` directory in the project root containing subdirectories:
  - `build/web`: Contains the built static assets for the web frontend.
  - `build/mobile`: Contains the exported static web build for the mobile frontend.
  - `build/backend`: Contains the packaged backend source code and requirements file.

**Notes:**

- The script will exit immediately if any prerequisite check or build step fails.
- The web frontend build relies on a `build` script being defined in its `package.json`.
- The mobile frontend build specifically performs a web export using Expo.
- The backend packaging copies source files; for more robust deployment, consider adding steps to build a wheel package.

---

## 5. Deployment Script (`deploy.sh`)

**Purpose:**
This script provides a **template** for automating the deployment of the built AlphaMind application artifacts (from the `build` directory) to a target server environment. It uses a release directory strategy with symbolic links for atomic switching and easy rollbacks.

**Prerequisites:**

- **Local:**
  - `ssh` client installed.
  - `scp` command installed (usually part of the SSH client).
  - The `build` directory must exist (created by running `build.sh`).
  - SSH access configured to the target server (e.g., via SSH keys).
- **Remote (Target Server):**
  - SSH server running.
  - User account with necessary permissions to create directories, manage files, and potentially restart services.
  - Required runtime environments (e.g., Python 3.x for backend, web server like Nginx for frontend).

**Configuration (CRITICAL):**

- **You MUST edit the script before first use.**
- Update the configuration variables at the top:
  - `TARGET_HOST`: SSH destination (e.g., `user@your_server_ip`).
  - `TARGET_BASE_DIR`: The root directory on the server where deployments will reside (e.g., `/var/www/alphamind`).
  - `SSH_KEY_PATH` (Optional): Path to your private SSH key if not using the default.
- **Customize Remote Steps:** The script contains placeholder comments for remote setup, configuration, and service restarts (e.g., installing Python dependencies, running migrations, configuring Nginx, restarting Gunicorn/systemd services). **You MUST replace these placeholders with the actual commands specific to your server environment and application stack.**

**Functionality:**

1.  **Prerequisite Checks:** Verifies local `ssh`, `scp` commands and the existence of the `build` directory. Tests SSH connectivity to the `TARGET_HOST`.
2.  **Remote Directory Setup:** Creates base deployment directories (`releases`, `shared`) on the target server if they don't exist.
3.  **Create Release Directory:** Creates a new timestamped directory within `releases` on the target server (e.g., `/path/to/deploy/alphamind/releases/20250503201100`).
4.  **Transfer Artifacts:** Copies the contents of the local `build` subdirectories (`backend`, `web`, `mobile`) to the newly created release directory on the target server using `scp`.
5.  **Remote Setup (Placeholders):** Includes commented-out example commands for setting up the backend (creating venv, installing dependencies, migrations) and configuring the frontend (web server setup). **These need customization.**
6.  **Activate Release:** Atomically switches the live deployment by updating a symbolic link (`current`) to point to the new release directory.
7.  **Restart Services (Placeholders):** Includes commented-out example commands for restarting backend services (e.g., Gunicorn/systemd). **These need customization.**
8.  **Cleanup (Optional):** Removes older release directories, keeping a configurable number of recent releases (`RELEASES_TO_KEEP`, default is 3).

**Usage (After Configuration):**

1.  Ensure the `build.sh` script has been run successfully.
2.  **Carefully configure** the variables and customize the remote commands within `deploy.sh`.
3.  Place the script in the root directory of the AlphaMind project.
4.  Make the script executable: `chmod +x deploy.sh`
5.  Run the script from the project root: `./deploy.sh`

**Notes:**

- This script is a template and **will not work** without proper configuration and customization of the remote commands.
- It uses `scp` for file transfer. For more advanced scenarios, `rsync` might be preferred if available on both local and remote machines.
- The release directory and symlink strategy allows for near-zero downtime deployments and easy rollbacks (by manually changing the `current` symlink to a previous release directory).
- Error handling is included (`set -e`), stopping the script on most failures.

---

## 6. Release Orchestration (`release.sh`)

**Purpose:**
This script orchestrates the entire release process for the AlphaMind project by sequentially running the test, code quality, and build scripts. It also includes optional steps for Git tagging and triggering the deployment script, providing a semi-automated release pipeline.

**Prerequisites:**

- All prerequisite tools for the individual scripts (`python3.11`, `npm`, `yarn`, `pip`, `ssh`, `scp`, etc.).
- The following scripts must exist in the project root directory and ideally be executable:
  - `run_all_tests.sh`
  - `lint_format_enhanced.sh`
  - `build.sh`
  - `deploy.sh` (only required if `AUTO_DEPLOY` is set to `true`).
- `git` command installed locally (only required if `AUTO_TAG` is set to `true`).
- A clean Git working directory (no uncommitted changes) if `AUTO_TAG` is enabled.

**Configuration (Optional):**

- Edit the variables at the top of the script:
  - `AUTO_DEPLOY`: Set to `true` to automatically run `deploy.sh` after a successful build (requires user confirmation during execution). Default is `false`.
  - `AUTO_TAG`: Set to `true` to automatically tag the release in Git after a successful build. Default is `false`.
  - `GIT_REMOTE_NAME`: The name of the Git remote to push tags to (e.g., `origin`). Default is `origin`.

**Functionality:**

1.  **Prerequisite Checks:** Verifies that the required dependent scripts (`run_all_tests.sh`, `lint_format_enhanced.sh`, `build.sh`, and `deploy.sh` if `AUTO_DEPLOY=true`) exist. Checks for `git` if `AUTO_TAG=true`.
2.  **Run Tests:** Executes `run_all_tests.sh`. If it fails, the release process stops.
3.  **Check Code Quality:** Executes `lint_format_enhanced.sh`. If it fails, the release process stops.
4.  **Build Project:** Executes `build.sh`. If it fails, the release process stops.
5.  **Git Tagging (Optional):** If `AUTO_TAG=true`:
    - Checks for uncommitted Git changes.
    - Prompts the user to enter a tag version (e.g., `v1.0.1`).
    - Creates an annotated Git tag.
    - Pushes the tag to the configured remote repository.
6.  **Deployment (Optional):** If `AUTO_DEPLOY=true`:
    - Warns the user that `deploy.sh` must be configured.
    - Prompts the user for confirmation before proceeding.
    - Executes `deploy.sh` if confirmed.
7.  **Reporting:** Provides color-coded output for each step and a final summary of the release process.

**Usage:**

1.  Ensure all prerequisite scripts are in the project root and executable.
2.  (Optional) Configure `AUTO_TAG` and `AUTO_DEPLOY` variables in `release.sh`.
3.  Place the script in the root directory of the AlphaMind project.
4.  Make the script executable: `chmod +x release.sh`
5.  Run the script from the project root: `./release.sh`

**Notes:**

- The script is designed to stop immediately (`set -e`) if any of the core steps (testing, linting, building) fail, preventing a broken release.
- Git tagging requires user input for the version number.
- Deployment requires user confirmation if enabled.
- Ensure the `deploy.sh` script is fully configured and tested before enabling `AUTO_DEPLOY` in `release.sh`.
