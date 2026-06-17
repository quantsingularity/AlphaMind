# Scripts Reference

AlphaMind ships shell helpers in `scripts/` for common tasks. Run them from the repository root unless noted. Review a script before running it in a new environment.

| Script                 | Purpose                                                       |
| :--------------------- | :------------------------------------------------------------ |
| `setup_environment.sh` | Prepare a development environment (dependencies and tooling). |
| `start_dev.sh`         | Start the local development stack.                            |
| `run_alphamind.sh`     | Launch the application.                                       |
| `run_tests.sh`         | Run the test suites.                                          |
| `test_components.sh`   | Run component-level tests.                                    |
| `lint_code.sh`         | Run linters across the codebase.                              |
| `build.sh`             | Build the project artifacts.                                  |
| `docker_build.sh`      | Build Docker images.                                          |
| `db_migrate.sh`        | Apply database migrations (Alembic).                          |
| `deploy_automation.sh` | Deployment automation.                                        |
| `release.sh`           | Release tasks.                                                |

There is no bespoke `alphamind` command-line binary; the backend is started with `python -m main` from `code/backend`, and the frontends with their npm scripts.

## Common npm scripts

Web frontend (`web-frontend`):

| Command           | Action                                  |
| :---------------- | :-------------------------------------- |
| `npm run dev`     | Start the Vite dev server on port 3000. |
| `npm run build`   | Production build.                       |
| `npm run preview` | Preview the production build.           |
| `npm test`        | Run Vitest.                             |
| `npm run lint`    | Run ESLint.                             |

Mobile frontend (`mobile-frontend`):

| Command             | Action                              |
| :------------------ | :---------------------------------- |
| `npm start`         | Start Expo (then press w, a, or i). |
| `npm run web`       | Start Expo web.                     |
| `npm run build:web` | Export the web build.               |
| `npm test`          | Run Jest.                           |
| `npm run lint`      | Run ESLint.                         |
