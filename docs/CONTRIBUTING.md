# Contributing

Thanks for working on AlphaMind. This guide keeps changes consistent and the tiers in sync.

## Conventions

Four standing conventions apply to all code and content:

- No em dashes.
- No emojis.
- Clean, professional tone.
- When a decision is reasonable to make, make the best choice rather than pausing to ask.

## Setup

Follow [INSTALLATION.md](INSTALLATION.md) to set up the backend and both frontends.

## Before you open a PR

Run the linters and tests for every tier you touched.

```bash
# Backend
cd code/backend && pytest

# Web
cd web-frontend && npm run lint && npm test && npm run build

# Mobile
cd mobile-frontend && npm run lint && npm test
```

## Changing an API shape

The clients depend on exact field names. If you change a response or request shape:

1. Update the router and service.
2. Update `code/backend/tests/test_frontend_contracts.py` to match the new shape.
3. Update the consuming client code and, for the web client, `src/services/api.test.ts`.
4. Update [API.md](API.md).

The contract tests exist specifically so that an accidental shape change fails CI rather than the running app.

## Keeping the two clients in parity

When you add a feature to one client, consider adding it to the other. The web and mobile apps deliberately cover the same domain.

## Documentation

If your change alters behavior, update the relevant file in `docs/`. Keep docs honest: if something is scaffolding or research-only, label it as such in [FEATURE_MATRIX.md](FEATURE_MATRIX.md) rather than describing it as a finished feature.

## Commit and PR style

Write clear, imperative commit messages. In the PR description, note which tiers you changed and how you verified them (the commands above and their results).
