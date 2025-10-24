# Project Context

## Purpose
This repository contains the OpenSpec-driven project for the HW3 workspace. The goal is to practice spec-driven development using OpenSpec: proposals, spec deltas, and lightweight design/docs that keep the specification as the single source of truth for behavior.

## Tech Stack
- Node.js (>=18)
- TypeScript for any runtime code (where applicable)
- OpenSpec tooling: `@fission-ai/openspec` (CLI)
- Test runner: Jest (unit + integration tests)
- Linting/formatting: ESLint + Prettier
- CI: GitHub Actions (preferred)

## Project Conventions

### Code Style
- Use Prettier for formatting. Follow the project's `.prettierrc` if present.
- ESLint with recommended rules; prefer explicit typing in TypeScript where helpful.
- File and directory names: kebab-case for repos and change-ids, camelCase for JS/TS identifiers.

### Architecture Patterns
- Small, single-purpose capabilities (one capability per spec folder under `openspec/specs/`).
- Keep implementation small and focused; favor composition over heavy frameworks.

### Testing Strategy
- Unit tests with Jest for pure logic.
- Integration tests for any I/O or filesystem/network interactions.
- Maintain at least 80% coverage on new capabilities where possible.

### Git Workflow
- Branching: protected `main` branch. Create feature branches named `change/<change-id>` or `feat/<short-desc>`.
- Commit messages: follow Conventional Commits (feat:, fix:, docs:, chore:, refactor:, perf:, test:).
- Pull requests: include a link to the OpenSpec change directory (if exists) and a checklist referencing `openspec/changes/<change-id>/tasks.md`.

## Domain Context
- This is an educational / tooling project focused on OpenSpec-driven proposal and spec workflows. The spec files under `openspec/specs/` are the authoritative source for expected behavior.

## Important Constraints
- Keep all behavioral changes expressed as spec deltas under `openspec/changes/`.
- Do not implement behavior before a proposal is reviewed and approved (see `openspec/AGENTS.md` workflow).
- Prefer minimal external dependencies; any new dependency must be justified in the proposal's Impact section.

## External Dependencies
- Node.js and npm (or compatible package manager).
- `@fission-ai/openspec` CLI (recommended globally for maintainers, or in CI).
- CI services (GitHub Actions) for validation and publishing checks.

## Contacts / Owners
- Primary owner: repository maintainer (update when known)
- Secondary owners: reviewers listed in `openspec/changes/*/proposal.md` when relevant

---

If you want, I can (A) add a `.github/workflows/openspec-validate.yml` that runs `openspec validate --strict` on PRs, and (B) add templates for `proposal.md` and `tasks.md` to make creating new changes faster. Which would you like me to do next?
