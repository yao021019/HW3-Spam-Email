## ADDED Requirements

### Requirement: Spec Generator CLI
The project SHALL provide a small CLI utility named `spec-generator` that scaffolds a valid OpenSpec change directory and minimal spec deltas from a short prompt.

#### Scenario: Generate minimal proposal interactively
- **WHEN** a user runs `spec-generator` and answers prompts for `change-id`, `title`, and `affected capability`
- **THEN** the tool creates `openspec/changes/<change-id>/proposal.md`, `tasks.md`, and a `specs/<capability>/spec.md` file
- **AND** each generated `spec.md` contains a `## ADDED Requirements` section and at least one `#### Scenario:` block

#### Scenario: Non-interactive template generation
- **WHEN** a user runs `spec-generator --id add-my-feature --capability my-cap` (non-interactive)
- **THEN** the tool generates the change directory and files with sensible placeholders for the author to edit

#### Scenario: Validation-ready output
- **WHEN** the generated change is created
- **THEN** running `openspec validate <change-id> --strict` returns zero fatal errors related to formatting and scenario structure (only content warnings may remain)
