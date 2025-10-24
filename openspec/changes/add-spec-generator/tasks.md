## 1. Scaffolding and repository changes
- [ ] Create `openspec/changes/add-spec-generator/proposal.md` (this file)
- [ ] Create `openspec/changes/add-spec-generator/tasks.md` (this file)
- [ ] Create delta spec in `openspec/changes/add-spec-generator/specs/spec-generator/spec.md`

## 2. Implementation (follow-up change)
- [ ] Implement `scripts/spec-generator.js` (or `bin/spec-generator`) that can scaffold a change directory
- [ ] Add unit tests for generator output (formatting and required headers)
- [ ] Add integration test: generate proposal + run `openspec validate <generated-change-id>` to ensure validity
- [ ] Add docs: `README.md` for the generator usage

## 3. CI
- [ ] Add GitHub Actions workflow to run `openspec validate` on generated change (optional future step)

## 4. Acceptance
- [ ] Ensure generated templates pass `openspec validate --strict`
- [ ] Merge implementation change and update `openspec/project.md` if additional conventions are introduced
