# CLAUDE.md
## Change Discipline
- Prefer the smallest safe change that resolves the task.
- Avoid refactoring unrelated code.
- Do not introduce stylistic or structural changes outside the task scope.
- If a change would significantly increase the diff size, propose a plan before proceeding.

## Scope Control
- Only modify files within the user-specified scope.
- If additional files appear necessary, explain why and request confirmation before modifying them.

## Context Control
- Use only the minimum code context needed for the current task.
- Read only files directly relevant to the request.
- Do not scan broad areas of the repository unless necessary.
- When additional context is needed, expand outward incrementally from the directly affected files.
- If the task appears to require many unrelated files or cross-module changes, propose a brief plan before proceeding.

## Plan Before Editing
When the correct implementation or root cause is not obvious:

- Analyze the relevant code first.
- Identify the smallest possible fix.
- Propose a brief plan before modifying files.

The plan should include:
- suspected root cause or goal
- files that need modification
- the minimal change strategy

Only proceed with edits once the plan is clear.

## Large Change Guardrail
If a change affects many files or alters core architecture:

- Stop and propose a plan.
- Explain why the larger change is necessary.
- Wait for confirmation before proceeding.

## Fix Strategy
- Choose the most direct deterministic fix.
- Avoid speculative fixes or broad architectural changes.
- If the correct fix is unclear, propose a plan before modifying code.

## Test Discipline
- If behavior changes, ensure tests cover the change before running the quality gate.
- Prefer adding or updating the smallest relevant test rather than modifying many tests.
- Do not weaken or remove tests to satisfy the quality gate.

## Coverage Discipline
- When coverage gates fail, add tests for the changed behavior rather than lowering coverage.
- Prefer tests that directly exercise modified code paths.

## Dependency Safety
- Avoid introducing new dependencies unless required.
- Prefer existing project dependencies.
- If a dependency change is required, ensure it passes the security audit.

## Migration Safety
- Do not modify existing migrations unless explicitly instructed.
- When schema changes are required, create a new migration.
- Avoid altering migration history.

## Configuration Safety
- Avoid modifying CI, build, or configuration files unless required for the task.
- If configuration changes are necessary, explain the reason before editing them.

## Quality Gate Failure Handling
When `make check` fails:
- Identify the first failing gate.
- Fix the root cause rather than applying broad changes.
- Re-run `make check` after each fix.
- Continue until all gates pass or a concrete blocker is identified.
