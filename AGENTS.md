# Vedex Agent Instructions

## Quality requirements

After each feature or meaningful refactor:

- Run focused tests, then the full `pytest` suite.
- Run `ruff check .`, `ruff format --check .`, and `basedpyright`.
- Run the relevant mocked or live integration check for CLI, model, environment, or benchmark changes.

## Reference projects

- [mini-SWE-agent repository](https://github.com/SWE-agent/mini-swe-agent)
