# Vedex Agent Instructions

## Goal

Vedex is a small terminal coding agent. Keep the core simple, readable, and benchmarked with SWE-bench execution.

## Target design

- Use a linear agent loop: query the model, execute Bash, append the observation, and repeat.
- Provide one default model provider using LiteLLM.
- Expose Bash as the only model-visible tool. Remove abstractions and tests that only support read, write, edit, or other tools.
- Support exactly two execution environments: local and Docker.
- Remove the resources, skills, project-context discovery, and related runtime layers; keep prompting and configuration minimal.
- Support SWE-bench as benchmark.
- Prefer mini-SWE-agent's simple, benchmark-oriented design while retaining only the Vedex contracts that directly support these goals.

## Quality requirements

After every feature or meaningful refactor, complete all of the following:

- Run focused Pytest tests while developing.
- Run the full Pytest suite.
- Run Ruff lint and formatting verification.
- Run BasedPyright for the package and tests using the checked-in project configuration.
- Run the relevant mocked or real integration check for CLI, adapter, environment, or benchmark changes.

## Reference projects

- [mini-SWE-agent repository](https://github.com/SWE-agent/mini-swe-agent)
