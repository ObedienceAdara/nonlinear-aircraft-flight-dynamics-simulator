# Contributing to Aircraft 6-DOF Flight Dynamics Simulator

Thank you for considering a contribution.

This repository is an engineering-oriented, nonlinear 6-DOF aircraft flight-dynamics framework. Contributions are welcome, especially when they improve the correctness, reproducibility, clarity, or usefulness of the simulator and its validation workflow.

## Before you contribute

Please read:

- [README.md](README.md) for the project scope and fidelity boundary.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current software and physics architecture.
- [docs/EQUATIONS.md](docs/EQUATIONS.md) for sign conventions and equations.
- [docs/VALIDATION.md](docs/VALIDATION.md) for the verification/validation approach.

Please keep in mind that the canonical aircraft model uses generic demonstration parameters. A contribution should not present generic results as validated data for a real aircraft.

## What is useful to contribute?

Examples include:

- bug fixes and regression tests;
- corrections to equations, signs, frames, or numerical methods;
- additional verification cases;
- improvements to trim, simulation, environment, actuator, or reporting infrastructure;
- documented engineering studies built on the canonical model;
- reproducibility improvements;
- documentation and examples;
- validation work against independent analytical, experimental, or reference-simulator results.

For physics changes, please include the reasoning, assumptions, reference material, and tests needed to establish that the change is intentional.

## Development setup

The project targets Python 3.10+.

Create an environment and install the development dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Running the tests

Run the full verification suite before opening a pull request:

```bash
pytest
```

For a focused change, run the relevant test module first, then run the complete suite.

A contribution that changes numerical behavior should normally add or update a regression test.

## Running the simulator

The project-level demonstration can be run with:

```bash
python main.py
```

Studies under `studies/` should be run from the repository root so that their imports and paths match the documented workflow.

## Code and design expectations

Keep the canonical physics implementation under `src/aircraft6dof/`.

Prefer extending an existing model or abstraction over creating a parallel implementation of the same physics.

For example, a study should normally call the canonical simulator, trim machinery, or analysis utilities rather than maintaining a private copy of the equations.

Keep assumptions explicit. If a parameter is:

- generic,
- provisional,
- fitted,
- estimated,
- taken from a published reference, or
- identified from data,

say so in the code or documentation.

Do not silently increase the claimed fidelity of the model.

## Numerical and physics changes

When changing flight-dynamics behavior, please consider the following:

1. State and frame conventions.
2. Units and dimensional consistency.
3. Force and moment sign conventions.
4. Limiting cases and invariants.
5. Numerical sensitivity to timestep or finite-difference step.
6. Regression behavior of existing studies.
7. Whether the result is verification or validation.

For a new aerodynamic or propulsion model, document where the coefficients or maps come from.

For a numerical method change, include an independent check where practical rather than relying only on a plot that looks plausible.

## Tests and studies

Tests should be deterministic where possible.

Studies should clearly separate:

- model assumptions;
- experiment/setup;
- generated results;
- interpretation;
- limitations.

Generated outputs that are intentionally excluded by Git should not be committed merely to make a test pass. When a result artifact is important for reproducibility or review, document how to regenerate it.

## Pull requests

Keep pull requests focused.

A good pull request should explain:

- what changed;
- why it changed;
- which files are affected;
- how it was tested;
- any numerical or physical behavior that changed;
- known limitations or follow-up work.

For physics changes, include representative numerical results when useful.

Please avoid combining unrelated refactors, formatting-only churn, new features, and validation changes in one pull request unless they are genuinely part of the same change.

## Commit messages

Use concise, imperative commit messages.

Examples:

```text
feat: add reusable aircraft trim solver
fix: correct body-to-wind force transformation
test: add quaternion sign-invariance regression
docs: clarify NED gravity convention
```

## Reporting bugs

When reporting a bug, include enough information to reproduce it:

- Python version;
- operating system;
- commit or branch;
- command that was run;
- observed result or error;
- expected result;
- a minimal reproduction where practical.

For numerical bugs, include relevant state, controls, environment, timestep, and configuration when available.

## Scope and safety

This repository is intended for civil, academic, educational, research, and general aerospace simulation work.

Please do not contribute offensive operational guidance, weapon-targeting logic, attack tooling, or code whose primary purpose is to enable harm. Contributions should remain focused on simulation, verification, validation, controls, autonomy research, and general engineering.

## Questions and proposals

For substantial architectural or physics changes, opening an issue before implementing the change is encouraged. This is especially useful when the proposal affects the public API, state conventions, numerical integration, or the interpretation of validation results.

## License

By contributing to this repository, you agree that your contribution is provided under the repository's [MIT License](LICENSE).
