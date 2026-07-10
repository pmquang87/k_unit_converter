# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyPI-ready packaging: `readme`, `authors`, `keywords`, `classifiers`,
  `[project.urls]`, and a modern SPDX `license = "MIT"` expression.
- Single-source version: `pyproject.toml` reads `__version__` from
  `kunit/__init__.py` via `dynamic = ["version"]`.
- `kunit --version` flag.
- Typing marker `kunit/py.typed` shipped in the wheel (PEP 561).
- `dev` optional dependencies (`coverage`, `ruff`, `mypy`) plus `ruff` and
  `coverage` tool config.
- `examples/` directory with small, detectable sample decks and a guide.
- Project docs: `CONTRIBUTING.md`, this changelog, GitHub issue/PR templates,
  Dependabot config, and a Trusted-Publishing release workflow.

### Changed
- README: library-usage section, Tkinter/OS-package note for the GUI, and a
  non-hardcoded description of the test suite.

## [0.2.0] - 2026-01-01

### Added
- Two-pass converter between any two (mass, length, time) unit systems with
  exact `Fraction`/`Decimal` scaling and field-preserving output.
- Unit-system auto-detection scored against densities, elastic moduli,
  detonation velocities, gravity curves, and header comments.
- `kunit check` coverage/convertibility report (`--json` for CI).
- `*INCLUDE`-tree conversion, blast-load UNIT remapping, self-check and
  `--verify-roundtrip` verification, and a Tkinter GUI.

[Unreleased]: https://github.com/pmquang87/k_unit_converter/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pmquang87/k_unit_converter/releases/tag/v0.2.0
