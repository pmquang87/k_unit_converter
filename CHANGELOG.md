# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Parameter-aware conversion (`--parameters` / `convert(..., parameters=True)`):
  `*PARAMETER` values are rescaled by the dimension of the `&name` fields
  that reference them (the references stay), `*PARAMETER_EXPRESSION` results
  feeding a dimensional field are wrapped as `(expr)*factor`, and detection
  resolves parametrised densities/moduli. Conflicting uses, integer or
  character parameters, undefined parameters, and expression results that
  are inputs of other expressions are refused — as are expressions whose
  inputs are themselves rescaled (double scaling), self-referencing MUTABLE
  redefinitions, names defined both plainly and by expression,
  `*PARAMETER_LOCAL`/duplicate definitions spanning files, and rescaled
  parameters that are also referenced in fields kunit does not scale.
  Inline `<expression>` fields are always refused. `kunit check` gained a
  matching `--parameters` flag; `--verify-roundtrip` reports "skipped" for
  decks with wrapped expressions. Without the flag `*PARAMETER` stays a
  hard stop (now also for the `_LOCAL`/`_MUTABLE`/`_NOECHO` option
  spellings; `*PARAMETER_DUPLICATION` is whitelisted).
- `g-mm-s` preset (EM/MEMS decks; density g/mm³, pressure Pa). It shares
  the pressure unit with `kg-m-s`, so modulus-only decks now detect as
  ambiguous instead of confidently `kg-m-s` — deliberate; pass `--from`.
- Schema: `*AIRBAG_WANG_NEFSKE` (with JETTING / MULTIPLE_JETTING / POP / CM
  options), `*AIRBAG_LINEAR_FLUID`, `*MAT_ADD_AIRBAG_POROSITY_LEAKAGE`,
  `*MAT_VISCOELASTIC` (006), `*MAT_GENERAL_VISCOELASTIC[_MOISTURE]` (076),
  `*MAT_CABLE_DISCRETE_BEAM` (071), `*MAT_OGDEN_RUBBER` (077_O),
  `*MAT_SEATBELT[_2D]` (B01), `*MAT_SPRING_GENERAL_NONLINEAR` (S06),
  `*MAT_GENERAL_NONLINEAR_1DOF/6DOF_DISCRETE_BEAM` (121/119),
  `*MAT_ADD_PERMEABILITY`, `*ELEMENT_SEATBELT`,
  `*ELEMENT_SEATBELT_ACCELEROMETER`, `*SECTION_SEATBELT`,
  `*SECTION_SOLID_SPG`/`_EFG`, `*BOUNDARY_FLUX_SEGMENT`/`_SET`,
  `*DAMPING_PART_STIFFNESS[_SET]`, `*DAMPING_PART_MASS_SET`, `*PART_MOVE`;
  whitelisted `*CONSTRAINED_ADAPTIVITY`, `*DATABASE_CROSS_SECTION_SET[_ID]`,
  `*DATABASE_BINARY_INTFOR_FILE`, `*COMMENT`.

### Changed
- `*MAT_FABRIC` no longer refuses a nonzero FLC/FAC: FLC is the dimensionless
  leakage flow coefficient and FAC's units follow the venting option
  (R16 Vol I p.3-29: OPT 1-4 unit-less, 5-6 s/m, 7-8 a velocity); FVOPT = 0
  defers to the deck's `*AIRBAG_WANG_NEFSKE` OPT and is refused when that
  is ambiguous or when a `*AIRBAG_HYBRID`/`*AIRBAG_PARTICLE` bag is present;
  for X0 = 1 the FLC/FAC curves run over ratios, not time/pressure.
- `*DAMPING_PART_MASS`: the FLAG = 1 scale-factor card is no longer treated
  as another PID/LCID card (its STY value was registered as a curve id).
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
