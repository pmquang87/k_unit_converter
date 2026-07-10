# kunit — Roadmap / "what to do next"

> **Status (this branch):** Tiers 0–3 and most of Tier 4 are now implemented
> (see the CHANGELOG). Test suite grew from 96 → 197. The **one deliberately
> deferred item** is the invasive `Ctx` god-object split (Tier 4) — it rewrites
> ~40 handler call sites for internal-structure reasons with real regression
> risk and no functional payoff, so it belongs in its own isolated, reviewed PR
> rather than bundled with feature/fix work. Everything else below is done.

A synthesis of a deep review of the codebase (engine, schema, tests, tooling)
and the repo's history. `kunit` is already mature: exact-arithmetic conversion,
two-pass semantics, non-circular self-check, a byte-preserving parser, a
thread-safe GUI, 96 green tests on a 3-OS/Python matrix. Nothing here is
"broken" — these are the highest-leverage next steps, grouped and prioritized.

The repo has **no open issues, discussions, releases, or PyPI presence** and
**0 stars/forks** — so priorities are inferred from the code and the commit
history (a long run of incremental keyword/material additions plus a
"kill silent physics corruption" bug-fix pass in PR #1), not from user demand.

---

## Tier 0 — Correctness & robustness hardening (do first)

These protect the project's core promise (never silently corrupt physics /
never corrupt untouched bytes). All are small, all have confirmed repros or
clear failure scenarios.

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| **T0.1** | **File I/O uses the platform-default encoding.** A deck with any non-ASCII byte (µ, °, accented author) raises `UnicodeDecodeError` on read or is re-encoded on write — breaking byte-preservation on Windows (the GUI's target). The CLI log already pins `utf-8`; deck I/O does not. Fix: open read/write with `encoding="latin-1"` (round-trips arbitrary bytes 1:1, never raises) or `errors="surrogateescape"`. | `parser.py:77, 211` | HIGH |
| **T0.2** | **`format_fixed` raises an uncaught `ValueError`** when a scaled value can't fit even at 1 sig-fig in a narrow field (confirmed: `format_fixed(Decimal('123456'), 4)`). `convert()` catches only `ParameterFieldError`, so this escapes as a raw traceback after nothing/part of the tree is written. Fix: catch and re-raise as `ConvertError` naming file/line/field. | `parser.py:57`, `convert.py:369` | HIGH |
| **T0.3** | **Free-format precision loss is silent.** The comma-line branch of `scale_field` formats via `float` + `%.9G` and never updates `max_fmt_err`, so the "worst rounding" report and self-check give no signal for comma decks. Fix: track `max_fmt_err` (or warn) on that branch too. | `parser.py:185-187` | HIGH |
| **T0.4** | **Header unit parsing picks tokens by order, not specificity** (confirmed: `"units in mm, kg, ms"` → length parsed as `in`/inch because the word "in" precedes "mm"). Biases/flips detection on decks with sparse physical anchors. Fix: match longest/most-specific unit token, or require word boundaries. | `detect.py:141-143` | MED |
| **T0.5** | **`report()` crashes if a used `Dim` lacks a `DIM_NAMES` entry** (`format(tuple, '<28')` → `TypeError`) — a post-write crash that a future schema extension would trigger. Fix: coerce the fallback to `str`. | `convert.py:499-500` | MED (latent trap) |
| **T0.6** | **Multi-file writes are non-atomic** — a mid-tree write failure leaves files already overwritten, contradicting the in-code "never leave a half-converted tree" comment (that only covers backup collisions). Fix: write temp files + atomic rename. | `convert.py:415-424` | LOW |

---

## Tier 1 — Keyword/material coverage (the dominant historical activity)

Anything unclassified aborts the run by design, so each addition directly
widens the set of real decks that convert. Adding one is "one `Spec` line"
(README). Ranked by value ÷ effort.

**Tier 1A — high value, ~trivial (Spec clone or one-line router change):**
1. **`CONTACT_*_MORTAR`** — the modern default contact; today forced to
   `unknown` by an exclusion in `resolve()`. Remove `"MORTAR"` from the
   exclusion so it reaches the existing `h_contact` penalty handler (verify the
   optional extra card count).
2. **`MAT_123` MODIFIED_PIECEWISE_LINEAR_PLASTICITY** — ubiquitous BIW/crash
   sheet metal; near-identical to the already-supported `MAT_024`. Clone the
   Spec + curve scan + alias.
3. **`DEFINE_TABLE_2D` / `_3D`** — common; route to the existing
   `h_define_table` (only `_TITLE`/`_ID` are stripped today, so these miss).
4. **Whitelist the flag-only cards** that abort needlessly:
   `SECTION_TSHELL`, `SECTION_SEATBELT`, `DEFINE_COORDINATE_VECTOR`.
5. **`MAT_098` SIMPLIFIED_JOHNSON_COOK**, **`MAT_002` ORTHOTROPIC_ELASTIC** —
   small Specs reusing existing patterns.

**Tier 1B — high value, moderate effort:**
6. **Foams — `MAT_063` CRUSHABLE_FOAM, `MAT_057` LOW_DENSITY_FOAM,
   `MAT_181` SIMPLIFIED_RUBBER/FOAM.** Energy-absorbing foam is everywhere in
   crash/seats/barriers and *none* are supported. Spec + register the
   stress-vs-strain LCID (abscissa dimensionless → safe).
7. **`MAT_054/055` ENHANCED_COMPOSITE_DAMAGE** — extend the `MAT_022` Spec.
8. **`MAT_100` SPOTWELD** — failure fields are force/moment; few dimensional fields.
9. **`CONSTRAINED_GENERALIZED_WELD_*`, `CONSTRAINED_TIED_NODES_FAILURE`** — common BIW.
10. **`EOS_IDEAL_GAS`, `EOS_TABULATED_COMPACTION`** — needed for ALE blast/soil.

**Tier 1C — high value, high effort / risky (design a flag-dependent handler, refuse ambiguous history-variable fields as the code already does elsewhere):**
11. **`AIRBAG_*` family** (`SIMPLE_AIRBAG_MODEL`, `PARTICLE`/CPM,
    `REFERENCE_GEOMETRY`) — the entire occupant-safety vocabulary is absent, so
    every airbag deck aborts. Gas constants/specific heats inherit the existing
    shared-temperature-unit caveat.
12. **`MAT_224` TABULATED_JOHNSON_COOK, `MAT_187` SAMP-1** — modern
    metals/plastics, but behaviour rides on flag-dependent rate/temperature
    curves and model-dependent history fields; treat like the CSCM/`INITIAL_STRESS`
    refusal paths unless fully modeled.

**Latent gap to audit separately:** any unmatched `CONTROL_*` is classified
`soft` (pass-through + warn), so a *dimensional* field on an unmodeled CONTROL
card slips through unscaled rather than aborting. Worth a targeted audit.

**Documented limitations that are natural next features:** `*INCLUDE_PATH` and
continued (`+`) include filename lines; the hard-stops (`*PARAMETER`,
`*DEFINE_TRANSFORMATION`, `*DEFINE_FUNCTION`, `*INCLUDE_TRANSFORM`); and
cross-unit-system temperature scaling (currently always refused).

---

## Tier 2 — Test coverage (lock in the hardening above)

Suite is 96 tests / all green, but skewed to the engine. Biggest gaps:

1. **CLI `convert` is end-to-end untested** (only `check`/`detect` are driven
   through `main()`). Add tests for: default output filename, `-o`,
   `--in-place`+`-o` mutual exclusion (rc 2), `src==dst` (rc 0), ambiguous
   auto-detect (rc 2), `ConvertError` (rc 1), **self-check FAILED → rc 3**,
   `.kunit.log` writing + `--no-log`, `--dry-run`. ~120 uncovered lines.
2. **`parse_curve_overrides`** string parsing (`"5=time:accel"` and the
   `SystemExit` error path) — currently only the kwarg form is tested.
3. **Detection branches never asserted:** `ambiguous is True`, `system is None`
   (no-evidence), detonation-velocity evidence, the *generic* header-comment
   path (only the kunit stamp is tested).
4. **Parser paths:** CRLF preservation, `LONG=Y` / `+` 20-char field layout,
   `format_fixed`/`set_field` overflow `ValueError` (see T0.2).
5. **Imperial round-trips + `describe()`** psi/lbf/Mbar labels; `parse_system`
   / `parse_dim_name` error messages.
6. **Add coverage measurement** — no `coverage`/`.coveragerc` today. Add a
   `[tool.coverage.run] source=["kunit"]` block and a CI step.
7. **Property-based tests (hypothesis) fit the invariants well:** round-trip
   identity over random values/preset pairs; `factor(A,B)·factor(B,A)==1` and
   `factor(A,B)·factor(B,C)==factor(A,C)` exactly; `parse_number(format_fixed(v,w))`
   within reported `rel_err` and never exceeding `w`.
8. **Golden-file fixtures** for the big decks (whole-file diff) to catch changes
   to *untouched* bytes, complementing the brittle column-slice asserts.

---

## Tier 3 — Ship it: packaging, release, distribution

The engineering is publication-grade; the packaging is "private repo." Highest
leverage for adoption.

**Trivial, high value (make `pip install kunit` presentable):**
1. `pyproject.toml` metadata: add `readme = "README.md"` (else the **PyPI page
   is blank** — the single biggest gap), `authors`, `classifiers`, `keywords`,
   `[project.urls]`; switch to SPDX `license = "MIT"` + `license-files`.
2. **Dedupe the version** — it's hardcoded in both `pyproject.toml` and
   `kunit/__init__.py`. Use `dynamic = ["version"]` from `kunit.__version__`.
3. Add a **`kunit --version`** flag (argparse `action="version"`).
4. Ship a **`py.typed`** marker (the code is fully type-hinted).
5. Cut the first **git tag `v0.2.0`** (none exist).

**Small, high value:**
6. **PyPI publish workflow** on tag (`pypa/gh-action-pypi-publish` + Trusted
   Publishing/OIDC). Zero runtime deps makes the wheel trivially portable.
7. **ruff** (lint + format) and **mypy** in CI; a `[project.optional-dependencies] dev`
   group; broaden the matrix (add 3.10–3.12, macOS).
8. **`examples/` dir** with 2–3 tiny sample `.k` decks so `kunit detect
   examples/foo.k` works out of the box, + a **GUI screenshot** in the README.
9. `CHANGELOG.md`, short `CONTRIBUTING.md`, issue/PR templates, `dependabot.yml`.

**Medium effort, strong audience fit:**
10. **PyInstaller `kunit-gui.exe`** in the release workflow — the real end users
    are analysts on locked-down Windows boxes without Python; zero deps + a GUI
    make this a clean single-file build.
11. **conda-forge** recipe (the CAE audience is conda-heavy) once on PyPI.

---

## Tier 4 — Architecture (enables everything above to scale)

These reduce the cost of every future keyword addition:

- **✅ Done: `resolve()` is now a declarative, ordered rule table** with a
  `register_keyword()` single entry point, guarded by 28 precedence
  regression tests.
- **⏸ Deferred (own PR): split the `Ctx` god-object** (~30 attrs, reused across scan+edit) into an
  immutable `ScanResult` and an `EditContext`, making the two-pass contract
  explicit and the `self.kf` "current file" rebinding safer.
- **Make `resolve()` a declarative prefix→handler registry** with an explicit
  per-family exclusion set, instead of the current order-dependent prefix chain.
- **One "register a keyword" entry point** — today adding one can touch `SPECS`,
  `CUSTOM`, `_MAT_ALIASES`, `WHITELIST`, `SCAN/EDIT_EXTRA`, `resolve()`, and
  `DIM_NAMES`.
- **De-duplicate** the inlined title-stripping (`detect.py` should call
  `_strip_title`) and the field-splice idiom.
- **Type hygiene:** annotate handler `ctx` via `TYPE_CHECKING` import; fix the
  `Ctx.kf: KFile` hint that can be `None`; tighten `Card.dims`, `format_fixed`,
  `scale_field(f)` signatures; correct the `format_fixed` docstring
  ("shortest" → "most precise that fits").

---

## Quick doc fix (do immediately)

README "Validation" still says **"30 unit tests"**; the suite is now **96**.
