# Contributing

Thanks for your interest in `kunit`. This is a small, focused tool — the goal
is correct, verifiable unit conversion of LS-DYNA keyword decks.

## Setup

```bash
python -m pip install -e ".[dev]"
```

`python kunit.py ...` runs the CLI without installing.

## Running the tests

```bash
python -m unittest discover -s tests          # the whole suite
python -m unittest discover -s tests -v        # verbose (what CI runs)
python -m unittest tests.test_kunit            # a single module
```

Coverage:

```bash
coverage run -m unittest discover -s tests && coverage report
```

## Lint & type-check

```bash
ruff check .
mypy kunit
```

The `ruff` config (in `pyproject.toml`) is intentionally lenient — please do
not reformat unrelated code in a PR.

## Extending coverage: one `Spec` line

The converter is table-driven. To make a new keyword scalable you almost
always add **one `Spec` entry** to `kunit/schema.py` that maps each card's
numeric fields to a dimension signature — for example:

```python
"MAT_SOME_NEW_MODEL": Spec(cards=[
    C({1: DENSITY, 2: PRESSURE, 4: PRESSURE}),   # card 1 fields (0-indexed)
    C()],                                        # card 2: no scaled fields
    probe={"ro": (0, 1), "e": (0, 2)}),          # optional detection anchors
```

Field→dimension maps are verified against the R16 manuals (cite the PDF page
in a comment, as the existing entries do). Keywords that genuinely cannot be
scaled belong on the hard-stop / whitelist lists instead. Add a test that
converts a minimal deck using the new keyword and asserts the self-check
passes.

## Pull requests

- Keep changes focused; one logical change per PR.
- Add or update tests for any behavior change.
- Make sure `python -m unittest discover -s tests` is green.
- Follow the PR template (summary / rationale / test plan).
