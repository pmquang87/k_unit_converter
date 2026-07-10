# kunit — LS-DYNA .k deck unit-system converter

[![tests](https://github.com/pmquang87/k_unit_converter/actions/workflows/ci.yml/badge.svg)](https://github.com/pmquang87/k_unit_converter/actions/workflows/ci.yml)

Converts a self-contained LS-DYNA keyword deck (or a whole `*INCLUDE` tree)
between **any two unit systems** (any mass–length–time triple), with
**auto-detection** of the deck's current units, built-in self-verification,
and a GUI. Field-preserving: only numeric fields that carry physical
dimensions are rewritten, byte-for-byte everything else.

```
pip install -e .            # gives the `kunit` and `kunit-gui` commands

kunit systems                            # list presets
kunit detect  deck.k                     # what units is this deck in?
kunit detect  deck.k --json              # machine-readable verdict (CI-friendly)
kunit check   deck.k                     # coverage report: convertible as-is?
kunit check   deck.k --follow-includes --json
kunit convert deck.k --to ton-mm-s       # auto-detect source, write deck__ton-mm-s.k
kunit convert deck.k --to g-mm-ms --from kg-m-s -o out.k
kunit convert deck.k --to ton-mm-s --in-place          # keeps .orig_<from> backups
kunit convert deck.k --to ton-mm-s --follow-includes   # convert the include tree
kunit convert deck.k --to ton-mm-s --dry-run           # report only, write nothing
kunit convert deck.k --to ton-mm-s --verify-roundtrip  # prove no precision loss
kunit convert deck.k --to ton-mm-s --curve 17=time:accel   # declare curve dims
kunit gui                                # or kunit-gui
```

`python kunit.py ...` still works without installing. Units: mass
`kg g ton(=Mg/tonne/t) lb slug slinch`, length `m mm cm in ft`, time `s ms us`
— any combination, e.g. `--to kg-cm-ms`.

> The GUI (`kunit gui` / `kunit-gui`) needs Tkinter, which ships with most
> CPython builds but is a **separate OS package** on some Linux distros —
> e.g. `sudo apt install python3-tk` on Debian/Ubuntu. The CLI has no such
> dependency.

## Use as a library

Everything the CLI does is importable — no subprocess required:

```python
from kunit import convert, detect, PRESETS

verdict = detect("deck.k")                 # auto-detect current units
print(verdict.system, verdict.ambiguous)   # e.g. kg-m-s, False

src = PRESETS["kg-m-s"]
dst = PRESETS["ton-mm-s"]
convert("deck.k", src, dst, "deck__ton-mm-s.k")   # write the converted deck
```

The full public API is `convert, report, scan, detect, ConvertError,
load_tree, UnitSystem, PRESETS, parse_system, parse_dim_name, factor`
(see `kunit/__init__.py`).

## Examples

Ready-to-run sample decks live in [`examples/`](examples/) — small valid
LS-DYNA decks in different unit systems you can `detect`, `check`, and
`convert`. See [`examples/README.md`](examples/README.md).

## How it works

* Every unit system is an (M, L, T) triple stored as exact `Fraction`s; every
  schema field carries a dimension signature `(a, b, c)` and scales by
  `(Ms/Md)^a (Ls/Ld)^b (Ts/Td)^c` — metric↔metric factors are exact powers
  of ten (applied via `Decimal`, no float drift).
* **Two passes.** Pass 1 resolves semantics: what each `*DEFINE_CURVE`
  ordinate/abscissa physically is (from every keyword that references it —
  gravity `*LOAD_BODY` → acceleration, `MAT_024 LCSS` → stress vs strain,
  `*BOUNDARY_PRESCRIBED_MOTION` VAD/DOF → vel/accel/disp, spring/damper
  materials via the `*SECTION_DISCRETE` DRO flag...), which curves belong to
  a `*DEFINE_TABLE` (both the `value, lcid` pair form and the
  curves-following-the-table form), plus a safety inventory. Pass 2 rewrites
  fields in place respecting true column layouts (`*NODE` I8+3E16,
  `*ELEMENT_MASS` I8,I8,F16,I8, curve data 2E20, long format ×20), comma
  free-format lines, E-less Fortran exponents (`7.85000-9`) and Fortran
  double exponents (`1.0D+5`). Blank lines inside a keyword block are
  treated as what they are in LS-DYNA — all-default data cards — so
  optional-card skipping never shifts later cards onto the wrong field map.
* **Safety net:** every keyword in the deck must be classified — scalable
  (schema/custom handler), dimensionless whitelist, or known-unsupported.
  Anything unknown **aborts** the conversion (override: `--allow-unknown`,
  which leaves them unchanged and lists them). `*PARAMETER`,
  `*DEFINE_TRANSFORMATION`, `*DEFINE_FUNCTION`, `*INCLUDE_TRANSFORM` are hard
  stops; a `&param` in any field to be scaled is always an error.
  Unresolvable curves can be declared with `--curve LCID=<xdim>:<ydim>`.
* **Auto-detection** scores all preset systems against material densities
  (steel 7850 kg/m³, Al 2700, …), elastic moduli (2.1e11 Pa steel — this
  pins the *time* unit via c=√(E/ρ)), detonation velocities, gravity-shaped
  `*LOAD_BODY` curves (9.80665 m/s²), and `$ Unit system :` header comments,
  gathering evidence across the whole include tree. Ambiguous verdicts refuse
  to convert without an explicit `--from`.

## Checking a deck without converting

`kunit check deck.k` classifies every keyword (scalable / dimensionless /
left-unchanged / hard-stop / unknown), reports how each `*DEFINE_CURVE`'s
axis dimensions were resolved (and which curves need a `--curve` override),
and gives a verdict: exit 0 = convertible, 1 = hard stops, 2 = unknown
keywords. `--json` (also on `kunit detect`) emits the same report as JSON
for CI pipelines.

## Verification built in

* **Self-check** (default on): after writing, the output is re-detected and
  must score as the *target* system — a missed dimensional field (schema gap)
  shows up immediately as `SELF-CHECK FAILED`. The re-detection ignores
  header comments (including kunit's own conversion stamp), so the check is
  driven by physical evidence only and cannot confirm its own claim.
* **`--verify-roundtrip`**: converts the output back to the source system and
  forward again; the two forward results must agree byte-for-byte (comments
  ignored), proving formatting lost no precision.
* A `<out>.kunit.log` report (factors, keyword counts, warnings, notes) is
  written next to every output.

## Include trees

`--follow-includes` loads the whole `*INCLUDE` tree (cycle-safe), gathers
curve/table/DRO semantics **globally across files**, converts every file, and
rewrites the `*INCLUDE` references: converted children are written next to
their sources as `<name>__<to>.k` (in-place mode overwrites each file and
keeps per-file backups; references then stay unchanged).
`*INCLUDE_PATH` and continued (`+`) filename lines are not supported.

## Blast loads

`*LOAD_BLAST_ENHANCED` / `*LOAD_BLAST` UNIT flags are remapped, not scaled:
the target system is looked up in the R16 built-in table
(1=lbm/ft/s, 2=kg/m/s, 3=slinch/in/s, 4=g/cm/µs, 6=kg/mm/ms, 7=ton/mm/s,
8=g/mm/ms); anything else — or `--blast-unit5` — emits UNIT=5 with
CFM/CFL/CFT/CFP computed per the R16 manual (conversions **to ConWep's
imperial lbm/ft/ms/psi**, note milliseconds). Charge mass M, XBO/YBO/ZBO,
TBO and DEATH scale as model units; a UNIT flag that contradicts the source
system is warned about loudly.

## Coverage notes

Field→dimension maps are verified against the R16 manuals (PDF page refs in
`kunit/schema.py`). Beyond the common structural/crash/blast vocabulary this
includes `MAT_JOHNSON_COOK` (TM/TR left as temperatures, CP scaled assuming
both systems share the temperature unit), discrete springs/dampers S01–S05
(translational vs torsional resolved through `*SECTION_DISCRETE` DRO and
`*PART` wiring), `*SECTION_DISCRETE`, `*RIGIDWALL_PLANAR` (+ORTHO/FINITE/
MOVING/FORCES/ID), AUTOMATIC contact `TIEBREAK` Card 4 (OPTION 13/14 and the
option-dependent PARAM are refused/warned), `*INITIAL_STRESS_SHELL/SOLID`
(LARGE=0, no history variables — those have material-dependent units),
`*INITIAL_STRESS_SECTION`, `*DEFINE_CURVE_SMOOTH`. Temperature fields are
classified but **never rescaled**. Anything else aborts loudly by design;
extending = one `Spec` line in `kunit/schema.py`.

## Validation

* A comprehensive unittest suite (run `python -m unittest discover -s tests`),
  green on the CI matrix (Linux + Windows, Python 3.9 and 3.13).
* End-to-end: the 8.6 MB W13 blast-vehicle deck (kg/m/s → ton/mm/s) matches a
  manually converted, OpenRadioss-starter-validated reference on **all
  110,565 data lines with zero numeric differences**; its k2rad→OpenRadioss
  starter run reproduces the baseline exactly (0 errors, WTNT 0.05, 388
  underground segments, 14 near-field warnings, added mass 2.05 ton); the
  round-trip check reproduces all 110,564 payload lines exactly and the
  output self-detects as ton-mm-s.
