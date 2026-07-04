# kunit — LS-DYNA .k deck unit-system converter

Converts a self-contained LS-DYNA keyword deck between **any two unit
systems** (any mass–length–time triple), with **auto-detection** of the
deck's current units. Field-preserving: only numeric fields that carry
physical dimensions are rewritten, byte-for-byte everything else.

```
python kunit.py systems                       # list presets
python kunit.py detect  deck.k                # what units is this deck in?
python kunit.py convert deck.k --to ton-mm-s  # auto-detect source, write deck__ton-mm-s.k
python kunit.py convert deck.k --to g-mm-ms --from kg-m-s -o out.k
python kunit.py convert deck.k --to ton-mm-s --in-place   # keeps .orig_<from> backup
```

Units: mass `kg g ton(=Mg/tonne/t) lb slug slinch`, length `m mm cm in ft`,
time `s ms us` — any combination, e.g. `--to kg-cm-ms`.

## How it works

* Every unit system is an (M, L, T) triple stored as exact `Fraction`s; every
  schema field carries a dimension signature `(a, b, c)` and scales by
  `(Ms/Md)^a (Ls/Ld)^b (Ts/Td)^c` — metric↔metric factors are exact powers
  of ten (applied via `Decimal`, no float drift).
* **Two passes.** Pass 1 resolves semantics: what each `*DEFINE_CURVE`
  ordinate/abscissa physically is (from every keyword that references it —
  gravity `*LOAD_BODY` → acceleration, `MAT_024 LCSS` → stress vs strain,
  `*BOUNDARY_PRESCRIBED_MOTION` VAD/DOF → vel/accel/disp, translational vs
  rotational...), plus a safety inventory. Pass 2 rewrites fields in place
  respecting true column layouts (`*NODE` I8+3E16, `*ELEMENT_MASS` I8,I8,F16,I8,
  curve data 2E20, long format ×20), comma free-format lines, and E-less
  Fortran exponents (`7.85000-9`).
* **Safety net:** every keyword in the deck must be classified — scalable
  (schema/custom handler), dimensionless whitelist, or known-unsupported.
  Anything unknown **aborts** the conversion (override: `--allow-unknown`,
  which leaves them unchanged and lists them). `*INCLUDE`, `*PARAMETER`,
  `*DEFINE_TRANSFORMATION`, `*DEFINE_FUNCTION` are hard stops; a `&param` in
  any field to be scaled is always an error.
* **Auto-detection** scores all preset systems against material densities
  (steel 7850 kg/m³, Al 2700, …), elastic moduli (2.1e11 Pa steel — this
  pins the *time* unit via c=√(E/ρ)), detonation velocities, gravity-shaped
  `*LOAD_BODY` curves (9.80665 m/s²), and `$ Unit system :` header comments.
  Ambiguous verdicts refuse to convert without an explicit `--from`.

## Blast loads

`*LOAD_BLAST_ENHANCED` / `*LOAD_BLAST` UNIT flags are remapped, not scaled:
the target system is looked up in the R16 built-in table
(1=lbm/ft/s, 2=kg/m/s, 3=slinch/in/s, 4=g/cm/µs, 6=kg/mm/ms, 7=ton/mm/s,
8=g/mm/ms); anything else — or `--blast-unit5` — emits UNIT=5 with
CFM/CFL/CFT/CFP computed per the R16 manual (conversions **to ConWep's
imperial lbm/ft/ms/psi**, note milliseconds). Charge mass M, XBO/YBO/ZBO,
TBO and DEATH scale as model units; a UNIT flag that contradicts the source
system is warned about loudly.

## Known limitations (v0.1)

* Coverage is the common structural/crash/blast vocabulary (see
  `kunit/schema.py`); anything else triggers the unknown-keyword abort by
  design. Extending = adding one `Spec` line with the field dimensions from
  the LS-DYNA manual.
* `*DEFINE_CURVE` never referenced by a supported keyword is left unchanged
  with a warning — tell the tool what it is by extending the schema, or scale
  it by hand.
* Unreferenced `*SET_SEGMENT` DA1–DA4 attributes are assumed dimensionless.
* Temperatures (Johnson-Cook TR, blast BLAST=3 TEMP) are never rescaled.
* `*ELEMENT_MASS` values are written ending 2 columns short of the F16 field
  edge — spec-conformant, and immune to a known k2rad parser truncation bug.

## Validation

* 17 unit tests (`python -m unittest discover -s tests`).
* End-to-end: the 8.6 MB W13 blast-vehicle deck (kg/m/s → ton/mm/s) matches a
  manually converted, OpenRadioss-starter-validated reference on **all
  110,565 data lines with zero numeric differences** (worst field-width
  rounding 3e-6), and its k2rad→OpenRadioss starter run reproduces the
  baseline exactly: 0 errors, WTNT 0.05, 388 underground segments, 14
  near-field warnings, total mass ×1e-3 to 13 digits, added mass 2.05 ton.
