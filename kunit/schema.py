"""Keyword dimension database + custom handlers.

Every keyword encountered in a deck must resolve to one of:
  * a Spec (table-driven field scaling),
  * a custom handler (flag-dependent formats: curves, blast, motion...),
  * the dimensionless whitelist (topology / ids / flags only),
  * a soft/hard flag (known-unsupported -> loud warning / abort).
Anything else is UNKNOWN and aborts the conversion unless --allow-unknown,
because silently passing a dimensional card through would corrupt physics.

All field->dimension maps are verified against the LS-DYNA R16 manuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .parser import STD8, Block, KFile, parse_number
from .units import (ACCEL, ANG_ACCEL, ANG_VEL, AREA, DAMP, DC_FRIC, DENSITY,
                    Dim, DIM_NAMES, DIMLESS, FORCE, FREQ, INERTIA, L4, LENGTH,
                    MASS, MASS_AREA, MASS_LEN, MOMENT, PRESSURE, RATE,
                    ROT_DAMP, SPEC_HEAT, STIFF, STIFF_LEN, STRESS_M3, TEMP,
                    THERM_COND, TIME, VELOCITY, VISCOSITY, VOLUME,
                    BLAST_BUILTIN_UNITS, BLAST_UNIT_SYSTEMS, CSCM_UNITS,
                    CSCM_UNIT_SYSTEMS, blast_unit5_factors)

STRAIN = DIMLESS


@dataclass
class Card:
    dims: Dict[int, object] = dfield(default_factory=dict)  # Dim or TEMP
    widths: Sequence[int] = tuple(STD8)
    pad_right: Dict[int, int] = dfield(default_factory=dict)
    heading: bool = False       # free-text line, never scaled


@dataclass
class Spec:
    cards: List[Card] = dfield(default_factory=list)
    repeat: Optional[Card] = None
    group: Optional[List[Card]] = None      # repeating multi-card pattern
    curves: List[Tuple[int, int, Dim, Dim]] = dfield(default_factory=list)
    probe: Dict[str, Tuple[int, int]] = dfield(default_factory=dict)
    extra_ok: bool = False      # tolerate trailing unmodelled cards silently


C = Card
NODE_W = (8, 16, 16, 16, 8, 8)
EMASS_W = (8, 8, 16, 8)
CURVE_W = (20, 20)

SPECS: Dict[str, Spec] = {
    # ── mesh / mass ─────────────────────────────────────────────────────────
    "NODE": Spec(repeat=C({1: LENGTH, 2: LENGTH, 3: LENGTH}, NODE_W)),
    "ELEMENT_MASS": Spec(repeat=C({2: MASS}, EMASS_W, pad_right={2: 2})),
    "ELEMENT_MASS_NODE_SET": Spec(repeat=C({2: MASS}, EMASS_W, pad_right={2: 2})),
    "ELEMENT_MASS_PART": Spec(repeat=C({1: MASS, 2: MASS})),
    "ELEMENT_MASS_PART_SET": Spec(repeat=C({1: MASS, 2: MASS})),
    "ELEMENT_INERTIA": Spec(group=[
        C({}, (8, 8, 8)),
        C({0: INERTIA, 1: INERTIA, 2: INERTIA, 3: INERTIA, 4: INERTIA,
           5: INERTIA, 6: MASS}, (10,) * 7)]),
    "ELEMENT_SHELL_THICKNESS": Spec(group=[
        C({}, (8,) * 10),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH}, (16,) * 5)]),
    "ELEMENT_DISCRETE": Spec(repeat=C({7: LENGTH}, (8, 8, 8, 8, 8, 16, 8, 16))),

    # ── parts / sections ────────────────────────────────────────────────────
    "PART": Spec(group=[C(heading=True), C()]),
    "PART_INERTIA": Spec(group=[
        C(heading=True), C(),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: MASS}),
        C({i: INERTIA for i in range(6)}),
        C({0: VELOCITY, 1: VELOCITY, 2: VELOCITY,
           3: ANG_VEL, 4: ANG_VEL, 5: ANG_VEL})]),
    "PART_CONTACT": Spec(group=[
        C(heading=True), C(),
        C({2: DC_FRIC, 3: PRESSURE, 4: LENGTH})]),
    "SECTION_SHELL": Spec(cards=[
        C(),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH, 5: MASS_AREA})]),
    "SECTION_SOLID": Spec(cards=[C()], extra_ok=True),
    # R16 Vol I p.41-108..41-110 (*SECTION_SPH): SECID CSLH HMIN HMAX SPHINI
    # DEATH START SPHKERN.  CSLH and HMIN/HMAX are scale factors on the
    # smoothing length (dimensionless); SPHINI is an optional initial
    # smoothing length (a length, overrides CSLH); DEATH/START are the stop /
    # start times of the particle approximation; SPHKERN is a kernel flag.
    # The INTERACTION and USER options share this one-card layout.
    "SECTION_SPH": Spec(cards=[C({4: LENGTH, 5: TIME, 6: TIME})]),
    "SECTION_SPH_INTERACTION": Spec(cards=[C({4: LENGTH, 5: TIME, 6: TIME})]),
    "SECTION_SPH_USER": Spec(cards=[C({4: LENGTH, 5: TIME, 6: TIME})]),
    # ELLIPSE adds Card 2 HXCSLH HYCSLH HZCSLH HXINI HYINI HZINI (p.41-109):
    # per-direction smoothing-length constants (dimensionless) and optional
    # per-direction initial smoothing lengths.
    "SECTION_SPH_ELLIPSE": Spec(cards=[
        C({4: LENGTH, 5: TIME, 6: TIME}),
        C({3: LENGTH, 4: LENGTH, 5: LENGTH})]),

    # ── materials (field maps cross-checked against the R16 manual / k2rad) ─
    "MAT_ELASTIC": Spec(cards=[C({1: DENSITY, 2: PRESSURE, 6: PRESSURE})],
                        probe={"ro": (0, 1), "e": (0, 2)}),
    # R16 Vol II p.2-145..2-148 (*MAT_ELASTIC_FLUID / MAT_001_FLUID):
    # Card1 MID RO E PR DA DB K - RO density, E Young's modulus, K bulk
    # modulus (E/PR are ignored for FLUID but keep their dimensions); DA/DB
    # are beam-only damping factors, unused for FLUID (solids only).
    # Card2 VC CP - VC is the dimensionless tensor viscosity coefficient
    # ("values between .1 and .5"), CP the cavitation pressure.
    # E < 0 (curve ID + extra Card 1.1) is refused by x_mat_001.
    "MAT_ELASTIC_FLUID": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 6: PRESSURE}),
        C({1: PRESSURE})],
        probe={"ro": (0, 1), "e": (0, 2)}),
    "MAT_PLASTIC_KINEMATIC": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 4: PRESSURE, 5: PRESSURE}),
        C({0: RATE})],
        probe={"ro": (0, 1), "e": (0, 2)}),
    "MAT_RIGID": Spec(cards=[C({1: DENSITY, 2: PRESSURE}), C(), C()],
                      probe={"ro": (0, 1), "e": (0, 2)}),
    "MAT_PIECEWISE_LINEAR_PLASTICITY": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 4: PRESSURE, 5: PRESSURE, 7: TIME}),
        C({0: RATE}),
        C(),                                     # EPS1-8 plastic strains
        C({i: PRESSURE for i in range(8)})],     # ES1-8 stresses
        probe={"ro": (0, 1), "e": (0, 2)}),
    # R16 Vol II p.2-202: Card1 MID RO G E PR DTF VP RATEOP;
    # Card2 A B N C M TM TR EPS0; Card3 CP PC SPALL IT D1-D4; Card4 rate form.
    "MAT_JOHNSON_COOK": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 5: TIME}),
        C({0: PRESSURE, 1: PRESSURE, 5: TEMP, 6: TEMP, 7: RATE}),
        C({0: SPEC_HEAT, 1: PRESSURE}),
        C()],                                    # card 4 (see EDIT_EXTRA)
        probe={"ro": (0, 1), "e": (0, 3)}),
    # R16 Vol II p.2-245..2-250 (*MAT_COMPOSITE_DAMAGE / MAT_022):
    # Card1 MID RO EA EB EC PRBA PRCA PRCB; Card2 GAB GBC GCA KFAIL AOPT
    # MACF ATRACK; Card3 XP YP ZP A1 A2 A3; Card4 V1 V2 V3 D1 D2 D3 BETA;
    # Card5 SC XT YT YC ALPH SN SYZ SZX.  KFAIL is a bulk modulus and
    # XP/YP/ZP are coordinates; ALPH is "in units of [stress^-3]" (p.2-249);
    # Poisson ratios, A/V/D direction vectors and BETA (degrees) stay.
    "MAT_COMPOSITE_DAMAGE": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 4: PRESSURE}),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE, 3: PRESSURE}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH}),
        C(),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE, 3: PRESSURE, 4: STRESS_M3,
           5: PRESSURE, 6: PRESSURE, 7: PRESSURE})],
        probe={"ro": (0, 1), "e": (0, 2)}),
    "MAT_NULL": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: VISCOSITY, 6: PRESSURE})],
        probe={"ro": (0, 1)}),
    "MAT_VACUUM": Spec(cards=[C({1: DENSITY})], probe={"ro": (0, 1)}),
    "MAT_HIGH_EXPLOSIVE_BURN": Spec(cards=[
        C({1: DENSITY, 2: VELOCITY, 3: PRESSURE, 5: PRESSURE, 6: PRESSURE,
           7: PRESSURE})],
        probe={"ro": (0, 1), "d": (0, 2)}),
    "MAT_ADD_EROSION": Spec(cards=[
        C({2: PRESSURE}),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE, 5: PRESSURE, 6: VISCOSITY,
           7: TIME})], extra_ok=True),
    "EOS_JWL": Spec(cards=[C({1: PRESSURE, 2: PRESSURE, 6: PRESSURE})]),
    "EOS_LINEAR_POLYNOMIAL": Spec(cards=[
        C({i: PRESSURE for i in range(1, 8)}), C({0: PRESSURE})]),
    "EOS_GRUNEISEN": Spec(cards=[C({1: VELOCITY, 7: PRESSURE})]),

    # ── loads / boundary / initial ──────────────────────────────────────────
    "LOAD_SEGMENT": Spec(repeat=C({2: TIME}), curves=[(0, 0, TIME, PRESSURE)]),
    "LOAD_SEGMENT_SET": Spec(repeat=C({3: TIME}),
                             curves=[(0, 1, TIME, PRESSURE)]),
    "LOAD_SHELL_ELEMENT": Spec(repeat=C({3: TIME}),
                               curves=[(0, 1, TIME, PRESSURE)]),
    "LOAD_SHELL_SET": Spec(repeat=C({3: TIME}),
                           curves=[(0, 1, TIME, PRESSURE)]),
    "INITIAL_VELOCITY": Spec(cards=[C()], repeat=C(
        {0: VELOCITY, 1: VELOCITY, 2: VELOCITY,
         3: ANG_VEL, 4: ANG_VEL, 5: ANG_VEL})),
    "INITIAL_VELOCITY_RIGID_BODY": Spec(repeat=C(
        {1: VELOCITY, 2: VELOCITY, 3: VELOCITY,
         4: ANG_VEL, 5: ANG_VEL, 6: ANG_VEL})),
    # R16 Vol I p.28-129: NID VX VY VZ VXR VYR VZR ICID
    "INITIAL_VELOCITY_NODE": Spec(repeat=C(
        {1: VELOCITY, 2: VELOCITY, 3: VELOCITY,
         4: ANG_VEL, 5: ANG_VEL, 6: ANG_VEL})),
    "INITIAL_VELOCITY_GENERATION": Spec(group=[
        C({2: ANG_VEL, 3: VELOCITY, 4: VELOCITY, 5: VELOCITY}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH})]),
    "INITIAL_DETONATION": Spec(repeat=C(
        {1: LENGTH, 2: LENGTH, 3: LENGTH, 4: TIME})),
    # R16 Vol I p.28-91: ISSID CSID LCID PSID VID IZSHEAR ISTIFF
    "INITIAL_STRESS_SECTION": Spec(cards=[C()],
                                   curves=[(0, 2, TIME, PRESSURE)]),
    "CONSTRAINED_SPOTWELD": Spec(repeat=C({2: FORCE, 3: FORCE, 6: TIME})),

    # ── control / database ──────────────────────────────────────────────────
    "CONTROL_TERMINATION": Spec(cards=[C({0: TIME})]),
    "CONTROL_TIMESTEP": Spec(cards=[C({0: TIME, 3: TIME, 4: TIME})],
                             curves=[(0, 5, TIME, TIME)], extra_ok=True),
    "CONTROL_DYNAMIC_RELAXATION": Spec(cards=[C({3: TIME})]),
    # R16 Vol I p.12-530..12-535 (*CONTROL_SPH): Card1 NCBS BOXID DT IDIM
    # NMNEIGH FORM START MAXV - DT is the SPH death time, START the particle-
    # approximation start time, MAXV the deactivation velocity threshold
    # (negative MAXV = clamp instead of deactivate; sign survives scaling).
    # Optional Card2 (CONT..ISYMP) holds flags/percentages and Card3 (ITHK
    # ISTAB QL SPHSORT ISHIFT) flags plus the dimensionless quasi-linear
    # coefficient QL, so both stay unscaled.
    "CONTROL_SPH": Spec(cards=[C({2: TIME, 6: TIME, 7: VELOCITY}), C(), C()]),
    "CONTROL_ALE": Spec(cards=[C(), C({0: TIME, 1: TIME, 6: PRESSURE})],
                        extra_ok=True),
    "CONTROL_IMPLICIT_GENERAL": Spec(cards=[C({1: TIME})]),
    "CONTROL_IMPLICIT_AUTO": Spec(cards=[C({3: TIME, 4: TIME})]),
    "CONTROL_IMPLICIT_DYNAMICS": Spec(cards=[C({3: TIME, 4: TIME, 5: TIME})]),
    "CONTROL_IMPLICIT_EIGENVALUE": Spec(cards=[C({1: FREQ})], extra_ok=True),
    "DAMPING_GLOBAL": Spec(cards=[C({1: FREQ})], curves=[(0, 0, TIME, FREQ)]),
    "DAMPING_PART_MASS": Spec(repeat=C(), curves=[(0, 1, TIME, FREQ)]),
    "DATABASE_CROSS_SECTION_PLANE": Spec(group=[
        C({1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH, 5: LENGTH, 6: LENGTH,
           7: LENGTH})]),
    "DATABASE_ALE_OPERATION": Spec(cards=[C(), C({0: TIME}), C()]),

    # ── defines ─────────────────────────────────────────────────────────────
    "DEFINE_BOX": Spec(cards=[C({1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH,
                                 5: LENGTH, 6: LENGTH})]),
    "DEFINE_VECTOR": Spec(repeat=C({1: LENGTH, 2: LENGTH, 3: LENGTH,
                                    4: LENGTH, 5: LENGTH, 6: LENGTH})),
    "DEFINE_COORDINATE_SYSTEM": Spec(group=[
        C({1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH, 5: LENGTH, 6: LENGTH}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH})]),
    "DEFINE_FRICTION": Spec(cards=[C({3: DC_FRIC, 4: PRESSURE})],
                            repeat=C({4: DC_FRIC, 5: PRESSURE})),
    # R16 Vol I p.17-146: LCID SIDR DIST TSTART TEND TRISE VMAX
    "DEFINE_CURVE_SMOOTH": Spec(repeat=C(
        {2: LENGTH, 3: TIME, 4: TIME, 5: TIME, 6: VELOCITY})),

    # ── ICFD incompressible-flow solver (R16 Vol III) ───────────────────────
    # R16 Vol III p.7-147..7-149 (*ICFD_MAT): Card1 MID FLG RO VIS ST
    # STSFLCID CA - RO is the flow density, VIS the dynamic viscosity, ST the
    # surface tension coefficient (force/length, STIFF signature), STSFLCID a
    # dimensionless time-scale-factor curve and CA a contact angle (degrees).
    # Card2 (thermal) HC TC BETA PRT HCSFLCID TCSFLCID - HC heat capacity, TC
    # thermal conductivity; BETA (1/temperature) and PRT stay unchanged.
    # Card3 NNMOID PMMOID SPTRID VID - model ids only.
    "ICFD_MAT": Spec(cards=[
        C({2: DENSITY, 3: VISCOSITY, 4: STIFF}),
        C({0: SPEC_HEAT, 1: THERM_COND}),
        C()],
        curves=[(0, 5, TIME, DIMLESS), (1, 4, TIME, DIMLESS),
                (1, 5, TIME, DIMLESS)],
        probe={"icfd_ro": (0, 2), "icfd_vis": (0, 3)}),
    # R16 Vol III p.7-70..7-72 (*ICFD_CONTROL_OUTPUT): Card1 MSGL OUTL DTOUT
    # LSPPOUT - ITOUT (DTOUT = output time interval); optional Card2 PITOUT.
    "ICFD_CONTROL_OUTPUT": Spec(cards=[C({2: TIME}), C()]),
    # R16 Vol III p.7-99 (*ICFD_DATABASE_DRAG[_VOL]): one card per surface,
    # PID CPID DTOUT PEROUT DIVI ELOUT SSOUT (DTOUT = output time interval).
    "ICFD_DATABASE_DRAG": Spec(repeat=C({2: TIME})),
    "ICFD_DATABASE_DRAG_VOL": Spec(repeat=C({2: TIME})),

    # ── *MESH volume mesher (R16 Vol III) ───────────────────────────────────
    # R16 Vol III p.8-19 (*MESH_SURFACE_NODE): NID X Y Z, coordinates are
    # lengths; i8 + 3e16 layout like *NODE (LS-PrePost writes 16-char floats;
    # p.8-18 documents the companion element card as 6i8, not 6i10).
    "MESH_SURFACE_NODE": Spec(repeat=C(
        {1: LENGTH, 2: LENGTH, 3: LENGTH}, (8, 16, 16, 16))),
}

# numeric aliases
_MAT_ALIASES = {
    "MAT_001": "MAT_ELASTIC", "MAT_001_FLUID": "MAT_ELASTIC_FLUID",
    "MAT_003": "MAT_PLASTIC_KINEMATIC",
    "MAT_008": "MAT_HIGH_EXPLOSIVE_BURN", "MAT_009": "MAT_NULL",
    "MAT_015": "MAT_JOHNSON_COOK",
    "MAT_020": "MAT_RIGID", "MAT_022": "MAT_COMPOSITE_DAMAGE",
    "MAT_024": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "MAT_140": "MAT_VACUUM",
    "MAT_159": "MAT_CSCM", "MAT_159_CONCRETE": "MAT_CSCM_CONCRETE",
    "MAT_S01": "MAT_SPRING_ELASTIC", "MAT_S02": "MAT_DAMPER_VISCOUS",
    "MAT_S03": "MAT_SPRING_ELASTOPLASTIC",
    "MAT_S04": "MAT_SPRING_NONLINEAR_ELASTIC",
    "MAT_S05": "MAT_DAMPER_NONLINEAR_VISCOUS",
    "EOS_001": "EOS_LINEAR_POLYNOMIAL", "EOS_002": "EOS_JWL",
    "EOS_004": "EOS_GRUNEISEN",
}

# keywords that carry no dimensional data at all
WHITELIST = {
    "KEYWORD", "TITLE", "END", "COMMENT",
    "ELEMENT_SHELL", "ELEMENT_SHELL_BETA", "ELEMENT_SOLID",
    "ELEMENT_SOLID_ORTHO", "ELEMENT_BEAM",
    "CONTROL_ENERGY", "CONTROL_OUTPUT", "CONTROL_ACCURACY", "CONTROL_SHELL",
    "CONTROL_SOLID", "CONTROL_HOURGLASS", "CONTROL_BULK_VISCOSITY",
    "CONTROL_CONTACT", "CONTROL_RIGID", "CONTROL_PARALLEL", "CONTROL_MPP",
    "CONTROL_CPU", "CONTROL_IMPLICIT_SOLUTION", "CONTROL_IMPLICIT_SOLVER",
    "DATABASE_EXTENT_BINARY", "DATABASE_FORMAT",
    "BOUNDARY_NON_REFLECTING", "LOAD_BLAST_SEGMENT_SET", "LOAD_BLAST_SEGMENT",
    "LOAD_BODY_PARTS", "HOURGLASS",
    "CONSTRAINED_NODAL_RIGID_BODY", "CONSTRAINED_EXTRA_NODES_NODE",
    "CONSTRAINED_EXTRA_NODES_SET", "CONSTRAINED_RIGID_BODIES",
    "DEFINE_COORDINATE_NODES", "DEFINE_SD_ORIENTATION",
    "ALE_MULTI-MATERIAL_GROUP", "MAT_ADD_PORE_AIR",
    "INITIAL_VOID_PART", "INITIAL_VOID_SET",
    # strain tensors are dimensionless
    "INITIAL_STRAIN_SHELL", "INITIAL_STRAIN_SHELL_SET",
    "INITIAL_STRAIN_SOLID", "INITIAL_STRAIN_SOLID_SET",
    # ICFD / MESH id-only keywords (R16 Vol III):
    # p.7-9/7-19 boundary pids; p.7-165..7-168 part/section ids;
    # p.8-17/8-21 element connectivity and volume-from-surface-pid lists
    "ICFD_BOUNDARY_FREESLIP", "ICFD_BOUNDARY_NONSLIP",
    "ICFD_PART", "ICFD_PART_VOL", "ICFD_SECTION",
    "MESH_SURFACE_ELEMENT", "MESH_VOLUME",
}
WHITELIST_PREFIXES = (
    "SET_", "BOUNDARY_SPC", "DATABASE_HISTORY", "CONTROL_MPP_",
    "DEFORMABLE_TO_RIGID", "INTERFACE_SPRINGBACK",
)
# known-unsupported: abort (hard). *INCLUDE is bypassed by --follow-includes.
HARD_FLAGS = {
    "INCLUDE": "multi-file deck - re-run with --follow-includes to convert "
               "the whole tree",
    "INCLUDE_TRANSFORM": "carries its own scale factors",
    "INCLUDE_PATH": "search-path includes not supported - flatten the deck "
                    "or use plain *INCLUDE with relative paths",
    "PARAMETER": "parameters may feed dimensional fields",
    "PARAMETER_EXPRESSION": "parameters may feed dimensional fields",
    "DEFINE_TRANSFORMATION": "carries its own scale factors",
    "DEFINE_FUNCTION": "free-form expressions cannot be auto-scaled",
    "DEFINE_CURVE_FUNCTION": "free-form expressions cannot be auto-scaled",
}


# ─────────────────────────────────────────────────────────────────────────────
# custom handlers (flag-dependent card layouts). Each is fn(block, ctx, edit).
# ─────────────────────────────────────────────────────────────────────────────

def _numint(kf: KFile, li: int, widths, long, fi) -> Optional[int]:
    v = kf.get_number(li, widths, long, fi)
    return int(v) if v is not None else None


def _strip_title(block: Block, data):
    opts = block.name.split("_")
    if "TITLE" in opts or "ID" in opts:
        return data[1:]
    return data


def h_define_curve(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    lcid = _numint(kf, data[0], STD8, block.long, 0)
    if not edit:
        ctx.curve_blocks.setdefault(lcid, []).append((kf, block))
        return
    dims = ctx.curve_dims.get(lcid)
    if not dims:
        ctx.warn(f"*{block.name} lcid={lcid}: no referencing keyword tells me "
                 "its axis dimensions - data points left UNCHANGED. Use "
                 "--curve {}=<xdim>:<ydim> if it is dimensional.".format(lcid))
        ctx.count(block.name + " (unreferenced, unchanged)")
        return
    if len(dims) > 1:
        # Referencers that agree on the abscissa but disagree on the ordinate
        # are still safe when every ordinate (and OFFO) is zero - a common
        # LS-PrePost pattern (one all-zero curve shared by e.g. a
        # zero-pressure outlet and a zero-velocity constraint).
        xdims = {xd for (xd, _yd) in dims}
        ords = [kf.get_number(li, CURVE_W, block.long, 1) for li in data[1:]]
        offo = kf.get_number(data[0], STD8, block.long, 5)
        if len(xdims) == 1 and not offo and not any(ords):
            wants = ", ".join(
                f"*{src} wants {DIM_NAMES.get(yd, yd)}"
                for (_xd, yd), src in sorted(dims.items(), key=str))
            ctx.warn(f"*{block.name} lcid={lcid}: referencers disagree on the "
                     f"ordinate dimension ({wants}) but every ordinate is "
                     "zero, so the conflict is immaterial - abscissas scaled, "
                     "ordinates left at zero.")
            xdim, ydim = next(iter(xdims)), DIMLESS
        else:
            ctx.error(f"*{block.name} lcid={lcid}: conflicting dimension "
                      f"demands from referencers: {sorted(dims.items())} - "
                      f"resolve with --curve {lcid}=<xdim>:<ydim>")
            return
    else:
        (xdim, ydim), _src = next(iter(dims.items()))
    fx, fy = ctx.fac(xdim), ctx.fac(ydim)
    kf.scale_field(data[0], STD8, block.long, 4, fx)   # OFFA
    kf.scale_field(data[0], STD8, block.long, 5, fy)   # OFFO
    for li in data[1:]:
        kf.scale_field(li, CURVE_W, block.long, 0, fx)
        kf.scale_field(li, CURVE_W, block.long, 1, fy)
    ctx.count(block.name)


def h_define_table(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    tbid = _numint(kf, data[0], STD8, block.long, 0)
    if not edit:
        ctx.table_blocks[tbid] = (kf, block)
        # 'value, lcid' pair form: second numeric field names the sub-curve
        pairs = []
        for li in data[1:]:
            fl = kf.fields(li, CURVE_W, block.long)
            sub = parse_number(fl[1][0]) if len(fl) > 1 else None
            if sub:
                pairs.append(int(sub))
        ctx.table_pairs[tbid] = pairs
        ctx.table_nvalues[tbid] = len(data) - 1
        return
    entry = ctx.table_dims.get(tbid)
    if not entry:
        ctx.warn(f"*{block.name} tbid={tbid}: unreferenced - left UNCHANGED.")
        return
    vdim, _xd, _yd = entry
    fv = ctx.fac(vdim)
    kf.scale_field(data[0], STD8, block.long, 2, fv)   # OFFA
    for li in data[1:]:
        kf.scale_field(li, CURVE_W, block.long, 0, fv)
    ctx.count(block.name)


def h_load_body(block: Block, ctx, edit: bool) -> None:
    axis = block.name.rsplit("_", 1)[-1]
    ydim = ACCEL if axis in ("X", "Y", "Z") else ANG_ACCEL
    kf = ctx.kf
    for li in block.data:
        lcid = _numint(kf, li, STD8, block.long, 0)
        if not edit:
            if lcid:
                ctx.register_curve(lcid, TIME, ydim, block.name)
                ctx.probes["gravity_lcids"].append(lcid)
        else:
            for fi in (3, 4, 5):   # XC YC ZC (angular arm point)
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(LENGTH))
    if edit:
        ctx.count(block.name)


def h_load_node_or_rb(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    for li in block.data:
        dof = _numint(kf, li, STD8, block.long, 1) or 0
        lcid = _numint(kf, li, STD8, block.long, 2)
        ydim = MOMENT if dof in (5, 6, 7, 8) else FORCE
        if not edit and lcid:
            ctx.register_curve(lcid, TIME, ydim, block.name)
    if edit:
        ctx.count(block.name + " (curve-carried)")


def h_prescribed_motion(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    for li in block.data:
        dof = abs(_numint(kf, li, STD8, block.long, 1) or 0)
        vad = _numint(kf, li, STD8, block.long, 2) or 0
        lcid = _numint(kf, li, STD8, block.long, 3)
        rot = dof in (5, 6, 7, 8, 9, 10, 11)
        if vad == 3:
            xdim, ydim = LENGTH, VELOCITY
        else:
            table = {0: VELOCITY, 1: ACCEL, 2: LENGTH, 4: LENGTH}
            rtable = {0: ANG_VEL, 1: ANG_ACCEL, 2: DIMLESS, 4: DIMLESS}
            xdim = TIME
            ydim = (rtable if rot else table).get(vad)
            if ydim is None:
                ctx.error(f"*{block.name}: unsupported VAD={vad}")
                continue
        if not edit:
            if lcid:
                ctx.register_curve(lcid, xdim, ydim, block.name)
        else:
            kf.scale_field(li, STD8, block.long, 6, ctx.fac(TIME))  # DEATH
            kf.scale_field(li, STD8, block.long, 7, ctx.fac(TIME))  # BIRTH
    if edit:
        ctx.count(block.name)


def h_contact(block: Block, ctx, edit: bool) -> None:
    if not edit:
        return
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    tiebreak = "TIEBREAK" in block.name
    plan = [
        {},                                              # card1: ids
        {2: DC_FRIC, 3: PRESSURE, 6: TIME, 7: TIME},     # card2
        {2: LENGTH, 3: LENGTH},                          # card3: SST MST
    ]
    if tiebreak:
        # R16 Vol I p.11-35 Card 4: OPTION NFLS SFLS PARAM ERATEN ERATES
        # CT2CN CN.  ERATEN/ERATES = energy/area, CN = stiffness/length.
        plan.append({1: PRESSURE, 2: PRESSURE, 4: STIFF, 5: STIFF,
                     7: STIFF_LEN})
    else:
        plan.append({})                                  # A (flags)
    plan.append({6: LENGTH, 7: PRESSURE})                # B: SLDTHK SLDSTF
    for ci, li in enumerate(data):
        if ci >= len(plan):
            ctx.warn(f"*{block.name}: optional card {ci + 1} left unscaled "
                     "(advanced options not modelled) - verify manually.")
            break
        if tiebreak and ci == 3:
            option = abs(_numint(kf, li, STD8, block.long, 0) or 0)
            if option in (13, 14):
                ctx.error(f"*{block.name}: TIEBREAK OPTION={option} adds "
                          "rate-dependent fracture cards that are not "
                          "modelled - convert manually.")
                return
            param = kf.get_number(li, STD8, block.long, 3)
            if param:
                ctx.warn(f"*{block.name}: TIEBREAK PARAM={param} left "
                         "unscaled - its meaning (and units) depend on "
                         "OPTION (length for 6/8/9/11, exponent for 2...) - "
                         "verify against the manual.")
        for fi, dim in plan[ci].items():
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(dim))
    ctx.count("CONTACT_*" + (" TIEBREAK" if tiebreak else ""))


def h_mat_024(block: Block, ctx) -> None:
    """Curve/table registration for MAT_PIECEWISE_LINEAR_PLASTICITY (scan)."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) > 1:
        lcss = _numint(kf, data[1], STD8, block.long, 2)
        lcsr = _numint(kf, data[1], STD8, block.long, 3)
        if lcss:
            ctx.register_curve(lcss, STRAIN, PRESSURE, "MAT_024 LCSS")
            ctx.register_table(lcss, RATE, STRAIN, PRESSURE)
        if lcsr:
            ctx.register_curve(lcsr, RATE, DIMLESS, "MAT_024 LCSR")


def h_element_sph(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.19-136 (*ELEMENT_SPH[_VOLUME]): NID PID MASS NEND in the
    i8,i8,e16,i8 layout shared with *ELEMENT_MASS.  MASS > 0 is the particle
    mass, but MASS < 0 - or any value with the VOLUME option - is a particle
    VOLUME (density then comes from the material card), so the field's
    dimension depends on its sign.  NEND generation replicates the same MASS
    value across NID..NEND, so scaling the one field covers the range."""
    if not edit:
        return
    kf = ctx.kf
    vol_opt = "VOLUME" in block.name.split("_")
    for li in block.data:
        v = kf.get_number(li, EMASS_W, block.long, 2)
        if not v:
            continue
        dim = VOLUME if (vol_opt or v < 0) else MASS
        kf.scale_field(li, EMASS_W, block.long, 2, ctx.fac(dim), pad_right=2)
    ctx.count(block.name)


def x_mat_001(block: Block, ctx) -> None:
    """Edit-time check: E < 0 in *MAT_ELASTIC[_FLUID] makes |E| a curve ID
    and inserts Card 1.1 (R16 Vol II p.2-145..2-146), which the fixed card
    layout cannot model - refuse rather than corrupt the curve reference."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    e = kf.get_number(data[0], STD8, block.long, 2)
    if e is not None and e < 0:
        ctx.error(f"*{block.name}: E < 0 means |E| is a curve ID and an "
                  "extra Card 1.1 (EFUNC CNVT ITERLM) follows Card 1 "
                  "(R16 Vol II p.2-145) - this layout is not modelled; "
                  "convert this material manually.")


def x_mat_015(block: Block, ctx) -> None:
    """Edit-time check: JC card 4 rate parameter is RATEOP-dependent."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 4:
        return
    rateop = kf.get_number(data[0], STD8, block.long, 7)
    p2 = kf.get_number(data[3], STD8, block.long, 1)
    if rateop and int(rateop) in (1, 3, 4, 5) and p2:
        ctx.warn(f"*{block.name}: Card 4 rate parameter C2/P/XNP={p2} left "
                 f"unscaled - its units depend on RATEOP={int(rateop)} "
                 "(1/time for Cowper-Symonds P) - verify and scale manually.")


def h_section_beam(block: Block, ctx, edit: bool) -> None:
    if not edit:
        return
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    kf.scale_field(data[0], STD8, block.long, 6, ctx.fac(MASS_LEN))  # NSM
    elform = _numint(kf, data[0], STD8, block.long, 1) or 1
    if len(data) < 2:
        return
    if elform in (1, 4, 5, 11):
        dims = {0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH}
    elif elform == 2:
        dims = {0: AREA, 1: L4, 2: L4, 3: L4, 4: AREA}
    elif elform == 3:
        dims = {0: AREA, 1: TIME, 2: PRESSURE}
    else:
        ctx.warn(f"*SECTION_BEAM elform={elform}: card 2 not modelled - "
                 "left unscaled, verify manually.")
        return
    for fi, dim in dims.items():
        kf.scale_field(data[1], STD8, block.long, fi, ctx.fac(dim))
    ctx.count("SECTION_BEAM")


def h_section_discrete(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.41-36: pairs of (SECID DRO KD V0 CL FD) / (CDL TDL).
    For DRO=1 (torsional) the deflections are radians (dimensionless)."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    for i in range(0, len(data) - len(data) % 2, 2):
        c1, c2 = data[i], data[i + 1]
        secid = _numint(kf, c1, STD8, block.long, 0)
        dro = _numint(kf, c1, STD8, block.long, 1) or 0
        if not edit:
            ctx.sec_discrete_dro[secid] = dro
            continue
        if dro == 0:
            kf.scale_field(c1, STD8, block.long, 3, ctx.fac(VELOCITY))  # V0
            for fi in (4, 5):                                           # CL FD
                kf.scale_field(c1, STD8, block.long, fi, ctx.fac(LENGTH))
            for fi in (0, 1):                                           # CDL TDL
                kf.scale_field(c2, STD8, block.long, fi, ctx.fac(LENGTH))
        else:
            kf.scale_field(c1, STD8, block.long, 3, ctx.fac(ANG_VEL))   # V0
    if edit:
        ctx.count("SECTION_DISCRETE")


def _smat_torsional(kf, block, ctx) -> bool:
    data = _strip_title(block, list(block.data))
    mid = _numint(kf, data[0], STD8, block.long, 0) if data else None
    return mid in ctx.torsional_mats


def h_smat_spring_elastic(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    if not edit:
        ctx.smat_blocks.append((kf, block, "S01"))
        return
    dim = MOMENT if _smat_torsional(kf, block, ctx) else STIFF
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(dim))
    ctx.count(block.name)


def h_smat_spring_elastoplastic(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    if not edit:
        ctx.smat_blocks.append((kf, block, "S03"))
        return
    tors = _smat_torsional(kf, block, ctx)
    kdim = MOMENT if tors else STIFF
    fdim = MOMENT if tors else FORCE
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(kdim))  # K
    kf.scale_field(data[0], STD8, block.long, 2, ctx.fac(kdim))  # KT
    kf.scale_field(data[0], STD8, block.long, 3, ctx.fac(fdim))  # FY
    ctx.count(block.name)


def h_smat_damper_viscous(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    if not edit:
        ctx.smat_blocks.append((kf, block, "S02"))
        return
    dim = ROT_DAMP if _smat_torsional(kf, block, ctx) else DAMP
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(dim))
    ctx.count(block.name)


def h_smat_curve_mats(block: Block, ctx, edit: bool) -> None:
    """S04 (MID LCD LCR) / S05 (MID LCDR): curves carry the physics; their
    dims depend on translational vs torsional, resolved post-scan."""
    if not edit:
        kind = "S05" if "DAMPER" in block.name else "S04"
        ctx.smat_blocks.append((ctx.kf, block, kind))
        return
    ctx.count(block.name + " (curve-carried)")


def h_rigidwall_planar(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.40-17. Repeating sets:
    [ID] Card1 Card2 [ORTHO c3 c4] [FINITE c5] [MOVING c6] [FORCES c7]."""
    if not edit:
        return
    kf = ctx.kf
    opts = set(block.name.split("_"))
    seq: List[Dict[int, Dim]] = []
    if "ID" in opts:
        seq.append({})
    seq.append({3: LENGTH, 4: TIME, 5: TIME})                     # card 1
    seq.append({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH,
                4: LENGTH, 5: LENGTH, 7: VELOCITY})               # card 2
    if "ORTHO" in opts:
        seq.append({4: DC_FRIC, 5: DC_FRIC})                      # card 3
        seq.append({})                                            # card 4
    if "FINITE" in opts:
        seq.append({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH,
                    4: LENGTH})                                   # card 5
    if "MOVING" in opts:
        seq.append({0: MASS, 1: VELOCITY})                        # card 6
    if "FORCES" in opts:
        seq.append({})                                            # card 7
    data = list(block.data)
    if len(data) % len(seq):
        ctx.warn(f"*{block.name}: {len(data)} data cards is not a multiple "
                 f"of the expected set size {len(seq)} - trailing cards "
                 "left unscaled, verify the option combination.")
    for base in range(0, len(data) - len(data) % len(seq), len(seq)):
        for ci, dims in enumerate(seq):
            for fi, dim in dims.items():
                kf.scale_field(data[base + ci], STD8, block.long, fi,
                               ctx.fac(dim))
    ctx.count(block.name)


def h_initial_stress_shell(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.28-95, LARGE=0 layout only."""
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    i = 0
    while i < len(data):
        c1 = data[i]
        nplane = _numint(kf, c1, STD8, block.long, 1) or 0
        nthick = _numint(kf, c1, STD8, block.long, 2) or 0
        flags = [(_numint(kf, c1, STD8, block.long, fi) or 0)
                 for fi in (3, 4, 5, 6, 7)]      # NHISV NTENSR LARGE NTHINT NTHHSV
        if any(flags):
            ctx.error(f"*{block.name}: NHISV/NTENSR/LARGE/NTHINT/NTHHSV != 0 "
                      "layouts are not modelled (history-variable units are "
                      "material-dependent) - convert manually.")
            return
        npts = max(nplane * nthick, 0)
        for li in data[i + 1: i + 1 + npts]:
            for fi in range(1, 7):               # SIGXX..SIGZX (T, EPS stay)
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(PRESSURE))
        i += 1 + npts
    ctx.count(block.name)


def h_initial_stress_solid(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.28-103, LARGE=0 layout only."""
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    i = 0
    while i < len(data):
        c1 = data[i]
        nint = _numint(kf, c1, STD8, block.long, 1) or 0
        flags = [(_numint(kf, c1, STD8, block.long, fi) or 0)
                 for fi in (2, 3, 4, 6, 7)]      # NHISV LARGE IVEFLG NTHINT NTHHSV
        if any(flags):
            ctx.error(f"*{block.name}: NHISV/LARGE/IVEFLG/NTHINT/NTHHSV != 0 "
                      "layouts are not modelled - convert manually.")
            return
        for li in data[i + 1: i + 1 + nint]:
            for fi in range(6):                  # SIGXX..SIGZX (EPS stays)
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(PRESSURE))
        i += 1 + nint
    ctx.count(block.name)


def h_lagrange_in_solid(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) > 1:
        pfac = kf.get_number(data[1], STD8, block.long, 2)
        if pfac is not None and pfac < 0:
            lcid = int(-pfac)
            if not edit:
                ctx.register_curve(lcid, LENGTH, PRESSURE, "CLIS PFAC curve")
    if edit:
        if len(data) > 1:
            kf.scale_field(data[1], STD8, block.long, 0, ctx.fac(TIME))
            kf.scale_field(data[1], STD8, block.long, 1, ctx.fac(TIME))
        ctx.count("CONSTRAINED_LAGRANGE_IN_SOLID")


def h_database_dt(block: Block, ctx, edit: bool) -> None:
    if not edit:
        return
    kf = ctx.kf
    if block.data:
        kf.scale_field(block.data[0], STD8, block.long, 0, ctx.fac(TIME))
        ctx.count("DATABASE_* (dt)")


def h_icfd_prescribed_vel(block: Block, ctx, edit: bool) -> None:
    """R16 Vol III p.7-32..7-33 (*ICFD_BOUNDARY_PRESCRIBED_VEL), repeating
    cards PID DOF VAD LCID SF VID DEATH BIRTH.  The LCID curve carries the
    motion value versus time; VAD picks its ordinate dimension (1 = linear
    velocity, 2 = angular velocity, 3 = parabolic velocity profile).  VAD = 4
    (synthetic turbulent field, *ICFD_CONTROL_TURB_SYNTHESIS) does not
    document the curve's meaning, so it is refused.  DEATH/BIRTH are times."""
    kf = ctx.kf
    for li in block.data:
        vad = _numint(kf, li, STD8, block.long, 2) or 1
        lcid = _numint(kf, li, STD8, block.long, 3)
        ydim = {1: VELOCITY, 2: ANG_VEL, 3: VELOCITY}.get(vad)
        if ydim is None:
            if lcid:
                ctx.error(f"*{block.name}: VAD={vad} is not modelled (the "
                          "manual does not document the curve's dimension "
                          "for synthetic turbulence) - convert manually.")
            continue
        if not edit:
            if lcid:
                ctx.register_curve(lcid, TIME, ydim, block.name)
        else:
            kf.scale_field(li, STD8, block.long, 6, ctx.fac(TIME))  # DEATH
            kf.scale_field(li, STD8, block.long, 7, ctx.fac(TIME))  # BIRTH
    if edit:
        ctx.count(block.name)


def h_icfd_prescribed_pre(block: Block, ctx, edit: bool) -> None:
    """R16 Vol III p.7-25 (*ICFD_BOUNDARY_PRESCRIBED_PRE), repeating cards
    PID LCID SF DEATH BIRTH ISO.  The LCID curve is pressure versus time;
    DEATH/BIRTH are times; SF and ISO stay."""
    kf = ctx.kf
    for li in block.data:
        lcid = _numint(kf, li, STD8, block.long, 1)
        if not edit:
            if lcid:
                ctx.register_curve(lcid, TIME, PRESSURE, block.name)
        else:
            kf.scale_field(li, STD8, block.long, 3, ctx.fac(TIME))  # DEATH
            kf.scale_field(li, STD8, block.long, 4, ctx.fac(TIME))  # BIRTH
    if edit:
        ctx.count(block.name)


def h_icfd_control_time(block: Block, ctx, edit: bool) -> None:
    """R16 Vol III p.7-82..7-84 (*ICFD_CONTROL_TIME): Card1 TTM DT CFL LCIDSF
    DTMIN DTMAX DTINIT TDEATH; optional Card2 DTT; optional Card3 DTBL DTST
    DTVISC (flags); optional Card4 IDR DTDR CFLDR LCIDSFDR DTMINDR DTMAXDR
    DTINITDR.  TTM/DT*/TDEATH are times; CFL[DR] is dimensionless; LCIDSF[DR]
    is a time-step scale-factor curve; negative DTMIN/DTMAX (p.7-83) point to
    time-dependent curves instead of holding a value."""
    kf = ctx.kf
    data = list(block.data)
    if not data:
        return

    def timestep_card(li, tfields, lcid_fi, neg_curve):
        lcid = _numint(kf, li, STD8, block.long, lcid_fi)
        if not edit and lcid:
            ctx.register_curve(lcid, TIME, DIMLESS, block.name + " LCIDSF")
        for fi in tfields:
            v = kf.get_number(li, STD8, block.long, fi)
            if v is not None and v < 0 and fi in neg_curve:
                if not edit:
                    ctx.register_curve(int(-v), TIME, TIME,
                                       block.name + " DTMIN/DTMAX")
                continue
            if edit:
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(TIME))

    timestep_card(data[0], (0, 1, 4, 5, 6, 7), 3, {4, 5})
    if len(data) > 1 and edit:
        kf.scale_field(data[1], STD8, block.long, 0, ctx.fac(TIME))   # DTT
    # data[2] (DTBL DTST DTVISC) holds flags only
    if len(data) > 3:
        timestep_card(data[3], (1, 4, 5, 6), 3, {4, 5})
    if edit:
        ctx.count(block.name)


def h_mesh_bl(block: Block, ctx, edit: bool) -> None:
    """R16 Vol III p.8-2..8-3 (*MESH_BL), repeating cards PID NELTH BLTH BLFE
    BLST BLDR.  BLTH is the boundary-layer thickness (length) for BLST = 1/2
    but a growth scale factor for BLST = 3; BLFE is a scale coefficient for
    BLST = 1/2 but the wall distance (length) for BLST = 3; both are ignored
    for BLST = 0.  Negative NELTH/BLTH/BLFE reference time-dependent load
    curves (Remark 5) and must not be rescaled."""
    kf = ctx.kf
    for li in block.data:
        blst = _numint(kf, li, STD8, block.long, 4) or 0
        dims = {1: DIMLESS}                       # NELTH: element count
        if blst in (1, 2):
            dims.update({2: LENGTH, 3: DIMLESS})  # BLTH thickness, BLFE coeff
        elif blst == 3:
            dims.update({2: DIMLESS, 3: LENGTH})  # BLTH factor, BLFE distance
        for fi, dim in dims.items():
            v = kf.get_number(li, STD8, block.long, fi)
            if v is None:
                continue
            if v < 0:
                if not edit:
                    ctx.register_curve(int(-v), TIME, dim,
                                       f"{block.name} (negative field "
                                       f"{fi + 1})")
                continue
            if edit and dim is not DIMLESS:
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(dim))
    if edit:
        ctx.count(block.name)


def h_blast(block: Block, ctx, edit: bool) -> None:
    """*LOAD_BLAST_ENHANCED and legacy *LOAD_BLAST."""
    kf = ctx.kf
    legacy = block.name == "LOAD_BLAST"
    data = list(block.data)
    i = 0
    while i < len(data):
        li1 = data[i]
        if legacy:
            m_f, coord_f, tbo_f, unit_f = 0, (1, 2, 3), 4, 5
        else:
            m_f, coord_f, tbo_f, unit_f = 1, (2, 3, 4), 5, 6
        unit = _numint(kf, li1, STD8, block.long, unit_f) or 2
        blast_type = (_numint(kf, li1, STD8, block.long, 7) or 2) if not legacy else 0
        if not edit:
            us = BLAST_UNIT_SYSTEMS.get(unit)
            if us is not None and ctx.src is not None and us != ctx.src:
                ctx.warn(
                    f"*{block.name}: UNIT={unit} declares {us.key} but the "
                    f"deck is being converted as {ctx.src.key} - the flag and "
                    "the model units disagree in the SOURCE deck. The charge "
                    "mass / location are scaled as MODEL units; verify.")
            nsets = 2 if not legacy else (2 if len(data) - i > 1 else 1)
            if blast_type in (3, 4):
                nsets = 3
            i += nsets
            continue
        # -- card 1: scale M, XBO/YBO/ZBO, TBO; rewrite UNIT ------------------
        kf.scale_field(li1, STD8, block.long, m_f, ctx.fac(MASS))
        for fi in coord_f:
            kf.scale_field(li1, STD8, block.long, fi, ctx.fac(LENGTH))
        kf.scale_field(li1, STD8, block.long, tbo_f, ctx.fac(TIME))
        key = (ctx.dst.mass, ctx.dst.length, ctx.dst.time)
        builtin = BLAST_BUILTIN_UNITS.get(key)
        if legacy and builtin not in (1, 2, 3, 4):
            builtin = None
        use5 = ctx.opts.get("blast_unit") == 5 or builtin is None
        if use5 and legacy and len(data) - i < 2:
            ctx.error("*LOAD_BLAST: target system needs UNIT=5 conversion "
                      "factors but the deck has no Card 2 - add one or use "
                      "*LOAD_BLAST_ENHANCED.")
            return
        new_unit = 5 if use5 else builtin
        w = 20 if block.long else 10
        kf.set_field(li1, STD8, block.long, unit_f, str(new_unit).rjust(w))
        # -- card 2: CFM CFL CFT CFP [NIDBO DEATH NEGPHS] ---------------------
        consumed = 1
        if i + 1 < len(data):
            li2 = data[i + 1]
            if use5:
                cfs = blast_unit5_factors(ctx.dst)
                for fi, v in enumerate(cfs):
                    s = f"{v:.9G}"[:w]
                    kf.set_field(li2, STD8, block.long, fi, s.rjust(w))
            else:
                for fi in range(4):
                    kf.set_field(li2, STD8, block.long, fi, "0.0".rjust(w))
            if not legacy:
                kf.scale_field(li2, STD8, block.long, 5, ctx.fac(TIME))  # DEATH
            consumed = 2
        if blast_type == 3 and i + 2 < len(data):
            kf.scale_field(data[i + 2], STD8, block.long, 0, ctx.fac(VELOCITY))
            ctx.warn("*LOAD_BLAST_ENHANCED BLAST=3: TEMP is in Fahrenheit "
                     "(unchanged); verify VEL/RATIO card.")
            consumed = 3
        elif blast_type == 4 and i + 2 < len(data):
            consumed = 3
        ctx.count(block.name + f" (UNIT->{new_unit})")
        i += consumed


def h_mat_cscm(block: Block, ctx, edit: bool) -> None:
    """*MAT_CSCM[_CONCRETE] (MAT_159), R16 Vol II p.2-1081..2-1093.

    CONCRETE variant: Card 1 MID RO NPLOT INCRE IRATE ERODE RECOV ITRETRC -
    only RO is dimensional (INCRE is a strain increment, IRATE/ITRETRC are
    flags, ERODE a principal-strain threshold/flag and RECOV a 0-1 / 10-11
    recovery ratio, p.2-1082..83); Card 2 PRED (damage fraction); Card 3
    FPC DAGG UNITS where FPC is a pressure, DAGG a length and UNITS declares
    the unit system they are in (table on p.2-1084).  LS-DYNA fits the
    concrete parameters internally from those three fields, so UNITS must be
    remapped to the destination system's value - and the conversion must
    abort when the destination has no UNITS value.

    The user-defined <BLANK> variant (Cards 4-8, p.2-1084..2-1087) is
    refused: the manual gives no units for CH (hardening rate) and the
    ETA0C/ETA0T rate parameters have value-dependent dimensions (the
    fluidity eta = ETA0x / rate^Nx is a time, p.2-1092), so a fixed
    per-field factor table cannot scale it safely."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if "CONCRETE" not in block.name.split("_"):
        if edit:
            ctx.error(f"*{block.name}: user-defined MAT_159 input (Cards "
                      "4-8) is not auto-convertible - the CH hardening rate "
                      "has no documented units and ETA0C/ETA0T dimensions "
                      "depend on the NC/NT values (R16 Vol II p.2-1085, "
                      "p.2-1092) - convert this material manually.")
        return
    if len(data) < 3:
        if edit:
            ctx.error(f"*{block.name}: expected 3 cards (MID/RO..., PRED, "
                      f"FPC/DAGG/UNITS), found {len(data)}.")
        return
    units = _numint(kf, data[2], STD8, block.long, 2) or 0
    if not edit:
        us = CSCM_UNIT_SYSTEMS.get(units)
        if us is None:
            ctx.warn(f"*{block.name}: UNITS={units} is not a documented "
                     "MAT_159 value (0-4, R16 Vol II p.2-1084) - it will be "
                     "rewritten to the destination system's value; verify "
                     "the source deck.")
        elif ctx.src is not None and us != ctx.src:
            ctx.warn(f"*{block.name}: UNITS={units} declares {us.key} but "
                     f"the deck is being converted as {ctx.src.key} - the "
                     "flag and the model units disagree in the SOURCE deck. "
                     "FPC/DAGG are scaled as MODEL units; verify.")
        return
    key = (ctx.dst.mass, ctx.dst.length, ctx.dst.time)
    new_units = CSCM_UNITS.get(key)
    if new_units is None:
        supported = ", ".join(f"{v}={s.key}"
                              for v, s in sorted(CSCM_UNIT_SYSTEMS.items()))
        ctx.error(f"*{block.name}: MAT_159 auto-generates its concrete "
                  "parameters from FPC/DAGG/UNITS, but the UNITS table "
                  f"(R16 Vol II p.2-1084) has no value for {ctx.dst.key}. "
                  f"Supported destinations: {supported}. Pick one of those "
                  "target systems or convert this material manually.")
        return
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(DENSITY))   # RO
    kf.scale_field(data[2], STD8, block.long, 0, ctx.fac(PRESSURE))  # FPC
    kf.scale_field(data[2], STD8, block.long, 1, ctx.fac(LENGTH))    # DAGG
    w = 20 if block.long else 10
    kf.set_field(data[2], STD8, block.long, 2, str(new_units).rjust(w))
    if len(data) > 3:
        ctx.warn(f"*{block.name}: {len(data) - 3} trailing card(s) beyond "
                 "the 3-card CONCRETE layout left unscaled - verify.")
    ctx.count(block.name + f" (UNITS->{new_units})")


def h_cnrb_inertia(block: Block, ctx, edit: bool) -> None:
    if not edit:
        return
    kf = ctx.kf
    d = block.data
    plans = [{}, {0: LENGTH, 1: LENGTH, 2: LENGTH, 3: MASS},
             {i: INERTIA for i in range(6)},
             {0: VELOCITY, 1: VELOCITY, 2: VELOCITY,
              3: ANG_VEL, 4: ANG_VEL, 5: ANG_VEL}]
    for ci, li in enumerate(d[:4]):
        for fi, dim in plans[ci].items():
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(dim))
    ctx.count("CONSTRAINED_NODAL_RIGID_BODY_INERTIA")


def h_part_composite(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.37-18..37-27 (*PART_COMPOSITE[_LONG]): Card1 HEADING;
    irregular optional Card2 (only if it starts with 'OPTCARD'; OPTC IRPL,
    flags only); Card3a PID ELFORM SHRF NLOC MAREA HGID ADPOPT THSHEL
    (MAREA = mass/area); then repeating layer cards, two layers per card
    MID1 THICK1 B1 TMID1 MID2 THICK2 B2 TMID2 or one per card with LONG
    (MID1 THICK1 B1 TMID1 PLYID1 SHRFAC1).  THICKi is a length; Bi is a
    material angle in degrees.  The TSHELL / IGA_SHELL / CONTACT variants
    have different card layouts and deliberately stay unknown."""
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    idx = 1                                      # skip HEADING
    if (idx < len(data)
            and kf.lines[data[idx]].lstrip().upper().startswith("OPTCARD")):
        idx += 1
    if idx < len(data):
        kf.scale_field(data[idx], STD8, block.long, 4, ctx.fac(MASS_AREA))
        idx += 1
    thick_fields = (1,) if block.name.endswith("_LONG") else (1, 5)
    for li in data[idx:]:
        for fi in thick_fields:
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(LENGTH))
    ctx.count(block.name)


def x_scan_part(block: Block, ctx) -> None:
    """Collect (pid, secid, mid) so DRO can classify discrete materials."""
    kf = ctx.kf
    data = list(block.data)
    for i in range(1, len(data), 2):     # heading, ids, heading, ids ...
        pid = _numint(kf, data[i], STD8, block.long, 0)
        secid = _numint(kf, data[i], STD8, block.long, 1)
        mid = _numint(kf, data[i], STD8, block.long, 2)
        if pid:
            ctx.part_links.append((pid, secid, mid))


CUSTOM: Dict[str, Callable] = {
    "DEFINE_CURVE": h_define_curve,
    "DEFINE_TABLE": h_define_table,
    "ELEMENT_SPH": h_element_sph,
    "ELEMENT_SPH_VOLUME": h_element_sph,
    "LOAD_BLAST_ENHANCED": h_blast,
    "LOAD_BLAST": h_blast,
    "LOAD_NODE_POINT": h_load_node_or_rb,
    "LOAD_NODE_SET": h_load_node_or_rb,
    "LOAD_RIGID_BODY": h_load_node_or_rb,
    "PART_COMPOSITE": h_part_composite,
    "PART_COMPOSITE_LONG": h_part_composite,
    "SECTION_BEAM": h_section_beam,
    "SECTION_DISCRETE": h_section_discrete,
    "MAT_SPRING_ELASTIC": h_smat_spring_elastic,
    "MAT_SPRING_ELASTOPLASTIC": h_smat_spring_elastoplastic,
    "MAT_DAMPER_VISCOUS": h_smat_damper_viscous,
    "MAT_SPRING_NONLINEAR_ELASTIC": h_smat_curve_mats,
    "MAT_DAMPER_NONLINEAR_VISCOUS": h_smat_curve_mats,
    "MAT_CSCM": h_mat_cscm,
    "MAT_CSCM_CONCRETE": h_mat_cscm,
    "ICFD_BOUNDARY_PRESCRIBED_VEL": h_icfd_prescribed_vel,
    "ICFD_BOUNDARY_PRESCRIBED_PRE": h_icfd_prescribed_pre,
    "ICFD_CONTROL_TIME": h_icfd_control_time,
    "MESH_BL": h_mesh_bl,
    "CONSTRAINED_LAGRANGE_IN_SOLID": h_lagrange_in_solid,
    "CONSTRAINED_NODAL_RIGID_BODY_INERTIA": h_cnrb_inertia,
    "INITIAL_STRESS_SHELL": h_initial_stress_shell,
    "INITIAL_STRESS_SHELL_SET": h_initial_stress_shell,
    "INITIAL_STRESS_SOLID": h_initial_stress_solid,
    "INITIAL_STRESS_SOLID_SET": h_initial_stress_solid,
}


def resolve(name: str):
    """Classify a keyword. Returns (kind, payload):
    kind in {spec, custom, white, soft, hard, unknown}."""
    base = name
    for opt in ("_TITLE", "_ID"):
        if base.endswith(opt):
            base = base[: -len(opt)]
    base = _MAT_ALIASES.get(base, base)
    if name in HARD_FLAGS:
        return "hard", HARD_FLAGS[name]
    if base in CUSTOM:
        return "custom", CUSTOM[base]
    if base in SPECS:
        return "spec", SPECS[base]
    if base in WHITELIST or name in WHITELIST:
        return "white", None
    for p in WHITELIST_PREFIXES:
        if name.startswith(p):
            return "white", None
    if name.startswith("LOAD_BODY_"):
        return "custom", h_load_body
    if name.startswith("BOUNDARY_PRESCRIBED_MOTION"):
        return "custom", h_prescribed_motion
    if name.startswith("RIGIDWALL_PLANAR"):
        return "custom", h_rigidwall_planar
    if name.startswith("CONTACT_"):
        if (name.startswith("CONTACT_TIEBREAK") or "DRAWBEAD" in name
                or "MPP" in name or "DAMPING" in name or "MORTAR" in name):
            return "unknown", None
        return "custom", h_contact
    if name.startswith("DATABASE_BINARY_"):
        tail = name[len("DATABASE_BINARY_"):]
        if tail in ("D3DUMP", "RUNRSF", "D3DRLF"):
            return "white", None
        return "custom", h_database_dt
    if name.startswith("DATABASE_"):
        return "custom", h_database_dt          # ascii files: field 0 = dt
    if name.startswith("CONTROL_"):
        return "soft", ("not in the dimension table - left unchanged; "
                        "most CONTROL cards are flags, but verify")
    return "unknown", None


# scan-time extras for keywords that already have a Spec
SCAN_EXTRA: Dict[str, Callable] = {
    "MAT_PIECEWISE_LINEAR_PLASTICITY": h_mat_024,
    "PART": x_scan_part,
}
# edit-time extras (warnings that need field values)
EDIT_EXTRA: Dict[str, Callable] = {
    "MAT_ELASTIC": x_mat_001,
    "MAT_ELASTIC_FLUID": x_mat_001,
    "MAT_JOHNSON_COOK": x_mat_015,
}
