"""Keyword dimension database + custom handlers.

Every keyword encountered in a deck must resolve to one of:
  * a Spec (table-driven field scaling),
  * a custom handler (flag-dependent formats: curves, blast, motion...),
  * the dimensionless whitelist (topology / ids / flags only),
  * a soft/hard flag (known-unsupported -> loud warning / abort).
Anything else is UNKNOWN and aborts the conversion unless --allow-unknown,
because silently passing a dimensional card through would corrupt physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .parser import STD8, Block, KFile, parse_number
from .units import (ACCEL, ANG_ACCEL, ANG_VEL, AREA, DC_FRIC, DENSITY, Dim,
                    DIMLESS, FORCE, FREQ, INERTIA, L4, LENGTH, MASS, MASS_AREA,
                    MASS_LEN, MOMENT, PRESSURE, RATE, TIME, VELOCITY, VISCOSITY,
                    BLAST_BUILTIN_UNITS, BLAST_UNIT_SYSTEMS, blast_unit5_factors)

STRAIN = DIMLESS


@dataclass
class Card:
    dims: Dict[int, Dim] = dfield(default_factory=dict)
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

    # ── materials (field maps cross-checked against the R16 manual / k2rad) ─
    "MAT_ELASTIC": Spec(cards=[C({1: DENSITY, 2: PRESSURE, 6: PRESSURE})],
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
    "INITIAL_VELOCITY_GENERATION": Spec(group=[
        C({2: ANG_VEL, 3: VELOCITY, 4: VELOCITY, 5: VELOCITY}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH})]),
    "INITIAL_DETONATION": Spec(repeat=C(
        {1: LENGTH, 2: LENGTH, 3: LENGTH, 4: TIME})),
    "CONSTRAINED_SPOTWELD": Spec(repeat=C({2: FORCE, 3: FORCE, 6: TIME})),

    # ── control / database ──────────────────────────────────────────────────
    "CONTROL_TERMINATION": Spec(cards=[C({0: TIME})]),
    "CONTROL_TIMESTEP": Spec(cards=[C({0: TIME, 3: TIME, 4: TIME})],
                             curves=[(0, 5, TIME, TIME)], extra_ok=True),
    "CONTROL_DYNAMIC_RELAXATION": Spec(cards=[C({3: TIME})]),
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

    # ── section beam (elform-dependent, handled in custom below) ────────────
}

# MAT numeric aliases
_MAT_ALIASES = {
    "MAT_001": "MAT_ELASTIC", "MAT_003": "MAT_PLASTIC_KINEMATIC",
    "MAT_008": "MAT_HIGH_EXPLOSIVE_BURN", "MAT_009": "MAT_NULL",
    "MAT_020": "MAT_RIGID", "MAT_024": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "MAT_140": "MAT_VACUUM",
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
}
WHITELIST_PREFIXES = (
    "SET_", "BOUNDARY_SPC", "DATABASE_HISTORY", "CONTROL_MPP_",
    "DEFORMABLE_TO_RIGID", "INTERFACE_SPRINGBACK",
)
# known-unsupported: abort (hard) or leave-with-warning (soft)
HARD_FLAGS = {
    "INCLUDE": "converts one self-contained file - convert includes separately",
    "INCLUDE_TRANSFORM": "carries its own scale factors",
    "INCLUDE_PATH": "converts one self-contained file",
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


def h_define_curve(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = list(block.data)
    if "TITLE" in block.name.split("_"):
        data = data[1:]
    if not data:
        return
    lcid = _numint(kf, data[0], STD8, block.long, 0)
    if not edit:
        ctx.curve_blocks.setdefault(lcid, []).append(block)
        return
    dims = ctx.curve_dims.get(lcid)
    if not dims:
        ctx.warn(f"*{block.name} lcid={lcid}: no referencing keyword tells me "
                 "its axis dimensions - data points left UNCHANGED. Scale "
                 "manually if this curve is dimensional.")
        ctx.count(block.name + " (unreferenced, unchanged)")
        return
    if len(dims) > 1:
        ctx.error(f"*{block.name} lcid={lcid}: conflicting dimension demands "
                  f"from referencers: {sorted(dims.items())}")
        return
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
    data = list(block.data)
    if "TITLE" in block.name.split("_"):
        data = data[1:]
    if not data:
        return
    tbid = _numint(kf, data[0], STD8, block.long, 0)
    if not edit:
        ctx.table_blocks[tbid] = block
        return
    entry = ctx.table_dims.get(tbid)
    if not entry:
        ctx.warn(f"*{block.name} tbid={tbid}: unreferenced - left UNCHANGED.")
        return
    vdim, _xd, _yd = entry
    fv = ctx.fac(vdim)
    kf.scale_field(data[0], STD8, block.long, 2, fv)   # OFFA
    for li in data[1:]:
        kf.scale_field(li, (20, 20), block.long, 0, fv)
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
    data = list(block.data)
    opts = block.name.split("_")
    if "ID" in opts or "TITLE" in opts:
        data = data[1:]
    plan = [
        {},                                              # card1: ids
        {2: DC_FRIC, 3: PRESSURE, 6: TIME, 7: TIME},     # card2
        {2: LENGTH, 3: LENGTH},                          # card3: SST MST
        {},                                              # A (flags)
        {6: LENGTH, 7: PRESSURE},                        # B: SLDTHK SLDSTF
    ]
    for ci, li in enumerate(data):
        if ci >= len(plan):
            ctx.warn(f"*{block.name}: optional card {ci + 1} left unscaled "
                     "(advanced options C+ not modelled) - verify manually.")
            break
        for fi, dim in plan[ci].items():
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(dim))
    ctx.count("CONTACT_*")


def h_mat_024(block: Block, ctx, edit: bool) -> None:
    """Curve/table registration for MAT_PIECEWISE_LINEAR_PLASTICITY."""
    kf = ctx.kf
    data = list(block.data)
    if "TITLE" in block.name.split("_"):
        data = data[1:]
    if len(data) > 1:
        lcss = _numint(kf, data[1], STD8, block.long, 2)
        lcsr = _numint(kf, data[1], STD8, block.long, 3)
        if lcss:
            ctx.register_curve(lcss, STRAIN, PRESSURE, "MAT_024 LCSS")
            ctx.register_table(lcss, RATE, STRAIN, PRESSURE)
        if lcsr:
            ctx.register_curve(lcsr, RATE, DIMLESS, "MAT_024 LCSR")


def h_section_beam(block: Block, ctx, edit: bool) -> None:
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    if "TITLE" in block.name.split("_"):
        data = data[1:]
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


def h_lagrange_in_solid(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = list(block.data)
    if "ID" in block.name.split("_") or "TITLE" in block.name.split("_"):
        data = data[1:]
    if len(data) > 1:
        pfac = kf.get_number(data[1], STD8, block.long, 2)
        if pfac is not None and pfac < 0:
            lcid = int(-pfac)
            if not edit:
                ctx.register_curve(lcid, LENGTH, PRESSURE,
                                   "CLIS PFAC curve")
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


def h_blast(block: Block, ctx, edit: bool) -> None:
    """*LOAD_BLAST_ENHANCED and legacy *LOAD_BLAST."""
    kf = ctx.kf
    legacy = block.name == "LOAD_BLAST"
    data = list(block.data)
    i = 0
    while i < len(data):
        li1 = data[i]
        if legacy:
            m_f, coord_f, tbo_f, unit_f, extra = 0, (1, 2, 3), 4, 5, None
        else:
            m_f, coord_f, tbo_f, unit_f, extra = 1, (2, 3, 4), 5, 6, 7
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


CUSTOM: Dict[str, Callable] = {
    "DEFINE_CURVE": h_define_curve,
    "DEFINE_TABLE": h_define_table,
    "LOAD_BLAST_ENHANCED": h_blast,
    "LOAD_BLAST": h_blast,
    "LOAD_NODE_POINT": h_load_node_or_rb,
    "LOAD_NODE_SET": h_load_node_or_rb,
    "LOAD_RIGID_BODY": h_load_node_or_rb,
    "SECTION_BEAM": h_section_beam,
    "CONSTRAINED_LAGRANGE_IN_SOLID": h_lagrange_in_solid,
    "CONSTRAINED_NODAL_RIGID_BODY_INERTIA": h_cnrb_inertia,
}


def resolve(name: str):
    """Classify a keyword. Returns (kind, payload):
    kind in {spec, custom, white, soft, hard, unknown}."""
    base = name
    # strip trailing option tokens that only add a heading line
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
    if name.startswith("CONTACT_"):
        if "TIEBREAK" in name or "DRAWBEAD" in name or "MPP" in name:
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


# extra registration hook: MAT_024 needs a scan even though it has a Spec
SCAN_EXTRA: Dict[str, Callable] = {
    "MAT_PIECEWISE_LINEAR_PLASTICITY": h_mat_024,
}
