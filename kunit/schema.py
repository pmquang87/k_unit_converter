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
from decimal import Decimal
from fractions import Fraction
from typing import (Callable, Dict, List, Optional, Sequence, Tuple,
                    TYPE_CHECKING)

from .parser import STD8, Block, KFile, format_fixed, parse_number

if TYPE_CHECKING:
    # imported for handler type annotations only; convert.py imports schema.py
    # at runtime, so importing it here would create a circular import.
    from .convert import Ctx
from .units import (ACCEL, ACCEL_PSD, ACOUST_IMP, ANG_ACCEL, ANG_VEL, AREA, DAMP, DC_FRIC,
                    DENSITY, Dim, DIM_NAMES, DIMLESS, DISP_PSD, FORCE,
                    FORCE_PSD, FREQ, HEAT_FLUX, INERTIA, INV_PRESSURE, L4,
                    LENGTH,
                    MASS, MASS_AREA, MASS_LEN, MOMENT, POWER, PRES_PSD,
                    PRESSURE, PWR_VOL, RATE,
                    ROT_DAMP, SPEC_HEAT, STIFF, STIFF_LEN, STRESS_M3,
                    STRESS_SQ, TEMP,
                    THERM_COND, TIME, VEL_PSD, VELOCITY, VISCOSITY, VOLUME,
                    BEM_UNITS, BEM_UNIT_SYSTEMS,
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
EOS_TAB_W = (16,) * 5   # *EOS_TABULATED data cards: 5 values over 10 slots

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
    # R16 Vol I p.19-99 (*ELEMENT_SHELL): the "Thickness Card" is the
    # additional card for the THICKNESS, BETA *and* MCID options alike, and
    # always carries THIC1-THIC4 (nodal thicknesses, 4 x 16 chars) before the
    # 5th field (BETA angle in degrees / MCID material-system id).  So the
    # BETA and MCID spellings hold four LENGTH fields too and must not be
    # whitelisted - they share the THICKNESS layout.
    "ELEMENT_SHELL_THICKNESS": Spec(group=[
        C({}, (8,) * 10),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH}, (16,) * 5)]),
    "ELEMENT_SHELL_BETA": Spec(group=[
        C({}, (8,) * 10),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH}, (16,) * 5)]),
    "ELEMENT_SHELL_MCID": Spec(group=[
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
    # R16 Vol II p.2-170..2-173 (*MAT_SOIL_AND_FOAM / MAT_005): six cards,
    # none optional.  Card 1 MID RO G KUN A0 A1 A2 PC (LS-PrePost labels KUN
    # "bulk"); Card 2 VCR REF LCID; Cards 3-4 the ten volumetric strains
    # EPS1-EPS10 ("the natural log of the relative volume", dimensionless);
    # Cards 5-6 the ten pressures P1-P10.
    #
    # A0/A1/A2 do NOT share a dimension.  Remark 2 gives the yield surface as
    # phi = J2 - [a0 + a1 p + a2 p^2] with "J2 = 1/2 s_ij s_ij", a product of
    # two stresses, so the bracket is a stress^2: a0 is STRESS_SQ, a1 a plain
    # stress and a2 dimensionless.  The degenerate von-Mises case in the same
    # remark ("set a1 = a2 = 0 and a0 = 1/3 sigma_y^2") confirms it.
    "MAT_SOIL_AND_FOAM": Spec(
        cards=[C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 4: STRESS_SQ,
                  5: PRESSURE, 7: PRESSURE}),
               C(), C(), C(),
               C({i: PRESSURE for i in range(8)}),
               C({0: PRESSURE, 1: PRESSURE})],
        curves=[(1, 2, DIMLESS, PRESSURE)]),
    "SECTION_SOLID": Spec(cards=[C()], extra_ok=True),
    # R16 Vol I p.41-52..41-57 (*SECTION_POINT_SOURCE_MIXTURE): Card 1 SECID
    # LCIDT - LCIDVEL NIDLC1-3 IDIR, Card 2 LCMD1-8, then one repeating
    # "Source Node Card" NODEID VECID ORIFA per source node.  ORIFA is "the
    # orifice area at each point source"; everything else on the cards is an
    # id or a flag, and all the physics rides on the curves: inflator
    # stagnation temperature and average velocity against time (Remark 1),
    # and the per-species mass flow rate m_dot(t), whose (1,0,-1) signature
    # kunit already carries as DAMP.
    "SECTION_POINT_SOURCE_MIXTURE": Spec(
        cards=[C(), C()], repeat=C({2: AREA}),
        curves=[(0, 1, TIME, TEMP), (0, 3, TIME, VELOCITY)]
                + [(1, i, TIME, DAMP) for i in range(8)]),
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
    # R16 Vol II p.2-223..2-225 (*MAT_POWER_LAW_PLASTICITY / MAT_018):
    # Card1 MID RO E PR K N SRC SRP - RO mass density, E Young's modulus,
    # K strength coefficient (sigma_y = k*eps^n with dimensionless strain,
    # so stress units), N hardening exponent and SRP the Cowper-Symonds
    # exponent P (both dimensionless); SRC is the Cowper-Symonds strain-rate
    # parameter C (a strain rate, 1/time).  Card2 SIGY VP EPSF - VP is a
    # formulation flag and EPSF the plastic failure strain (dimensionless);
    # SIGY is value-dependent (yield stress, or elastic strain to yield when
    # 0 < SIGY < 0.02, p.2-224) and is handled by x_mat_018, not the table.
    "MAT_POWER_LAW_PLASTICITY": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 4: PRESSURE, 6: RATE}),
        C()],
        probe={"ro": (0, 1), "e": (0, 2)}),
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
    # R16 Vol II p.2-153..2-156 (*MAT_ORTHOTROPIC_ELASTIC / MAT_002):
    # Card1 MID RO EA EB EC PRBA PRCA PRCB - EA/EB/EC the orthotropic Young's
    # moduli (stresses), Poisson ratios dimensionless; Card2 GAB GBC GCA AOPT
    # (G/MACF...) - the three shear moduli are stresses, AOPT/MACF are flags;
    # Card3 XP YP ZP A1 A2 A3 and Card4 V1 V2 V3 D1 D2 D3 BETA REF are the AOPT
    # coordinate/vector cards (XP/YP/ZP a point, the rest direction cosines and
    # an angle), handled exactly like MAT_COMPOSITE_DAMAGE.
    "MAT_ORTHOTROPIC_ELASTIC": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 4: PRESSURE}),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH}),
        C()],
        probe={"ro": (0, 1), "e": (0, 2)}, extra_ok=True),
    # R16 Vol II p.2-262..2-271 (*MAT_ENHANCED_COMPOSITE_DAMAGE / MAT_054/055):
    # extends MAT_022.  Card1 MID RO EA EB EC PRBA PRCA PRCB; Card2 GAB GBC GCA
    # KF AOPT (KF = bulk modulus of failed material, a stress); Card3 XP YP ZP
    # A1 A2 A3 MANGLE (point + direction + angle); Card4 V1 V2 V3 D1 D2 D3
    # DFAILM DFAILS (direction cosines + failure strains); Card5 TFAIL ALPH
    # SOFT FBRT YCFAC DFAILT DFAILC EFS - all dimensionless / value-dependent
    # (TFAIL is a time-step-size criterion only for 0<TFAIL<=.1 else a ratio,
    # ALPH a 0-1 shear weight, the DFAIL*/EFS are failure strains), so Card5
    # is left unscaled; Card6 XC XT YC YT SC CRIT BETA - the five strengths
    # are stresses.  Cards 7/8 (PEL/EPSF/EPSR strains, SLIM* factors) tolerated.
    "MAT_ENHANCED_COMPOSITE_DAMAGE": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 4: PRESSURE}),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE, 3: PRESSURE}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH}),
        C(),
        C(),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE, 3: PRESSURE, 4: PRESSURE})],
        probe={"ro": (0, 1), "e": (0, 2)}, extra_ok=True),
    # R16 Vol II p.2-330..2-332 (*MAT_SIMPLIFIED_JOHNSON_COOK / MAT_098):
    # Card1 MID RO E PR VP - E the Young's modulus (stress), PR/VP flags;
    # Card2 A B N C PSFAIL SIGMAX SIGSAT EPS0 - A/B (yield & hardening
    # coefficients) and SIGMAX/SIGSAT (max & saturation stresses) are stresses;
    # N, C (the dimensionless strain-rate sensitivity coefficient, exactly as
    # in the full MAT_015 layout) and PSFAIL (failure plastic strain) are
    # dimensionless.  EPS0 (reference strain rate, default 1.0) is left
    # unscaled - verify manually if a non-default EPS0 is used and the time
    # unit changes.  An optional failure/erosion card is tolerated (extra_ok).
    "MAT_SIMPLIFIED_JOHNSON_COOK": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE}),
        C({0: PRESSURE, 1: PRESSURE, 5: PRESSURE, 6: PRESSURE})],
        probe={"ro": (0, 1), "e": (0, 2)}, extra_ok=True),
    # R16 Vol II p.2-289..2-290 (*MAT_CRUSHABLE_FOAM / MAT_063): Card1 MID RO E
    # PR LCID TSC DAMP - E the Young's modulus and TSC the tension stress
    # cutoff (both stresses); LCID is yield stress vs volumetric strain
    # (abscissa dimensionless, ordinate a stress); PR and DAMP (rate-
    # sensitivity damping coefficient) are dimensionless.
    "MAT_CRUSHABLE_FOAM": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 5: PRESSURE})],
        curves=[(0, 4, DIMLESS, PRESSURE)],
        probe={"ro": (0, 1), "e": (0, 2)}),
    # R16 Vol II p.2-274..2-277 (*MAT_LOW_DENSITY_FOAM / MAT_057): Card1 MID RO
    # E LCID TC HU BETA DAMP - E the initial Young's modulus, TC the tension
    # cutoff (stresses); LCID is nominal stress vs strain (abscissa
    # dimensionless, ordinate a stress).  HU (hysteretic unloading factor) and
    # DAMP are dimensionless; BETA (a decay constant, 1/time) and the optional
    # Card2 relaxation fields are value/model dependent and left unscaled
    # (extra_ok) - verify BETA manually if the time unit changes.
    "MAT_LOW_DENSITY_FOAM": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 4: PRESSURE})],
        curves=[(0, 3, DIMLESS, PRESSURE)],
        probe={"ro": (0, 1), "e": (0, 2)}, extra_ok=True),
    # R16 Vol II p.2-560..2-564 (*MAT_SIMPLIFIED_RUBBER/FOAM / MAT_181): Card1
    # MID RO KM MU G SIGF REF PRTEN - KM the linear bulk modulus (a stress).
    # Only RO and KM are scaled here: MU/G/SIGF/PRTEN are value/flag dependent,
    # and the Card2 specimen geometry (SGL/SW/ST lengths) plus the LC/TBID
    # stress-vs-stretch data are flag dependent (engineering stress-strain vs
    # force-displacement per the SGL/SW/ST normalisation), so that card is
    # left for a --curve override rather than mis-scaled (extra_ok).
    "MAT_SIMPLIFIED_RUBBER/FOAM": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE})],
        probe={"ro": (0, 1)}, extra_ok=True),
    # MAT_NULL / MAT_VACUUM are registered below via register_keyword() as a
    # demonstration of the one-stop hook (see end of file).
    "MAT_HIGH_EXPLOSIVE_BURN": Spec(cards=[
        C({1: DENSITY, 2: VELOCITY, 3: PRESSURE, 5: PRESSURE, 6: PRESSURE,
           7: PRESSURE})],
        probe={"ro": (0, 1), "d": (0, 2)}),
    "MAT_ADD_EROSION": Spec(cards=[
        C({2: PRESSURE}),
        C({0: PRESSURE, 1: PRESSURE, 2: PRESSURE, 5: PRESSURE, 6: VISCOSITY,
           7: TIME})], extra_ok=True),
    # R16 Vol II p.2-139..2-140 + R17 Vol II p.2-146..2-147
    # (*MAT_ADD_THERMAL_EXPANSION): ID LCID MULT LCIDY MULTY LCIDZ MULTZ TREF.
    # MULT/MULTY/MULTZ are either a scale factor on the curve or - when the
    # matching LCID is 0 - the constant expansion coefficient alpha itself.
    # alpha has units 1/temperature and kunit factors temperature out of the
    # (M,L,T) signature, so BOTH readings are dimensionless and no
    # flag-dependence arises.  TREF (R17 only, but already written by R16-era
    # decks) is a reference temperature.  Net effect: nothing is rescaled;
    # the entry exists so the three curves get an explicit "leave alone"
    # classification instead of surfacing as unresolved.
    # (If kunit ever gains real temperature conversion, which is affine, the
    # MULT fields and the curve ordinates all need a 1/temperature factor.)
    "MAT_ADD_THERMAL_EXPANSION": Spec(
        cards=[C({7: TEMP})],
        curves=[(0, 1, DIMLESS, DIMLESS), (0, 3, DIMLESS, DIMLESS),
                (0, 5, DIMLESS, DIMLESS)]),
    # R16 Vol II p.2-149..2-155 (*MAT_ANISOTROPIC_ELASTIC / MAT_002 ANISO
    # option, "5 cards follow"): Card 1b.1 MID RO C11 C12 C22 C13 C23 C33;
    # Card 1b.2 C14 C24 C34 C44 C15 C25 C35 C45; Card 1b.3 C55 C16 C26 C36
    # C46 C56 C66 AOPT.  Every Cij is a term of the 6x6 constitutive matrix
    # relating stress to (dimensionless) strain, so all 21 are stresses.
    # Card 2 XP YP ZP A1 A2 A3 MACF IHIS - only XP/YP/ZP are coordinates, the
    # a-vector components are a direction.  Card 3 V1 V2 V3 D1 D2 D3 BETA REF
    # is direction vectors, an angle in degrees and a flag - listed with an
    # empty map so the trailing-card warning does not fire.
    # NOTE: bare *MAT_002 is the ORTHO option with a different 4-card layout
    # and must NOT be aliased here.
    "MAT_ANISOTROPIC_ELASTIC": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 4: PRESSURE, 5: PRESSURE,
           6: PRESSURE, 7: PRESSURE}),
        C({i: PRESSURE for i in range(8)}),
        C({i: PRESSURE for i in range(7)}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH}),
        C()],
        probe={"ro": (0, 1)}),
    # R16 Vol II p.2-183..2-188 (*MAT_ELASTIC_PLASTIC_HYDRO / MAT_010): the
    # base option is 5 cards (Card 2 exists if and only if the SPALL option
    # is used).  Card 1 MID RO G SIG0 EH PC FS CHARL - G shear modulus, SIG0
    # yield stress, EH plastic hardening modulus and PC the (<= 0) pressure
    # cutoff are all stresses, FS an erosion strain, CHARL the characteristic
    # element thickness for deletion.  Cards 3/4 are EPS1..EPS16 (effective
    # plastic strains, dimensionless) and Cards 5/6 ES1..ES16 (effective
    # stresses).  The pressure branch comes from the companion *EOS.
    # No density probe on purpose: a hydro material always shares its RO with
    # the plasticity/EOS material beside it, so the duplicate evidence only
    # dilutes the discriminating modulus evidence - it pushed the Taylor-test
    # decks over the ambiguity threshold in the corpus.
    "MAT_ELASTIC_PLASTIC_HYDRO": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 3: PRESSURE, 4: PRESSURE, 5: PRESSURE,
           7: LENGTH}),
        C(), C(),
        C({i: PRESSURE for i in range(8)}),
        C({i: PRESSURE for i in range(8)})]),
    # R16 Vol II p.2-167..2-169 (*MAT_ELASTIC_PLASTIC_THERMAL / MAT_004):
    # fixed 7-card keyword.  Card 1 MID RO; Card 2 T1..T8 temperatures;
    # Card 3 E1..E8 moduli; Card 4 PR1..PR8 Poisson ratios; Card 5
    # ALPHA1..ALPHA8 thermal expansion coefficients (1/temperature, hence
    # dimensionless here); Card 6 SIGY1..SIGY8 yield stresses; Card 7
    # ETAN1..ETAN8 plastic hardening moduli.  Cards 4 and 5 are listed empty
    # so the trailing-card warning does not fire.
    "MAT_ELASTIC_PLASTIC_THERMAL": Spec(cards=[
        C({1: DENSITY}),
        C({i: TEMP for i in range(8)}),
        C({i: PRESSURE for i in range(8)}),
        C(), C(),
        C({i: PRESSURE for i in range(8)}),
        C({i: PRESSURE for i in range(8)})]),
    # R16 Vol II p.2-437..2-441 (*MAT_LOW_DENSITY_FOAM / MAT_057):
    # Card 1 MID RO E LCID TC HU BETA DAMP - E is the tensile Young's
    # modulus, TC the tensile stress cut-off, HU a 0..1 hysteretic unloading
    # factor and BETA the unloading creep decay constant of 1 - exp(-BETA*t),
    # i.e. a reciprocal time.  DAMP is the 0.05..0.50 viscous coefficient
    # (and, when negative, a curve id) - dimensionless in both roles.
    # Card 2 SHAPE FAIL BVFLAG ED BETA1 KCON REF - ED is the Young's
    # relaxation modulus of g(t) = Ed*exp(-BETA1*t), BETA1 its decay constant
    # (again 1/time) and KCON a contact stiffness coefficient compared
    # directly with E in Remark 6.  BETA and BETA1 are the two fields that
    # silently break a ms<->s conversion if missed.
    "MAT_LOW_DENSITY_FOAM": Spec(cards=[
        C({1: DENSITY, 2: PRESSURE, 4: PRESSURE, 6: RATE}),
        C({3: PRESSURE, 4: RATE, 5: PRESSURE})],
        curves=[(0, 3, DIMLESS, PRESSURE)],
        probe={"ro": (0, 1)}),
    # R16 Vol II p.2-278..2-281 (*MAT_MOONEY-RIVLIN_RUBBER / MAT_027):
    # Card 1 MID RO PR A B REF - W = A(I-3) + B(II-3) + ... with dimensionless
    # Cauchy-Green invariants and "2(A+B) is the shear modulus of linear
    # elasticity", so A and B are stresses.  Card 2 SGL SW ST LCID is the
    # least-squares specimen: gauge length, width and thickness.  Unlike
    # MAT_031, A and B are merely IGNORED when the fit is used - never
    # reinterpreted as inclusion flags - so a plain Spec is safe.
    "MAT_MOONEY-RIVLIN_RUBBER": Spec(cards=[
        C({1: DENSITY, 3: PRESSURE, 4: PRESSURE}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH})],
        curves=[(1, 3, LENGTH, FORCE)],
        probe={"ro": (0, 1)}),
    # R16 Vol II p.2-645..2-646 (*MAT_ACOUSTIC / MAT_090): Card 1 MID RO C
    # BETA CF ATMOS GRAV - C is the sound speed, ATMOS the (optional)
    # atmospheric pressure and GRAV the gravitational acceleration constant;
    # BETA is a 0.1..1.0 "damping factor" the manual gives no formula for -
    # the unit-free recommended range is the only evidence that it is a ratio
    # and not a Rayleigh alpha, so it is left unscaled.  Card 2 XP YP ZP XN
    # YN ZN - a point on the free surface plus the DIRECTION COSINES of its
    # normal, which must not be scaled.  The COMPLEX / DAMP / POROUS_DB
    # variants have completely different layouts and stay unknown.
    # No density probe: the working fluid is air (RO ~ 1.2), which would drag
    # unit detection towards g-cm systems.
    "MAT_ACOUSTIC": Spec(cards=[
        C({1: DENSITY, 2: VELOCITY, 5: PRESSURE, 6: ACCEL}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH})]),
    "EOS_JWL": Spec(cards=[C({1: PRESSURE, 2: PRESSURE, 6: PRESSURE})]),
    # R16 Vol II p.1-53 (*EOS_MURNAGHAN / EOS_019): EOSID GAMMA K0 V0 with
    # p = k0[(rho/rho0)^gamma - 1], so only K0 is dimensional; GAMMA is the
    # polytropic exponent and V0 the initial RELATIVE volume.
    "EOS_MURNAGHAN": Spec(cards=[C({2: PRESSURE})]),
    # R16 Vol II p.1-30..1-31 (*EOS_TABULATED / EOS_009): Card 1 EOSID GAMA
    # E0 V0 LCC LCT, where E0 is the initial internal energy per unit
    # reference volume ("force per unit area").  Then three PAIRS of data
    # cards in the fixed order EV, C, T.  Those six cards are drawn with ten
    # 8-char slots for five variables, i.e. 5 fields of 16 chars - NOT the
    # default 8x10 - so they carry their own width tuple.  EV holds
    # volumetric strains ln(V) and T is unitless; only the two C cards
    # (block indices 3 and 4) are pressures.  All six are omitted when LCC
    # and LCT are both given, which _apply_spec tolerates.
    "EOS_TABULATED": Spec(cards=[
        C({2: PRESSURE}),
        C({}, EOS_TAB_W), C({}, EOS_TAB_W),
        C({i: PRESSURE for i in range(5)}, EOS_TAB_W),
        C({i: PRESSURE for i in range(5)}, EOS_TAB_W),
        C({}, EOS_TAB_W), C({}, EOS_TAB_W)],
        curves=[(0, 4, DIMLESS, PRESSURE), (0, 5, DIMLESS, DIMLESS)]),
    "EOS_LINEAR_POLYNOMIAL": Spec(cards=[
        C({i: PRESSURE for i in range(1, 8)}), C({0: PRESSURE})]),
    # R16 Vol II p.1-15..1-16 (*EOS_GRUNEISEN): Card1 EOSID C S1 S2 S3 GAMMAO
    # A E0 - C is the vs(vp) curve intercept (velocity units); S1-S3, GAMMAO
    # and A are unitless; E0 is the initial internal energy per unit reference
    # volume (pressure dims: p = ... + (GAMMAO + A*mu)*E).  Card2 V0 (unused)
    # LCID - V0 is the initial relative volume (dimensionless); LCID defines
    # the energy-deposition rate dE/dt as a function of time (power/volume).
    "EOS_GRUNEISEN": Spec(cards=[C({1: VELOCITY, 7: PRESSURE}), C()],
                          curves=[(1, 2, TIME, PWR_VOL)]),

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
    # R16 Vol I p.28-115..28-116 (*INITIAL_TEMPERATURE_NODE / _SET): one
    # repeating card NID-or-NSID TEMP LOC ("include one card for each node or
    # node set").  TEMP is classified and reported but never rescaled; LOC is
    # the -1/0/1 thick-thermal-shell surface flag.  (The manual's Type row
    # prints "I" for TEMP, but its default is "0." and decks write floats.)
    "INITIAL_TEMPERATURE_SET": Spec(repeat=C({1: TEMP})),
    "INITIAL_TEMPERATURE_NODE": Spec(repeat=C({1: TEMP})),
    # R16 Vol I p.28-40..28-41 (*INITIAL_GAS_MIXTURE): Card 1 SID STYPE MMGID
    # TEMP, Card 2 RO1-RO8 "initial densities of the ALE material(s) ... for
    # up to eight different gas species", in the species order of the
    # *MAT_GAS_MIXTURE they belong to.  One keyword instance per AMMG, so the
    # two cards are a fixed set and not a repeat.
    "INITIAL_GAS_MIXTURE": Spec(cards=[
        C({3: TEMP}), C({i: DENSITY for i in range(8)})]),
    # R17 Vol I p.30-77 (*INTERFACE_LINKING_SEGMENT): SSID IFID DEATH,
    # repeating.  R16 documents only SSID and IFID, but LS-PrePost already
    # writes the third column and DEATH is a real death time - whitelisting
    # on the strength of R16 alone would leave it unconverted.
    "INTERFACE_LINKING_SEGMENT": Spec(repeat=C({2: TIME})),
    # R16 Vol I p.28-91: ISSID CSID LCID PSID VID IZSHEAR ISTIFF
    "INITIAL_STRESS_SECTION": Spec(cards=[C()],
                                   curves=[(0, 2, TIME, PRESSURE)]),
    "CONSTRAINED_SPOTWELD": Spec(repeat=C({2: FORCE, 3: FORCE, 6: TIME})),

    # ── constrained (R16 Vol I) ─────────────────────────────────────────────
    # p.10-24..10-26 + p.10-31..10-32 (*CONSTRAINED_GENERALIZED_WELD_BUTT):
    # Card 1 NSID CID FILTER WINDOW NPR NPRT - FILTER is a saved-vector COUNT
    # and WINDOW the time window for filtering (the R16 Type row prints "I",
    # but the identical TW of *CONSTRAINED_SPOTWELD_FILTERED_FORCE, p.10-211,
    # is typed F with the same wording, so it is a TIME).  Card 2c TFAIL EPSF
    # SIGY BETA L D [Lt] - TFAIL failure time, SIGY the brittle failure STRESS
    # sigma_f (the name is a red herring), L the weld length and D its
    # thickness; EPSF and BETA are dimensionless.  Field 6 (Lt, the transverse
    # butt-weld length) is dropped from the R16/R17 card tables but still
    # appears in the manual's own example header on p.10-32 - marking it
    # LENGTH is defensive only, and a no-op on the blank/zero fields decks
    # actually write.
    "CONSTRAINED_GENERALIZED_WELD_BUTT": Spec(cards=[
        C({3: TIME}),
        C({0: TIME, 2: PRESSURE, 4: LENGTH, 5: LENGTH, 6: LENGTH})]),
    # p.10-38..10-39 (*CONSTRAINED_GLOBAL): TC RC DIR X Y Z TOL - X/Y/Z are a
    # point on the constraint plane and TOL is documented verbatim as a
    # tolerance "in length units".  Modelled with repeat= because real decks
    # stack several triplets under one keyword (the manual carries no
    # repetition note, but a repeat= that never repeats costs nothing while
    # cards=[] would leave lines 2..N silently unscaled).
    "CONSTRAINED_GLOBAL": Spec(repeat=C(
        {3: LENGTH, 4: LENGTH, 5: LENGTH, 6: LENGTH})),
    # p.10-61..10-65 + Figure 10-29 p.10-76 (*CONSTRAINED_JOINT_SCREW):
    # Card 1 is nodes + the RPS/DAMP scale factors (nothing dimensional, but
    # it must still occupy a slot in cards= or LENGTH would land on N1/N4);
    # Card 2 PARM LCID TYPE R1 H_ANGLE - PARM is the helix ratio x_dot/omega,
    # i.e. (L/T)/(rad/T) = LENGTH per radian, and R1 the gear/pulley radius.
    # H_ANGLE is a helix angle in degrees.  Never share this Spec with the
    # other joint types: GEARS/PULLEY read PARM as the dimensionless ratio
    # R2/R1 and RACK_AND_PINION as a pitch (LENGTH).
    "CONSTRAINED_JOINT_SCREW": Spec(cards=[C(), C({0: LENGTH, 3: LENGTH})]),
    # p.10-145..10-151 (*CONSTRAINED_NODAL_RIGID_BODY_SPC): Card 1 PID CID
    # NSID PNODE IPRT DRFLAG RRFLAG (ids/flags); Card 2 CMO CON1 CON2 SPCNID
    # XSPC YSPC ZSPC - CMO is a float-typed flag (0.0, +-1.0, +-2.0), CON1/
    # CON2 are constraint codes or, for CMO < 0, a coordinate-system id and a
    # 6-digit SPC bit pattern, and only XSPC/YSPC/ZSPC ("coordinates where the
    # constraints act") are lengths.  Valid for the exact _SPC spelling only -
    # the options may be combined in any order and each combination shifts the
    # cards, so e.g. _SPC_INERTIA must keep falling through to unknown.
    "CONSTRAINED_NODAL_RIGID_BODY_SPC": Spec(cards=[
        C(), C({4: LENGTH, 5: LENGTH, 6: LENGTH})]),
    # p.10-161..10-163 (*CONSTRAINED_NODE_SET): NSID DOF TF, TF being the
    # failure time at which the nodal constraint goes inactive (default 1E20).
    # cards= rather than repeat=: the _ID option supplies exactly one CNSID
    # per block, and a stacked deck would raise the loud trailing-card warning
    # instead of being silently half-scaled.
    "CONSTRAINED_NODE_SET": Spec(cards=[C({2: TIME})]),

    # ── control / database ──────────────────────────────────────────────────
    "CONTROL_TERMINATION": Spec(cards=[C({0: TIME})]),
    "CONTROL_TIMESTEP": Spec(cards=[C({0: TIME, 3: TIME, 4: TIME})],
                             curves=[(0, 5, TIME, TIME)], extra_ok=True),
    "CONTROL_DYNAMIC_RELAXATION": Spec(cards=[C({3: TIME})]),
    # R16 Vol I p.12-568..12-571 (*CONTROL_THERMAL_SOLVER) Card 1
    # ATYPE PTYPE SOLVER - GPT EQHEAT FWORK SBC.  SBC is the "Stefan
    # Boltzmann constant.  Value is used with enclosure radiation surfaces",
    # the same HEAT_FLUX signature as *BOUNDARY_RADIATION's f - and it MUST
    # be rescaled with it, or a converted radiation deck ends up carrying
    # sigma in two different unit systems at once.  EQHEAT (mechanical
    # equivalent of heat) and FWORK (fraction of work turned into heat) are
    # ratios inside one consistent unit system, and EQHEAT < 0 is a curve id
    # - unscaled either way.  Cards 2a/2b (solver tolerances, branching on
    # SOLVER) and Card 3 are deliberately NOT modelled: they trip the
    # "trailing card(s) beyond the modelled layout" warning instead.
    "CONTROL_THERMAL_SOLVER": Spec(cards=[C({7: HEAT_FLUX})]),
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

    # ── ALE / S-ALE (R16 Vol I) ─────────────────────────────────────────────
    # p.4-80..4-87 (*ALE_REFERENCE_SYSTEM_GROUP): Card 1 SID STYPE PRTYPE
    # PRID BCTRAN BCEXP BCROT ICR/NID is all ids and constraint codes.
    # Card 2 XC YC ZC EXPLIM EFAC - FRCPAD IEXPND: only XC/YC/ZC, the centre
    # of mesh expansion and rotation, are coordinates; EXPLIM is a limit
    # RATIO, EFAC a 0..1 remapping factor and FRCPAD a 0.01..0.2 padding
    # fraction.  Card 3 (optional) IPIDXCL IPIDTYP is a set id and its type
    # flag.  Field 5 of Card 2 is blank in the R16/R17 card table - see
    # x_ale_ref_group for the guard on decks that write there anyway.
    "ALE_REFERENCE_SYSTEM_GROUP": Spec(cards=[
        C(), C({0: LENGTH, 1: LENGTH, 2: LENGTH}), C()]),
    # p.4-102..4-105 (*ALE_STRUCTURED_MESH): Card 1 MSHID DPID NBID EBID - - -
    # TDEATH, where NBID/EBID are the STARTING node and element ids of the
    # generated mesh (pure counters, never scaled) and TDEATH the mesh death
    # time.  Card 2 CPIDX CPIDY CPIDZ NID0 LCSID is all ids - the mesh
    # geometry itself lives in the CONTROL_POINTS cards.
    "ALE_STRUCTURED_MESH": Spec(cards=[C({7: TIME}), C()]),
    "CONTROL_IMPLICIT_GENERAL": Spec(cards=[C({1: TIME})]),
    "CONTROL_IMPLICIT_AUTO": Spec(cards=[C({3: TIME, 4: TIME})]),
    "CONTROL_IMPLICIT_DYNAMICS": Spec(cards=[C({3: TIME, 4: TIME, 5: TIME})]),
    "CONTROL_IMPLICIT_EIGENVALUE": Spec(cards=[C({1: FREQ})], extra_ok=True),
    # R16 Vol I p.15-2..15-4 (*DAMPING_FREQUENCY_RANGE): CDAMP FLOW FHIGH
    # PSID - PIDREL IFLG ICARD2.  CDAMP is a fraction of critical damping,
    # which Remark 1 explicitly contrasts with the mass-weighted
    # (*DAMPING_GLOBAL) and stiffness-proportional forms, so it is a ratio;
    # FLOW/FHIGH are "lowest/highest frequency in the range of interest
    # (cycles per unit time)" and must be rescaled with the time unit.  The
    # optional Card 2 (ICARD2=1 with OPTION1=DEFORM) is CDAMPV IPWP, again
    # dimensionless - hence extra_ok.
    "DAMPING_FREQUENCY_RANGE": Spec(cards=[C({1: FREQ, 2: FREQ})],
                                    extra_ok=True),
    "DAMPING_GLOBAL": Spec(cards=[C({1: FREQ})], curves=[(0, 0, TIME, FREQ)]),
    "DAMPING_PART_MASS": Spec(repeat=C(), curves=[(0, 1, TIME, FREQ)]),
    # R16 Vol I p.16-32: Card1 PSID XCT YCT ZCT XCH YCH ZCH RADIUS;
    # Card2 XHEV YHEV ZHEV LENL LENM NSID ID ITYPE - the edge vector and
    # the in-plane extents are lengths, NSID/ID/ITYPE are ids/flags
    "DATABASE_CROSS_SECTION_PLANE": Spec(group=[
        C({1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH, 5: LENGTH, 6: LENGTH,
           7: LENGTH}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH})]),
    "DATABASE_ALE_OPERATION": Spec(cards=[C(), C({0: TIME}), C()]),

    # ── frequency-domain field-point meshes (R16 Vol I p.23-31..23-36) ─────
    # *FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_{OPTION} generates a surface of
    # field points on which the SPL fringe is plotted.  SPHERE: CENTER R
    # DENSITY X Y Z HALF1 HALF2 - R is the sphere radius and X/Y/Z its
    # centre, while DENSITY is a MESH-refinement integer between 3 and 39
    # ("3 gives 24 elements, 39 gives 8664"), NOT a mass density - scaling it
    # would be a physics-corrupting mistake.  PLATE: NORM LEN_X LEN_Y X Y Z
    # NELM_X NELM_Y - two edge lengths and the plate centre.  CUBE: Card 1d.1
    # LEN_X LEN_Y LEN_Z X Y Z (all six lengths: three edges plus the corner
    # with the smallest nodal coordinates) and Card 1d.2 three element
    # counts.  The PART / PART_SET / NODE_SET options carry a single id and
    # are whitelisted instead.
    "FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_SPHERE": Spec(cards=[
        C({1: LENGTH, 3: LENGTH, 4: LENGTH, 5: LENGTH})]),
    "FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_PLATE": Spec(cards=[
        C({1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH, 5: LENGTH})]),
    "FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_CUBE": Spec(cards=[
        C({i: LENGTH for i in range(6)}), C()]),

    # ── defines ─────────────────────────────────────────────────────────────
    # *DEFINE_BOX and *DEFINE_VECTOR are custom handlers, not Specs: an
    # S-ALE keyword can change what their fields MEAN (see h_define_box /
    # h_define_vector).
    # R16 Vol I p.17-344..17-345 (*DEFINE_PLANE): Card 1 PID X1 Y1 Z1 X2 Y2
    # Z2 CID and Card 2 X3 Y3 Z3 - three genuine non-collinear POINTS, so all
    # nine coordinates are lengths (unlike *DEFINE_COORDINATE_VECTOR, whose
    # triples are directions).
    "DEFINE_PLANE": Spec(cards=[
        C({1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH, 5: LENGTH, 6: LENGTH}),
        C({0: LENGTH, 1: LENGTH, 2: LENGTH})]),
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
    # R16 Vol III p.7-116 (*ICFD_DATABASE_TEMP): repeating PID DTOUT cards.
    # DTOUT is an output TIME INTERVAL throughout the *ICFD_DATABASE family
    # (*ICFD_DATABASE_TPD p.7-118 spells it out: "Time interval to print the
    # output.  If DTOUT is equal to 0.0, then the ICFD time step is used"),
    # not a frequency in Hz.  The generic DATABASE_ fallback in resolve()
    # cannot reach an ICFD_-prefixed name, so it needs its own entry.
    "ICFD_DATABASE_TEMP": Spec(repeat=C({1: TIME})),

    # ── *MESH volume mesher (R16 Vol III) ───────────────────────────────────
    # R16 Vol III p.8-19 (*MESH_SURFACE_NODE) and p.8-9 (*MESH_NODE, which
    # supersedes it): NID X Y Z [TC RC], coordinates are lengths.  Remark 1 of
    # p.8-9 states the card format is identical to *NODE, so both share
    # NODE_W - the trailing TC/RC columns are load-bearing, real mesher output
    # writes them (p.8-18 documents the companion element card as 6i8).
    "MESH_SURFACE_NODE": Spec(repeat=C(
        {1: LENGTH, 2: LENGTH, 3: LENGTH}, (8, 16, 16, 16))),

    # ── airbags (R16 Vol I) ─────────────────────────────────────────────────
    # R16 Vol I p.5-2 (*AIRBAG_REFERENCE_GEOMETRY): repeating node cards
    # NID X Y Z (i8 + 3e16, like *NODE) giving the reference/unstretched airbag
    # geometry - only the coordinates are dimensional (lengths).  The _BIRTH /
    # _RDT / _RDT_ID options keep their own base names and prepend a BIRTH-time
    # or flag card, so they are deliberately not routed here.
    "AIRBAG_REFERENCE_GEOMETRY": Spec(repeat=C(
        {1: LENGTH, 2: LENGTH, 3: LENGTH}, (8, 16, 16, 16))),
    "MESH_NODE": Spec(repeat=C({1: LENGTH, 2: LENGTH, 3: LENGTH}, NODE_W)),
}

# numeric aliases
_MAT_ALIASES = {
    "MAT_001": "MAT_ELASTIC", "MAT_001_FLUID": "MAT_ELASTIC_FLUID",
    "MAT_002": "MAT_ORTHOTROPIC_ELASTIC",
    "MAT_003": "MAT_PLASTIC_KINEMATIC",
    "MAT_008": "MAT_HIGH_EXPLOSIVE_BURN",
    # "MAT_009" -> MAT_NULL registered via register_keyword() (end of file)
    "MAT_002_ANIS": "MAT_ANISOTROPIC_ELASTIC",   # bare MAT_002 is the ORTHO
                                                 # option: different layout
    "MAT_004": "MAT_ELASTIC_PLASTIC_THERMAL",
    "MAT_010": "MAT_ELASTIC_PLASTIC_HYDRO",
    "MAT_015": "MAT_JOHNSON_COOK",
    "MAT_018": "MAT_POWER_LAW_PLASTICITY",
    "MAT_020": "MAT_RIGID", "MAT_022": "MAT_COMPOSITE_DAMAGE",
    "MAT_024": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "MAT_054": "MAT_ENHANCED_COMPOSITE_DAMAGE",
    "MAT_055": "MAT_ENHANCED_COMPOSITE_DAMAGE",
    "MAT_057": "MAT_LOW_DENSITY_FOAM",
    "MAT_063": "MAT_CRUSHABLE_FOAM",
    "MAT_098": "MAT_SIMPLIFIED_JOHNSON_COOK",
    "MAT_100": "MAT_SPOTWELD",
    "MAT_123": "MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY",
    "MAT_181": "MAT_SIMPLIFIED_RUBBER/FOAM",
    # "MAT_140" -> MAT_VACUUM registered via register_keyword() (end of file)
    # newer *DEFINE_TABLE spellings share the base handler
    "DEFINE_TABLE_2D": "DEFINE_TABLE", "DEFINE_TABLE_3D": "DEFINE_TABLE",
    "MAT_027": "MAT_MOONEY-RIVLIN_RUBBER",
    "MAT_031": "MAT_FRAZER_NASH_RUBBER_MODEL",
    "MAT_090": "MAT_ACOUSTIC",
    "MAT_102": "MAT_INV_HYPERBOLIC_SIN",
    "MAT_034": "MAT_FABRIC",
    # R16 Vol II p.2-201: *MAT_SOIL_AND_FOAM_FAILURE "input for this
    # model is the same as *MATERIAL_SOIL_AND_FOAM (Type 5)" - it only
    # adds tensile-failure behaviour, no card of its own.
    "MAT_005": "MAT_SOIL_AND_FOAM", "MAT_014": "MAT_SOIL_AND_FOAM_FAILURE",
    "MAT_SOIL_AND_FOAM_FAILURE": "MAT_SOIL_AND_FOAM",
    "MAT_148": "MAT_GAS_MIXTURE",
    # R16 Vol II p.2-2000: *MAT_ALE_GAS_MIXTURE is "exactly the same as
    # *MAT_GAS_MIXTURE or *MAT_148".  Its _ADV spelling is NOT - it swaps PDV
    # for a species count and repeats a 4-field card - and stays unknown
    # because resolve() only ever strips _TITLE / _ID from a name.
    "MAT_ALE_GAS_MIXTURE": "MAT_GAS_MIXTURE",
    "MAT_159": "MAT_CSCM", "MAT_159_CONCRETE": "MAT_CSCM_CONCRETE",
    "MAT_S01": "MAT_SPRING_ELASTIC", "MAT_S02": "MAT_DAMPER_VISCOUS",
    "MAT_S03": "MAT_SPRING_ELASTOPLASTIC",
    "MAT_S04": "MAT_SPRING_NONLINEAR_ELASTIC",
    "MAT_S05": "MAT_DAMPER_NONLINEAR_VISCOUS",
    "MAT_T01": "MAT_THERMAL_ISOTROPIC",
    "EOS_001": "EOS_LINEAR_POLYNOMIAL", "EOS_002": "EOS_JWL",
    "EOS_004": "EOS_GRUNEISEN",
    "EOS_009": "EOS_TABULATED", "EOS_019": "EOS_MURNAGHAN",
}

# keywords that carry no dimensional data at all
WHITELIST = {
    "KEYWORD", "TITLE", "END", "COMMENT",
    "ELEMENT_SHELL", "ELEMENT_SOLID",
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
    # R16 Vol I p.10-... *CONSTRAINED_TIED_NODES_FAILURE: NSID EPPF ETYPE -
    # EPPF is a plastic strain at failure (dimensionless), the rest ids/flags.
    "CONSTRAINED_TIED_NODES_FAILURE",
    # R16 Vol I p.41-... *SECTION_TSHELL: SECID ELFORM SHRF NIP PROPT QR/IRID
    # ICOMP TSHEAR (+ optional Bi angle card) - a thick-shell section whose
    # thickness comes from the element geometry, so it carries flags/counts/
    # dimensionless factors only.  *DEFINE_COORDINATE_VECTOR: CID XX YX ZX XV
    # YV ZV NID - normalised direction cosines only (no origin, unlike
    # _SYSTEM), treated dimensionless like *DEFINE_SD_ORIENTATION.
    "SECTION_TSHELL", "DEFINE_COORDINATE_VECTOR",
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
    # DATABASE keywords that carry ids/flags only (no DT field)
    "DATABASE_NODAL_FORCE_GROUP", "DATABASE_SPRING_FORWARD",
    "DATABASE_MASSOUT",
    # R16 Vol III p.7-117 (*ICFD_DATABASE_TIMESTEP): one OUTLV on/off flag -
    # despite the name the card carries no timestep value.
    "ICFD_DATABASE_TIMESTEP",
    # R16 Vol III p.8-6 (*MESH_BL_SYM): eight surface part ids, no sizing data
    # (unlike *MESH_BL, which is handled by h_mesh_bl).
    "MESH_BL_SYM",

    # ── structural topology: thick shells, beams, IGA (R16 Vol I) ───────────
    # p.19-5/19-12: Card 1 is EID PID N1 N2 N3 + release codes, Card 8 is the
    # VX/VY/VZ orientation vector.  Per Remark 1 (p.19-14) only the DIRECTION
    # of that vector enters (it locates a virtual third node fixing the r-s
    # plane), so it is invariant under a uniform length rescale.
    "ELEMENT_BEAM_ORIENTATION",
    # p.19-140: EID PID N1..N8, pure connectivity.  The BETA option adds
    # Card 2a whose only field is the orthotropic base offset angle in
    # DEGREES (5th field of a 5 x 16-char card) - also dimensionless.  The
    # COMPOSITE option instead carries ply THICKnesses and stays unknown.
    "ELEMENT_TSHELL", "ELEMENT_TSHELL_BETA",
    # p.19-126..19-131 (*ELEMENT_SOLID_NURBS_PATCH): Card 1 NPID PID NPR PR
    # NPS PS NPT PT and Card 2 WFL NISR NISS NIST IMASS IINT - IDFNE are
    # counts, polynomial orders and flags; Cards 3-5 are the r/s/t knot
    # vectors (parametric knot values on the reference domain, dimensionless);
    # Card 6 lists control-point *NODE ids (the geometry lives in *NODE and is
    # scaled there); Card 7 (WFL != 0) holds NURBS weights, dimensionless.
    # So the whole patch definition carries no (M,L,T) quantity.
    "ELEMENT_SOLID_NURBS_PATCH",
    # p.29-16..29-17 (*INTEGRATION_SHELL): S is a natural through-thickness
    # coordinate in [-1, 1] and WF the weight dt_i/t, both ratios.  (Do NOT
    # generalise to *INTEGRATION_BEAM: its D1-D6 / SREF / TREF are lengths.)
    "INTEGRATION_SHELL",
    # p.41-112..41-114 (*SECTION_TSHELL): SECID ELFORM SHRF NIP PROPT QR
    # ICOMP TSHEAR plus, for ICOMP=1, the B1..B8 material angles in degrees.
    # Unlike *SECTION_SHELL there is no thickness field at all - a thick
    # shell's thickness comes from the element connectivity.
    "SECTION_TSHELL",

    # ── constrained (R16 Vol I) ────────────────────────────────────────────
    # p.10-61..10-64: these joint types read Card 1 only (Card 2 is required
    # for MOTOR/GEARS/RACK_AND_PINION/PULLEY/SCREW).  Card 1 is N1..N6 plus
    # RPS, a RELATIVE penalty-stiffness factor on a stiffness LS-DYNA computes
    # itself (Remark 3, p.10-68), and DAMP, a "damping scale factor on default
    # damping value" - both pure factors.  Exact names only: the _FAILURE and
    # _LOCAL spellings add TIME/FORCE/MOMENT cards and must stay unknown.
    # PLANAR joins them: p.10-62's card summary lists Card 2 as "required for
    # joint types: MOTOR, GEARS, RACK_AND_PINION, PULLEY, and SCREW", which
    # PLANAR is not, and the plane itself comes from the node geometry
    # (Fig. 10-20) rather than from any field.  The _ID option only prepends
    # an id + 70-character heading card, and resolve() strips _ID.
    "CONSTRAINED_JOINT_REVOLUTE", "CONSTRAINED_JOINT_SPHERICAL",
    "CONSTRAINED_JOINT_TRANSLATIONAL", "CONSTRAINED_JOINT_PLANAR",
    # p.10-41..10-46: Card 1 ICID DNID DDOF CIDD ITYP IDNSW FGM is ids, a
    # packed dof digit-string and flags; the repeating Card 2 adds a node id,
    # a dof string and six weighting factors for which "there is no
    # requirement on the values that are chosen ... The default value for the
    # weighting factor is unity" - ratios.  The LOCAL option's Card 3 holds
    # one more coordinate-system ID.  Not one field in the keyword carries a
    # unit, despite the name suggesting interpolated positions.
    "CONSTRAINED_INTERPOLATION", "CONSTRAINED_INTERPOLATION_LOCAL",
    # p.10-181: NID + NSID, a shell node tied to a set of solid nodes.
    "CONSTRAINED_SHELL_TO_SOLID",
    # p.10-224: NSID EPPF ETYPE [PID].  EPPF is a plastic strain, volumetric
    # strain or GISSMO damage at failure - dimensionless in every branch.

    # ── S-ALE (R16 Vol I) ──────────────────────────────────────────────────
    # p.4-117..4-119 (*ALE_STRUCTURED_MESH_REFINE[_REGION]): MSHID plus the
    # integer refinement factors IFX/IFY/IFZ; the REGION card holds nodal
    # INDICES (IMIN..KMAX) into the control-point list, not coordinates.
    "ALE_STRUCTURED_MESH_REFINE", "ALE_STRUCTURED_MESH_REFINE_REGION",

    # ── misc (R16 Vol I) ───────────────────────────────────────────────────
    # p.17-75: CID XX YX ZX XV YV ZV NID - the origin is fixed at (0,0,0) and
    # the axes follow from z = x cross v_xy, so both triples are DIRECTIONS
    # and carry no unit.  (*DEFINE_COORDINATE_SYSTEM gives three real points
    # and is a Spec with LENGTH fields - do not confuse the two.)
    "DEFINE_COORDINATE_VECTOR",
    # p.30-63..30-65: SID CID NID only; the recorded motion goes into the
    # binary interface file, not into the deck.
    "INTERFACE_COMPONENT_SEGMENT",
    # p.23-32 (*FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_{OPTION}): the PART,
    # PART_SET and NODE_SET options take a single id card; the geometric
    # SPHERE / PLATE / CUBE options carry lengths and are Specs.
    "FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_PART",
    "FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_PART_SET",
    "FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_NODE_SET",
}
WHITELIST_PREFIXES = (
    "SET_", "BOUNDARY_SPC", "DATABASE_HISTORY", "CONTROL_MPP_",
    "DEFORMABLE_TO_RIGID", "INTERFACE_SPRINGBACK",
    "DATABASE_EXTENT_",     # output-content flags only
    # R16 Vol I p.7-1..7-4: *CASE, *CASE_BEGIN_n and *CASE_END_n are pure
    # subcase markers.  *CASE's own cards hold a case id, a job-id string,
    # free-text command-line arguments and a list of subcase ids; BEGIN/END
    # have no data-card table at all.  The numbered suffix rules out exact
    # keys, and no other LS-DYNA keyword starts with "CASE".
    "CASE",
    # R16 Vol III p.7-10..7-13: *ICFD_BOUNDARY_FSI and its _EXCLUDE, _FIXED
    # and _ONEWAY sub-options carry a fluid part id (plus, for _ONEWAY, the
    # 1/2 coupling-direction flag IOWC) - nothing dimensional in any of them.
    "ICFD_BOUNDARY_FSI",
)
# ASCII output files whose card 1 is 'DT BINARY LCUR IOOPT' (R16 Vol I
# *DATABASE_OPTION table) - the only DATABASE_ keywords where field 0 is a
# time interval
_DATABASE_ASCII = {
    "ABSTAT", "ABSTAT_CPM", "ATDOUT", "BEARING", "BNDOUT", "CURVOUT",
    "DCFAIL", "DEFGEO", "DEFORC", "DISBOUT", "ELOUT", "GCEOUT", "GLSTAT",
    "GLSTAT_MASS_PROPERTIES", "H3OUT", "JNTFORC", "MATSUM", "MOVIE", "MPGS",
    "NCFORC", "NODFOR", "NODOUT", "PBSTAT", "PLLYOUT", "PRTUBE", "RBDOUT",
    "RCFORC", "RWFORC", "SBTOUT", "SECFORC", "SLEOUT", "SPCFORC",
    "SPHMASSFLOW", "SPHOUT", "SSSTAT", "SSSTAT_MASS_PROPERTIES", "SWFORC",
    "TPRINT", "TRHIST",
}
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
    if lcid is None:
        ctx.error(f"*{block.name}: LCID field is blank or unparseable - "
                  "the curve cannot be identified.")
        return
    if not edit:
        ctx.scan.curve_blocks.setdefault(lcid, []).append((kf, block))
        return
    dims = ctx.scan.curve_dims.get(lcid)
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
            # sort by str(): a TEMP ordinate is the sentinel string "TEMP",
            # which cannot be ordered against a Dim tuple.
            demands = sorted(dims.items(), key=str)
            ctx.error(f"*{block.name} lcid={lcid}: conflicting dimension "
                      f"demands from referencers: {demands} - "
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
    if tbid is None:
        ctx.error(f"*{block.name}: TBID field is blank or unparseable - "
                  "the table cannot be identified.")
        return
    if not edit:
        ctx.scan.table_blocks[tbid] = (kf, block)
        # 'value, lcid' pair form: second numeric field names the sub-curve
        pairs = []
        for li in data[1:]:
            fl = kf.fields(li, CURVE_W, block.long)
            sub = parse_number(fl[1][0]) if len(fl) > 1 else None
            if sub:
                pairs.append(int(sub))
        ctx.scan.table_pairs[tbid] = pairs
        ctx.scan.table_nvalues[tbid] = len(data) - 1
        return
    entry = ctx.scan.table_dims.get(tbid)
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
    """*LOAD_BODY_X/Y/Z/RX/RY/RZ only (R16 Vol I p.33-24): LCID SF LCIDDR
    XC YC ZC CID.  resolve() keeps GENERALIZED/POROUS away from here -
    their layouts differ."""
    axis = block.name.rsplit("_", 1)[-1]
    ydim = ACCEL if axis in ("X", "Y", "Z") else ANG_ACCEL
    kf = ctx.kf
    for li in block.data:
        lcid = _numint(kf, li, STD8, block.long, 0)
        lciddr = _numint(kf, li, STD8, block.long, 2)   # dyn. relax. curve
        if not edit:
            if lcid:
                ctx.scan.register_curve(lcid, TIME, ydim, block.name)
                ctx.scan.probes["gravity_lcids"].append(lcid)
            if lciddr:
                ctx.scan.register_curve(lciddr, TIME, ydim, block.name + " LCIDDR")
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
            ctx.scan.register_curve(lcid, TIME, ydim, block.name)
    if edit:
        ctx.count(block.name + " (curve-carried)")


def h_prescribed_motion(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.7-115..7-124: repeating cards ID/SID DOF VAD LCID SF VID
    DEATH BIRTH; the _ID option prepends one 'ID HEADING' card, which must
    not be parsed as a motion card (its free text would be misread as
    DOF/VAD/LCID and any number in the DEATH/BIRTH columns rescaled)."""
    kf = ctx.kf
    for li in _strip_title(block, list(block.data)):
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
                ctx.scan.register_curve(lcid, xdim, ydim, block.name)
        else:
            kf.scale_field(li, STD8, block.long, 6, ctx.fac(TIME))  # DEATH
            kf.scale_field(li, STD8, block.long, 7, ctx.fac(TIME))  # BIRTH
    if edit:
        ctx.count(block.name)


def h_contact(block: Block, ctx, edit: bool) -> None:
    """*CONTACT_... mandatory Cards 1-3 plus Optional Cards A and B.

    The MORTAR spelling (OPTION1) adds no card - the card ORDER is that of
    the matching non-Mortar contact (R16 Vol I p.11-6..11-7) - but it changes
    two field meanings by value: Card 3 SFSA < 0 makes |SFSA| a load curve of
    contact pressure versus penetration depth (p.11-33), and Optional Card B
    PENMAX is the maximum penetration DISTANCE for Mortar contact rather than
    a fraction of the segment thickness (p.11-102).

    The MPP option (OPTION5) PREPENDS MPP Card 1 and, when the following line
    begins with an ampersand, MPP Card 2 (p.11-6, p.11-19).  Neither carries
    a dimensional field - BCKT is a bucket-sort frequency in CYCLES, PARMAX a
    parametric extension around 1.0, PENSF a per-cycle multiplier and IGTOL a
    scale factor on the segment+node thickness - but the shift they cause
    must be modelled or Card 2's friction/time map would land on MPP Card 1.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    opts = block.name.split("_")
    tiebreak = "TIEBREAK" in block.name
    drawbead = "DRAWBEAD" in block.name
    mortar = "MORTAR" in opts
    plan: List[Dict[int, Dim]] = []
    if "MPP" in opts:
        plan.append({})                                  # MPP Card 1
        if len(data) > 1 and kf.lines[data[1]].lstrip().startswith("&"):
            plan.append({})                              # MPP Card 2
    mpp_off = len(plan)
    plan += [
        {},                                              # card1: ids
        {2: DC_FRIC, 3: PRESSURE, 6: TIME, 7: TIME},     # card2
        {2: LENGTH, 3: LENGTH},                          # card3: SST MST
    ]
    if mortar and len(data) > mpp_off + 2:
        sfsa = kf.get_number(data[mpp_off + 2], STD8, block.long, 0)
        if sfsa is not None and sfsa < 0 and not edit:
            ctx.register_curve(int(-sfsa), LENGTH, PRESSURE,
                               block.name + " Mortar p(penetration)")
    if drawbead and len(data) > mpp_off + 3:
        # R16 Vol I p.11-53: LCIDRF LCIDNF DBDTH DFSCL NUMINT DBPID ELOFF
        # NBEAD.  Both curves give a restraining force "per unit draw bead
        # length as a function of displacement" - a line load (1,0,-2),
        # numerically kunit's STIFF but semantically not a stiffness - and
        # DBDTH is the bead depth that turns contact displacement into that
        # abscissa.  The deck's own header on curve 3 of pan.drawbead.31
        # spells the axes out: "DEPTH   FORC/LGTH".
        db = data[mpp_off + 3]
        lcidrf = _numint(kf, db, STD8, block.long, 0) or 0
        if lcidrf < 0:
            ctx.error(
                f"*{block.name}: LCIDRF={lcidrf} switches the curve to "
                "'maximum bead force as a function of normalized draw bead "
                "length' (R16 Vol I p.11-53), and the manual says 'force' "
                "there where the positive branch says 'force per unit draw "
                "bead length' - the ordinate is ambiguous, so this bead is "
                "not converted.  Convert it manually.")
            return
        if not edit:
            for fi in (0, 1):                            # LCIDRF LCIDNF
                lc = _numint(kf, db, STD8, block.long, fi)
                if lc:
                    ctx.register_curve(lc, LENGTH, STIFF,
                                       block.name + " force/length(depth)")
    if not edit:
        return
    if tiebreak:
        # R16 Vol I p.11-35 Card 4: OPTION NFLS SFLS PARAM ERATEN ERATES
        # CT2CN CN.  ERATEN/ERATES = energy/area, CN = stiffness/length.
        # It comes BEFORE the optional cards A and B.
        plan.append({1: PRESSURE, 2: PRESSURE, 4: STIFF, 5: STIFF,
                     7: STIFF_LEN})
    elif drawbead:
        plan.append({2: LENGTH})                         # Card 4.1 DBDTH
        nbead = _numint(kf, data[mpp_off + 3], STD8, block.long, 7) or 0
        if nbead > 0:
            # Card 4.4 POINT1 POINT2 WIDTH EFFHGT (p.11-54): the bead width
            # and the binder gap below which the bead starts to act.
            plan.append({2: LENGTH, 3: LENGTH})
    # Optional Card A follows Card 4, it does not replace it (card-order
    # table, p.11-6): appending it unconditionally also fixes TIEBREAK decks,
    # where Optional Card A used to be given Optional Card B's field map.
    plan.append({})                                      # A (flags)
    # B: PENMAX SLDTHK SLDSTF.  PENMAX is a fraction of the segment
    # thickness for the a3/a5/a10/13/15/26 types but a maximum penetration
    # DISTANCE for Mortar (and the old 3/5/8/9/10 types), R16 Vol I p.11-102.
    plan.append({6: LENGTH, 7: PRESSURE}
                if not mortar else {0: LENGTH, 6: LENGTH, 7: PRESSURE})
    for ci, li in enumerate(data):
        if ci >= len(plan):
            ctx.warn(f"*{block.name}: optional card {ci + 1} left unscaled "
                     "(advanced options not modelled) - verify manually.")
            break
        if tiebreak and ci == mpp_off + 3:
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
            ctx.scan.register_curve(lcss, STRAIN, PRESSURE, "MAT_024 LCSS")
            ctx.scan.register_table(lcss, RATE, STRAIN, PRESSURE)
        if lcsr:
            ctx.scan.register_curve(lcsr, RATE, DIMLESS, "MAT_024 LCSR")


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


def x_mat_018(block: Block, ctx) -> None:
    """Edit-time SIGY handling for *MAT_POWER_LAW_PLASTICITY (R16 Vol II
    p.2-224): 0 < SIGY < 0.02 is read as the elastic strain to initial
    yield (dimensionless - left unchanged), SIGY >= 0.02 as the initial
    yield stress (scaled here, not by the Spec).  Refuse when scaling a
    stress would drop it below 0.02 and flip LS-DYNA's interpretation."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 2:
        return
    sigy = kf.get_number(data[1], STD8, block.long, 0)
    if not sigy or sigy < 0:
        return
    if sigy < Decimal("0.02"):
        ctx.note(f"*{block.name}: SIGY={sigy} < 0.02 is the elastic strain "
                 "to initial yield (R16 Vol II p.2-224) - left unchanged.")
        return
    f = ctx.fac(PRESSURE)
    if Fraction(sigy) * f < Fraction(2, 100):
        ctx.error(f"*{block.name}: SIGY={sigy} would scale below 0.02, "
                  "which LS-DYNA re-interprets as a strain (R16 Vol II "
                  "p.2-224) - convert this material manually.")
        return
    kf.scale_field(data[1], STD8, block.long, 0, f)


def h_airbag_simple(block: Block, ctx, edit: bool) -> None:
    """*AIRBAG_SIMPLE_AIRBAG_MODEL and *AIRBAG_SIMPLE_PRESSURE_VOLUME,
    R16 Vol I p.3-1..3-16.

    Shared core Card 1 SID SIDTYP RBID VSCA PSCA VINI MWD SPSF: VINI is the
    initial filled VOLUME and MWD the mass-weighted damping factor D of
    F_i = m_i*D*(v_i - v_cg), i.e. a reciprocal time.  SPSF is a 0..1
    stagnation-pressure factor.

    RBID selects how many sensor cards follow, so the card count is
    flag-dependent: RBID = 0 none; RBID > 0 Card 2a (N) plus ceil(N/5) cards
    of user constants; RBID < 0 three cards of activation thresholds
    (accelerations + a duration, velocities, displacements).

    VSCA and PSCA are only dimensionless when the control-volume
    thermodynamics run in the SAME unit system as the FE model
    (V_cvolume = VSCA*V_femodel, P_femodel = PSCA*P_cvolume, p.3-4).  Any
    other value means the gas data are in a foreign system that nothing in
    the deck names, so it is refused.

    SIMPLE_AIRBAG_MODEL then reads Card 3 CV CP T LCID MU AREA PE RO (heat
    capacities, inlet temperature, the mass-flow-rate curve, the exit-hole
    shape factor, the exit AREA, ambient pressure and density; MU < 0 and
    AREA < 0 make those fields curve ids) and Card 4a LOU T_EXT A B MW GASC.
    Card 4a is only read when CV = 0 - otherwise the card degenerates to
    Card 4b, which holds LOU alone and ignores fields 1-5, so they are left
    untouched.  A, B and GASC are per-mole-per-kelvin quantities that reduce
    to an ENERGY once mole and temperature are factored out, which the
    c_p = (a + bT)/MW identity of Remark 3 confirms: energy/mass = specific
    heat.

    SIMPLE_PRESSURE_VOLUME instead reads Card 3 CN BETA LCID LCIDDR with
    Pressure = BETA*CN/RelativeVolume; the relative volume is dimensionless,
    so CN carries the whole pressure dimension (and is a curve id when
    negative) while BETA is a pure multiplier.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    pv = "PRESSURE" in block.name.split("_")
    c1 = data[0]
    for fi, nm in ((3, "VSCA"), (4, "PSCA")):
        v = kf.get_number(c1, STD8, block.long, fi)
        if v is not None and v != 0 and v != 1:
            ctx.error(f"*{block.name}: {nm}={v} means the control-volume gas "
                      "thermodynamics run in a DIFFERENT unit system than "
                      "the FE model (R16 Vol I p.3-4); the deck does not say "
                      "which, so CV/CP/PE/RO/VINI cannot be scaled safely - "
                      "convert this airbag manually.")
            return
    rbid = _numint(kf, c1, STD8, block.long, 2) or 0
    idx = 1
    sensor: List[Tuple[int, Dict[int, Dim]]] = []
    if rbid > 0:
        if idx >= len(data):
            ctx.error(f"*{block.name}: RBID={rbid} > 0 requires the sensor "
                      "Card 2a (R16 Vol I p.3-5) - not found.")
            return
        n = _numint(kf, data[idx], STD8, block.long, 0) or 0
        nconst = -(-max(n, 0) // 5)              # ceil(n / 5)
        if nconst:
            ctx.warn(f"*{block.name}: RBID={rbid} selects a user-defined "
                     f"sensor with {n} constant(s) whose units the manual "
                     "does not document - those cards are left UNSCALED, "
                     "verify them manually.")
        idx += 1 + nconst
    elif rbid < 0:
        sensor = [(idx, {0: ACCEL, 1: ACCEL, 2: ACCEL, 3: ACCEL, 4: TIME}),
                  (idx + 1, {i: VELOCITY for i in range(4)}),
                  (idx + 2, {i: LENGTH for i in range(4)})]
        idx += 3
    if idx >= len(data):
        ctx.error(f"*{block.name}: RBID={rbid} implies the gas card at data "
                  f"line {idx + 1}, but the block has only {len(data)}.")
        return
    c3 = data[idx]

    if not edit:
        if pv:
            cn = kf.get_number(c3, STD8, block.long, 0)
            if cn is not None and cn < 0:
                ctx.register_curve(int(-cn), TIME, PRESSURE,
                                   block.name + " CN(time)")
            lcid = _numint(kf, c3, STD8, block.long, 2)
            if lcid:                       # pressure vs RELATIVE volume
                ctx.register_curve(lcid, DIMLESS, PRESSURE,
                                   block.name + " p(relative volume)")
            lciddr = _numint(kf, c3, STD8, block.long, 3)
            if lciddr:
                ctx.register_curve(lciddr, TIME, PRESSURE,
                                   block.name + " CN(time), relaxation")
            return
        lcid = _numint(kf, c3, STD8, block.long, 3)
        if lcid:                    # mass flow rate in, (1,0,-1) = DAMP
            ctx.register_curve(lcid, TIME, DAMP,
                               block.name + " mass flow rate in(time)")
        mu = kf.get_number(c3, STD8, block.long, 4)
        if mu is not None and mu < 0:
            ctx.register_curve(int(-mu), PRESSURE, DIMLESS,
                               block.name + " shape factor(pressure)")
        area = kf.get_number(c3, STD8, block.long, 5)
        if area is not None and area < 0:
            ctx.register_curve(int(-area), PRESSURE, AREA,
                               block.name + " exit area(pressure)")
        if idx + 1 < len(data):
            lou = _numint(kf, data[idx + 1], STD8, block.long, 0)
            if lou:
                ctx.register_curve(lou, PRESSURE, DAMP,
                                   block.name + " mass flow rate out(p)")
        return

    kf.scale_field(c1, STD8, block.long, 5, ctx.fac(VOLUME))   # VINI
    kf.scale_field(c1, STD8, block.long, 6, ctx.fac(RATE))     # MWD
    for li_idx, dims in sensor:
        for fi, dim in dims.items():
            kf.scale_field(data[li_idx], STD8, block.long, fi, ctx.fac(dim))
    if pv:
        cn = kf.get_number(c3, STD8, block.long, 0)
        if cn is None or cn >= 0:
            kf.scale_field(c3, STD8, block.long, 0, ctx.fac(PRESSURE))
        else:
            ctx.note(f"*{block.name}: CN={cn} is a curve id (R16 Vol I "
                     "p.3-10) - left unchanged.")
        ctx.count(block.name)
        return
    cv = kf.get_number(c3, STD8, block.long, 0)
    for fi in (0, 1):                                          # CV CP
        kf.scale_field(c3, STD8, block.long, fi, ctx.fac(SPEC_HEAT))
    area = kf.get_number(c3, STD8, block.long, 5)
    if area is None or area >= 0:
        kf.scale_field(c3, STD8, block.long, 5, ctx.fac(AREA))
    kf.scale_field(c3, STD8, block.long, 6, ctx.fac(PRESSURE))  # PE
    kf.scale_field(c3, STD8, block.long, 7, ctx.fac(DENSITY))   # RO
    if idx + 1 < len(data):
        if cv:
            ctx.note(f"*{block.name}: CV={cv} != 0, so the fourth card is "
                     "read as Card 4b (LOU only) and its A/B/MW/GASC columns "
                     "are ignored by LS-DYNA (R16 Vol I p.3-12) - left "
                     "unchanged.")
        else:
            for fi in (2, 3, 5):                               # A B GASC
                kf.scale_field(data[idx + 1], STD8, block.long, fi,
                               ctx.fac(MOMENT))
            kf.scale_field(data[idx + 1], STD8, block.long, 4,
                           ctx.fac(MASS))                      # MW
    ctx.count(block.name)


def h_boundary_temperature(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.5-146..5-147 (*BOUNDARY_TEMPERATURE_{NODE|SET}).

    The NODE and SET spellings share one card table: NID TLCID TMULT LOC
    TDEATH TBIRTH, one card per node / node set (the manual prints a single
    "Card 1", but decks stack several - 0168__ex_22 has two).  Only TDEATH
    and TBIRTH are dimensional.

    TMULT is never rescaled whichever branch applies: with TLCID = 0 "T is a
    constant defined by the value TMULT", a temperature; with TLCID > 0 it
    multiplies a (time, temperature) curve and is a plain factor.
    """
    kf = ctx.kf
    for li in block.data:
        lcid = _numint(kf, li, STD8, block.long, 1)
        if not edit:
            if lcid:
                ctx.register_curve(lcid, TIME, TEMP,
                                   block.name + " temperature(time)")
        else:
            kf.scale_field(li, STD8, block.long, 4, ctx.fac(TIME))  # TDEATH
            kf.scale_field(li, STD8, block.long, 5, ctx.fac(TIME))  # TBIRTH
            if not lcid and kf.get_number(li, STD8, block.long, 2):
                ctx.note(f"*{block.name}: temperature field left unchanged "
                         "(temperatures are never rescaled)")
    if edit:
        ctx.count(block.name)


def _thermal_coef_pair(block: Block, ctx, edit: bool, li: int, what: str,
                       page: str) -> None:
    """Shared Card 2 of *BOUNDARY_CONVECTION_SET / *BOUNDARY_RADIATION_SET.

    Layout XLCID XMULT TLCID TMULT LOC, where X is the convection coefficient
    h or the radiation coefficient f = sigma*eps*F.  Both are HEAT_FLUX under
    kunit's same-temperature-unit assumption.  Three branches, and the SIGN
    of XLCID moves the curve's abscissa between time and temperature:

      XLCID = 0   XMULT *is* the coefficient          -> scale XMULT
      XLCID > 0   curve (time, coefficient)           -> XMULT is a factor
      XLCID < 0   curve (temperature, coefficient)    -> XMULT is a factor

    TLCID is the same shape for the environment temperature T_inf, except
    that no negative branch is documented and the ordinate is a TEMP, so
    TMULT never scales either way.

    A positive XLCID may name a *DEFINE_FUNCTION instead of a *DEFINE_CURVE
    (Remark 2) and the two ids live in different tables.  That cannot corrupt
    a deck here because *DEFINE_FUNCTION is a hard flag: any deck carrying
    one is refused before this code registers anything.
    """
    kf = ctx.kf
    xlcid = _numint(kf, li, STD8, block.long, 0) or 0
    tlcid = _numint(kf, li, STD8, block.long, 2) or 0
    if not edit:
        if xlcid > 0:
            ctx.register_curve(xlcid, TIME, HEAT_FLUX,
                               f"{block.name} {what}(time)")
        elif xlcid < 0:
            ctx.register_curve(-xlcid, TEMP, HEAT_FLUX,
                               f"{block.name} {what}(temperature)")
        if tlcid > 0:
            ctx.register_curve(tlcid, TIME, TEMP,
                               block.name + " temperature(time)")
        return
    if xlcid == 0:
        kf.scale_field(li, STD8, block.long, 1, ctx.fac(HEAT_FLUX))
    if not tlcid and kf.get_number(li, STD8, block.long, 3):
        ctx.note(f"*{block.name}: temperature field left unchanged "
                 "(temperatures are never rescaled)")


def _thermal_pairs(block: Block, ctx, edit: bool, what: str, page: str):
    """Walk the repeating 2-card sets of the CONVECTION / RADIATION keywords.

    "Include the following 2 cards for each set" - 0169__ex_23 really does
    stack two pairs under one *BOUNDARY_CONVECTION_SET header, so an odd card
    count means the layout was misread and the deck must be refused rather
    than half-scaled.
    """
    data = list(block.data)
    if len(data) % 2:
        ctx.error(f"*{block.name}: {len(data)} data cards is not a multiple "
                  f"of the 2-card set ({page}) - the layout was not "
                  "understood, refusing to guess.")
        return
    for i in range(1, len(data), 2):
        _thermal_coef_pair(block, ctx, edit, data[i], what, page)
    if edit:
        ctx.count(block.name)


def h_boundary_convection(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.5-30..5-32 (*BOUNDARY_CONVECTION_SET).

    Card 1 is SSID PSEROD (segment-set id + erosion part-set id, no
    dimensional field); Card 2 carries the convection coefficient h - see
    _thermal_coef_pair.  The SEGMENT spelling puts four node ids on Card 1
    instead and is NOT routed here: the exact key keeps them apart.
    """
    _thermal_pairs(block, ctx, edit, "h", "R16 Vol I p.5-30")


def h_boundary_radiation(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.5-120..5-122 (*BOUNDARY_RADIATION_SET).

    Card 1 is SSID TYPE - - - - PSEROD (note the four blank columns: PSEROD
    sits in field 7 here but in field 2 of *BOUNDARY_CONVECTION_SET); Card 2
    carries f = sigma*eps*F - see _thermal_coef_pair.

    Remark 1 (p.5-114) requires an ABSOLUTE temperature scale for radiation
    ("zero degrees must correspond to absolute zero"), which is one more
    reason the T_inf column is left alone: T^4 tolerates no offset shift.
    """
    _thermal_pairs(block, ctx, edit, "f=sigma*eps*F", "R16 Vol I p.5-120")


def h_load_thermal_variable_node(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.33-174 (*LOAD_THERMAL_VARIABLE_NODE): repeating NID TS TB
    LCID, "one card per node", no header card.

    Remark 1 gives T = TB + TS * f(t), so TS and TB are temperatures (never
    rescaled) and the curve is a pure multiplier against TIME - registering
    its ordinate as DIMLESS is what stops a shared curve from being scaled by
    somebody else's factor.  Nothing on the card itself is dimensional.

    Deliberately an exact key: *LOAD_THERMAL_VARIABLE (p.33-168) is a 2-card
    set, and _SHELL / _BEAM (p.33-175, p.33-170) add normalised through-
    thickness coordinates - none of them share this layout.
    """
    kf = ctx.kf
    for li in block.data:
        if not edit:
            lcid = _numint(kf, li, STD8, block.long, 3)
            if lcid:
                ctx.register_curve(lcid, TIME, DIMLESS,
                                   block.name + " multiplier(time)")
        elif any(kf.get_number(li, STD8, block.long, fi) for fi in (1, 2)):
            ctx.note(f"*{block.name}: temperature field left unchanged "
                     "(temperatures are never rescaled)")
    if edit:
        ctx.count(block.name)


def h_load_thermal_load_curve(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.33-160 (*LOAD_THERMAL_LOAD_CURVE): repeating LCID LCIDDR
    cards.  Nothing on the card itself is dimensional, but both curves give a
    uniform nodal temperature as a function of TIME, so their abscissae need
    the time factor - which is why this is not a whitelist entry.  A handler
    rather than spec.curves because spec.curves only ever sees the first data
    line of a repeating card."""
    kf = ctx.kf
    if not edit:
        for li in block.data:
            for fi in (0, 1):                                 # LCID LCIDDR
                lcid = _numint(kf, li, STD8, block.long, fi)
                if lcid:
                    ctx.register_curve(lcid, TIME, TEMP,
                                       block.name + " temperature(time)")
        return
    ctx.count(block.name + " (curve-carried)")


def h_load_thermal_variable(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.33-168..33-169 (*LOAD_THERMAL_VARIABLE): repeating SETS of
    two cards.  Card 1 NSID NSIDEX BOXID holds ids; Card 2 TS TB LCID TSE TBE
    LCIDE LCIDR LCIDEDR gives T = TB + TS*f(t) (Remark 1), so TS/TB and the
    exempted-node pair TSE/TBE are temperatures and the four LCIDs are shape
    curves against TIME.  Whether the curve ordinate is the temperature
    itself (TS = 1.0) or a normalised 0..1 shape is a modelling convention;
    both are numerically identical here because temperatures are never
    rescaled."""
    kf = ctx.kf
    data = list(block.data)
    if len(data) % 2:
        ctx.error(f"*{block.name}: {len(data)} data cards is not a multiple "
                  "of the 2-card set (R16 Vol I p.33-168) - the layout was "
                  "not understood, refusing to guess.")
        return
    for i in range(1, len(data), 2):
        li = data[i]
        if not edit:
            for fi in (2, 5, 6, 7):            # LCID LCIDE LCIDR LCIDEDR
                lcid = _numint(kf, li, STD8, block.long, fi)
                if lcid:
                    ctx.register_curve(lcid, TIME, TEMP,
                                       block.name + " temperature(time)")
        elif any(kf.get_number(li, STD8, block.long, fi)
                 for fi in (0, 1, 3, 4)):      # TS TB TSE TBE
            ctx.note(f"*{block.name}: temperature field left unchanged "
                     "(temperatures are never rescaled)")
    if edit:
        ctx.count(block.name)


# *RIGIDWALL_GEOMETRIC shape card (Card 3a-3d), R16 Vol I p.40-9..40-12.
# FLAT/PRISM give the head of the edge vector l plus the edge lengths;
# SPHERE gives a radius.  CYLINDER is deliberately absent - its NSEGS-driven
# extra cards and the DEFORM sub-cards are not modelled.
_RW_SHAPE_CARD: Dict[str, Dict[int, Dim]] = {
    "FLAT": {0: LENGTH, 1: LENGTH, 2: LENGTH, 3: LENGTH, 4: LENGTH},
    "PRISM": {i: LENGTH for i in range(6)},
    "SPHERE": {0: LENGTH},
}


def h_rigidwall_geometric(block: Block, ctx, edit: bool) -> None:
    """*RIGIDWALL_GEOMETRIC_{SHAPE}_{OPTION}..., R16 Vol I p.40-4..40-15.

    One SET of cards per rigid wall, repeated to the end of the block:
    [ID/title card] Card 1 NSID NSIDEX BOXID BIRTH DEATH; Card 2 XT YT ZT XH
    YH ZH FRIC - the tail and head of the normal are absolute POINTS, so all
    six are lengths while FRIC is a Coulomb coefficient; the shape card
    (3a FLAT, 3b PRISM, 3d SPHERE); then, in this order, Card 4 for the
    MOTION option (LCID OPT VX VY VZ - the direction cosines and the flag are
    dimensionless, the curve is prescribed motion versus time and OPT picks
    its ordinate) and Card 5 for the DISPLAY option (PID RO E PR - the
    display body's density and modulus; the manual types them "I", which is a
    typo given the 1e-9 / 1e-4 defaults).

    The options may be written in any order, so the plan is assembled from
    the option tokens rather than keyed on one spelling.  The CYLINDER shape
    is refused: its Card 3c is followed by NSEGS segment cards and, with
    DEFORM, two more cards including four curve ids the manual gives no
    dimensions for.
    """
    kf = ctx.kf
    opts = block.name.split("_")
    shape = next((s for s in ("FLAT", "PRISM", "CYLINDER", "SPHERE")
                  if s in opts), None)
    if shape is None or shape == "CYLINDER":
        ctx.error(f"*{block.name}: geometric rigid-wall shape "
                  f"{shape or '<none>'} is not modelled (CYLINDER adds "
                  "NSEGS segment cards and, with DEFORM, curve ids with "
                  "undocumented dimensions, R16 Vol I p.40-10..40-12) - "
                  "convert this rigid wall manually.")
        return
    seq: List[Dict[int, Dim]] = []
    if "ID" in opts or "TITLE" in opts:
        seq.append({})                                   # id + A70 heading
    seq.append({3: TIME, 4: TIME})                       # Card 1 BIRTH DEATH
    seq.append({i: LENGTH for i in range(6)})            # Card 2 tail/head
    seq.append(_RW_SHAPE_CARD[shape])                    # Card 3a/3b/3d
    motion = "MOTION" in opts
    if motion:
        seq.append({})                                   # Card 4
    if "DISPLAY" in opts:
        seq.append({1: DENSITY, 2: PRESSURE})            # Card 5
    data = list(block.data)
    if len(data) % len(seq):
        ctx.error(f"*{block.name}: {len(data)} data cards is not a multiple "
                  f"of the expected set size {len(seq)} for this option "
                  "combination - refusing to guess which card is which.")
        return
    for base in range(0, len(data), len(seq)):
        if motion:
            li = data[base + len(seq) - (2 if "DISPLAY" in opts else 1)]
            lcid = _numint(kf, li, STD8, block.long, 0)
            opt = _numint(kf, li, STD8, block.long, 1) or 0
            if lcid and not edit:
                ydim = {0: VELOCITY, 1: LENGTH}.get(opt)
                if ydim is None:
                    ctx.error(f"*{block.name}: MOTION OPT={opt} is not "
                              "documented (0 = velocity, 1 = displacement, "
                              "R16 Vol I p.40-13).")
                    return
                ctx.register_curve(lcid, TIME, ydim, block.name + " motion")
        if not edit:
            continue
        for ci, dims in enumerate(seq):
            for fi, dim in dims.items():
                kf.scale_field(data[base + ci], STD8, block.long, fi,
                               ctx.fac(dim))
    if edit:
        ctx.count(block.name)


def h_mat_frazer_nash(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-296..2-299 (*MAT_FRAZER_NASH_RUBBER_MODEL / MAT_031).

    Card 1 MID RO PR C100 C200 C300 C400; Card 2 C110 C210 C010 C020 EXIT
    EMAX EMIN REF; Card 3 SGL SW ST LCID.

    The eight strain-energy coefficients are strain-energy DENSITIES (the
    invariants of the right Cauchy-Green tensor are dimensionless, so every
    Cijk carries stress units) - but only when they are given directly.  When
    a least-squares fit is requested each of them degenerates to a 0/1
    INCLUSION FLAG: "If a least squares fit is being used, set this constant
    to 1.0 if the term it belongs to in the strain energy functional is to be
    included" (p.2-296..2-297).  Scaling a 1.0 flag by the pressure factor
    would switch terms of U off, so the branch is decided first.

    The manual's trigger for the fit is the whole of Card 3 (SGL, SW, ST and
    LCID); LCID != 0 is used as the proxy here because no fit is possible
    without the force/elongation curve.  RO, SGL, SW and ST are scaled in
    both branches; EMAX/EMIN are Green-St-Venant strain limits.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 3:
        ctx.error(f"*{block.name}: expected 3 cards (MID/RO/PR/C100..C400, "
                  f"C110..C020/EXIT/EMAX/EMIN, SGL/SW/ST/LCID), found "
                  f"{len(data)}.")
        return
    lcid = _numint(kf, data[2], STD8, block.long, 3) or 0
    if not edit:
        if lcid:
            ctx.register_curve(lcid, LENGTH, FORCE,
                               block.name + " force(gauge-length change)")
        return
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(DENSITY))    # RO
    for fi in (0, 1, 2):                                    # SGL SW ST
        kf.scale_field(data[2], STD8, block.long, fi, ctx.fac(LENGTH))
    if lcid:
        ctx.note(f"*{block.name}: LCID={lcid} requests a least-squares fit, "
                 "so C100..C020 are 0/1 term-inclusion flags (R16 Vol II "
                 "p.2-296) - left unchanged.")
    else:
        for fi in (3, 4, 5, 6):                             # C100 C200 C300 C400
            kf.scale_field(data[0], STD8, block.long, fi, ctx.fac(PRESSURE))
        for fi in (0, 1, 2, 3):                             # C110 C210 C010 C020
            kf.scale_field(data[1], STD8, block.long, fi, ctx.fac(PRESSURE))
    if len(data) > 3:
        ctx.warn(f"*{block.name}: {len(data) - 3} trailing card(s) beyond the "
                 "3-card layout left unscaled - verify manually.")
    ctx.count(block.name)


def h_mat_inv_hyperbolic_sin(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-710..2-713 (*MAT_INV_HYPERBOLIC_SIN / MAT_102).

    Card 1a MID RO E PR T HC VP; Card 2a ALPHA N A Q G EPS0 LCQ.

    The flow stress is sigma = (1/ALPHA)*asinh[(Z/A)^(1/N)] with the
    Zener-Hollomon parameter Z = max(eps_dot, EPS0)*exp(Q/(G*T)).  asinh(...)
    is a pure number, so 1/ALPHA is a stress and ALPHA an INVERSE pressure;
    the heat-generation coefficient HC ~ 0.9/(rho*Cv) (Remark 1, p.2-713)
    reduces to the same signature once temperature is factored out.  A is
    documented as 1/sec and EPS0 as a minimum strain rate.

    Q (molar activation energy, J/mol) and G (the gas constant, 8.3144
    J/(mol K) or 40.8825 lb in/(mol degR)) only ever enter as the ratio
    Q/(G*T).  'mol' is not a base dimension kunit models and G additionally
    bundles the temperature unit, so both are left UNCHANGED: leaving both
    alone and scaling both by the same energy factor are equally correct, and
    only a mixed treatment would change the exponent.  The LCQ curve carries
    the same units as Q and is therefore left alone too.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 2:
        ctx.error(f"*{block.name}: expected 2 cards (MID/RO/E/PR/T/HC/VP and "
                  f"ALPHA/N/A/Q/G/EPS0/LCQ), found {len(data)}.")
        return
    lcq = _numint(kf, data[1], STD8, block.long, 6) or 0
    if not edit:
        if lcq:
            # LCQ > 0: Q vs plastic strain; LCQ < 0: Q vs temperature.  Both
            # axes stay unscaled under the Q-unchanged convention.
            ctx.register_curve(abs(lcq), DIMLESS, DIMLESS,
                               block.name + " Q curve (left unscaled)")
        return
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(DENSITY))        # RO
    kf.scale_field(data[0], STD8, block.long, 2, ctx.fac(PRESSURE))       # E
    kf.scale_field(data[0], STD8, block.long, 5, ctx.fac(INV_PRESSURE))   # HC
    kf.scale_field(data[1], STD8, block.long, 0, ctx.fac(INV_PRESSURE))   # ALPHA
    kf.scale_field(data[1], STD8, block.long, 2, ctx.fac(RATE))           # A
    kf.scale_field(data[1], STD8, block.long, 5, ctx.fac(RATE))           # EPS0
    ctx.note(f"*{block.name}: Q (molar activation energy) and G (gas "
             "constant) left UNCHANGED - only the ratio Q/(G*T) enters the "
             "Zener-Hollomon parameter, and both are defined per mole and "
             "per kelvin/rankine, dimensions kunit does not model "
             "(R16 Vol II p.2-713).")
    if len(data) > 2:
        ctx.warn(f"*{block.name}: {len(data) - 2} trailing card(s) beyond the "
                 "2-card layout left unscaled - verify manually.")
    ctx.count(block.name)


def h_mat_spotweld(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-678..2-697 (*MAT_SPOTWELD / MAT_100), base option.

    Card 1 MID RO E PR SIGY EH DT TFAIL; Card 2a EFAIL NRR NRS NRT MRR MSS
    MTT NF.  DT is the mass-scaling time step and TFAIL the failure time;
    EFAIL is a plastic strain and NF a filter count.

    Four groups of fields are sign-overloaded (p.2-681..2-682): SIGY "GT.0:
    Initial yield stress ... LT.0: A yield curve or table is assigned by
    |SIGY|", and each of NRR/NRS/NRT (force resultants at failure) and
    MRR/MSS/MTT (torsional and bending moment resultants) "GT.0: Constant
    value / LT.0: |X| is a load curve ID ... as a function of the effective
    strain rate".  A negative value is a bare id and must survive untouched.
    E < 0 only selects the legacy uniaxial solid-weld formulation while |E|
    remains the modulus, so E is scaled unconditionally.

    A negative SIGY may name a *DEFINE_TABLE instead of a curve; the manual
    does not document that table's value axis, so kunit refuses rather than
    guess it.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 2:
        ctx.error(f"*{block.name}: expected 2 cards (MID/RO/E/PR/SIGY/EH/DT/"
                  f"TFAIL and EFAIL/NRR..MTT/NF), found {len(data)}.")
        return
    resultants = ((1, FORCE, "NRR"), (2, FORCE, "NRS"), (3, FORCE, "NRT"),
                  (4, MOMENT, "MRR"), (5, MOMENT, "MSS"), (6, MOMENT, "MTT"))
    sigy = kf.get_number(data[0], STD8, block.long, 4)
    if not edit:
        if sigy is not None and sigy < 0:
            ctx.error(f"*{block.name}: SIGY={sigy} < 0 names a yield CURVE OR "
                      "TABLE by |SIGY| (R16 Vol II p.2-681); the manual does "
                      "not document the table's value axis, so kunit cannot "
                      "tell the two apart - convert this material manually "
                      f"or resolve it with --curve {int(-sigy)}=<x>:<y>.")
            return
        for fi, dim, name in resultants:
            v = kf.get_number(data[1], STD8, block.long, fi)
            if v is not None and v < 0:
                ctx.register_curve(int(-v), RATE, dim,
                                   f"{block.name} {name}(strain rate)")
        return
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(DENSITY))    # RO
    kf.scale_field(data[0], STD8, block.long, 2, ctx.fac(PRESSURE))   # E
    if sigy is not None and sigy > 0:
        kf.scale_field(data[0], STD8, block.long, 4, ctx.fac(PRESSURE))
    kf.scale_field(data[0], STD8, block.long, 5, ctx.fac(PRESSURE))   # EH
    kf.scale_field(data[0], STD8, block.long, 6, ctx.fac(TIME))       # DT
    kf.scale_field(data[0], STD8, block.long, 7, ctx.fac(TIME))       # TFAIL
    for fi, dim, name in resultants:
        v = kf.get_number(data[1], STD8, block.long, fi)
        if v is None or v <= 0:
            if v is not None and v < 0:
                ctx.note(f"*{block.name}: {name}={v} is a curve id "
                         "(R16 Vol II p.2-682) - left unchanged.")
            continue
        kf.scale_field(data[1], STD8, block.long, fi, ctx.fac(dim))
    if len(data) > 2:
        ctx.warn(f"*{block.name}: {len(data) - 2} trailing card(s) beyond the "
                 "2-card base layout left unscaled - the DAMAGE-FAILURE and "
                 "UNIAXIAL options are separate keywords; verify manually.")
    ctx.count(block.name)


def h_mat_thermal_isotropic(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.3-2..3-3 (*MAT_THERMAL_ISOTROPIC / MAT_T01).

    Card 1 TMID TRO TGRLC TGMULT TLAT HLAT; Card 2 HC TC.  TRO is the thermal
    density, TLAT the phase-change temperature, HC the specific heat and TC
    the thermal conductivity.  HLAT is the latent heat: *MAT_T09 Remark 1
    (p.3-36) defines it as the integral of the SPECIFIC heat over the
    phase-change interval, i.e. energy per mass - the same (0,2,-2) signature
    as HC, different meaning.

    TGMULT is flag-dependent.  With TGRLC = 0 "thermal generation rate is the
    constant multiplier, TGMULT", so the field IS the volumetric heat
    generation rate (W/m3 in SI, Remark 2) - PWR_VOL.  With TGRLC != 0 the
    rate comes from the curve and TGMULT is a plain multiplier on it (the
    worked example on p.3-3 has TGMULT = 1.0 against curve ordinates of
    1.43e7 W/m3), so scaling it would apply the factor twice.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    tgrlc = _numint(kf, data[0], STD8, block.long, 2) or 0
    if not edit:
        if tgrlc > 0:
            ctx.register_curve(tgrlc, TIME, PWR_VOL,
                               block.name + " generation rate(time)")
        elif tgrlc < 0:
            # abscissa is a temperature, which kunit never rescales
            ctx.register_curve(-tgrlc, DIMLESS, PWR_VOL,
                               block.name + " generation rate(temperature)")
        return
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(DENSITY))     # TRO
    if tgrlc == 0:
        kf.scale_field(data[0], STD8, block.long, 3, ctx.fac(PWR_VOL))  # TGMULT
    kf.scale_field(data[0], STD8, block.long, 5, ctx.fac(SPEC_HEAT))   # HLAT
    if kf.get_number(data[0], STD8, block.long, 4):
        ctx.note(f"*{block.name}: temperature field left unchanged "
                 "(temperatures are never rescaled)")
    if len(data) > 1:
        kf.scale_field(data[1], STD8, block.long, 0, ctx.fac(SPEC_HEAT))    # HC
        kf.scale_field(data[1], STD8, block.long, 1, ctx.fac(THERM_COND))   # TC
    if len(data) > 2:
        ctx.warn(f"*{block.name}: {len(data) - 2} trailing card(s) beyond the "
                 "2-card layout left unscaled - the _TD, _TD_LC and "
                 "_PHASE_CHANGE variants are separate keywords; verify.")
    ctx.count(block.name)


def h_mat_fabric(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-312..2-338 (*MAT_FABRIC / MAT_034).

    Card 1 MID RO EA EB - PRBA PRAB, Card 2 GAB - - CSE EL PRL LRATIO DAMP,
    Card 3 AOPT FLC/X2 FAC/X3 ELA LNRC FORM FVOPT TSRFAC, Card 5 - RGBRTH
    A0REF A1 A2 A3 X0 X1, Card 6 V1 V2 V3 - - - BETA ISREFG.  Every "modulus"
    in this material is a STRESS for every FORM value - EA/EB are "Young's
    modulus - longitudinal / transverse direction", GAB the "shear modulus in
    the ab direction", EL the liner's Young's modulus - so there is no
    force-per-unit-width branch to detect.  RGBRTH is the reference-geometry
    birth TIME.  A1-A3 / V1-V3 are directions and BETA an angle in degrees.

    Three things make this keyword refuse rather than guess:

    * FLC and FAC (Card 3 fields 2-3, when X0 is 0, -1 or 1) are defined only
      as "optional porous leakage flow coefficient" / "optional characteristic
      fabric parameter. (See theory manual.)"  Neither volume of R16 states
      their units anywhere, and for FVOPT >= 7 the FAC *curve ordinate* is
      explicitly a "leakage volume flux rate ... equivalent to relative porous
      gas speed", i.e. a VELOCITY - so the scalar cannot simply be assumed
      dimensionless either.  Nonzero values are refused.
      (For 0 < X0 < 1 the same two columns are X2/X3, dimensionless porosity
      coefficients in A_leak = A0(X0 + X1 rs + X2 rp + X3 rs rp) - safe.)
    * FVOPT < 0 adds Card 4 L R C1 C2 C3, whose C1 is only described as
      "pressure coefficient (dependent on unit system)".  Remark 16 makes
      (C1 dp^C2 - C3) dimensionless, so C1 carries pressure^(-C2) - a
      data-dependent dimension that also shifts every later card.  Refused.
    * FORM = -14 turns LCA/LCB into *DEFINE_TABLEs and adds a coating Card 8.
      Refused; FORM = 4, 14 and 24 (plain stress-strain curves on Card 7) are
      handled.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 5:
        ctx.error(f"*{block.name}: {len(data)} data cards, but the keyword "
                  "always has at least 5 (R16 Vol II p.2-312) - refusing.")
        return
    form = _numint(kf, data[2], STD8, block.long, 5) or 0
    fvopt = _numint(kf, data[2], STD8, block.long, 6) or 0
    # Card 4 only exists for FVOPT < 0 (refused below), so the five cards
    # present are 1, 2, 3, 5, 6 - the manual's "Card 5" is data[3].
    x0 = kf.get_number(data[3], STD8, block.long, 6) or 0
    if fvopt < 0:
        ctx.error(f"*{block.name}: FVOPT={fvopt} adds the St. Venant-Wantzel "
                  "Card 4, whose C1 is only 'dependent on unit system' "
                  "(R16 Vol II p.2-327) - it carries pressure^(-C2) and "
                  "shifts every later card. Convert this fabric manually.")
        return
    if form == -14:
        ctx.error(f"*{block.name}: FORM=-14 makes LCA/LCB *DEFINE_TABLE ids "
                  "and adds the coating Card 8 (R16 Vol II p.2-330..2-332) - "
                  "layout not modelled, convert manually.")
        return
    if not (0 < x0 < 1):
        for fi, nm in ((1, "FLC"), (2, "FAC")):
            if kf.get_number(data[2], STD8, block.long, fi):
                ctx.error(
                    f"*{block.name}: {nm} is nonzero, and R16 defines it only "
                    f"as an 'optional {'porous leakage flow' if fi == 1 else 'characteristic fabric'}"
                    f" parameter. (See theory manual.)' (Vol II p.2-320) - "
                    "its units are stated nowhere, so it cannot be rescaled "
                    "or safely left alone. Convert this fabric manually.")
                return
    if not edit:
        ela = kf.get_number(data[2], STD8, block.long, 3)
        tsr = kf.get_number(data[2], STD8, block.long, 7)
        if ela and ela < 0:              # effective leakage area vs time
            ctx.register_curve(int(-ela), TIME, DIMLESS, block.name + " ELA")
        if tsr and (tsr < 0 or tsr >= 1):   # strain-restoration factor vs time
            ctx.register_curve(int(abs(tsr)), TIME, DIMLESS,
                               block.name + " TSRFAC")
        if len(data) > 5:                # Card 7: stress as a function of strain
            for fi in range(6):
                lc = _numint(kf, data[5], STD8, block.long, fi)
                if lc:
                    ctx.register_curve(abs(lc), DIMLESS, PRESSURE,
                                       block.name + " Card 7")
        return
    for fi in (1, 2, 3):                                       # RO EA EB
        kf.scale_field(data[0], STD8, block.long, fi,
                       ctx.fac(DENSITY if fi == 1 else PRESSURE))
    kf.scale_field(data[1], STD8, block.long, 0, ctx.fac(PRESSURE))   # GAB
    kf.scale_field(data[1], STD8, block.long, 4, ctx.fac(PRESSURE))   # EL
    kf.scale_field(data[3], STD8, block.long, 1, ctx.fac(TIME))       # RGBRTH
    if len(data) > 6:
        ctx.warn(f"*{block.name}: {len(data) - 6} trailing card(s) beyond "
                 "Card 7 left unscaled - verify.")
    ctx.count(block.name)


def h_mat_enhanced_composite_damage(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-419..2-436 (*MAT_ENHANCED_COMPOSITE_DAMAGE, MAT_054/055).

    Six required cards, then up to three optional ones when CRIT = 54:
      1 MID RO EA EB EC PRBA PRCA PRCB
      2 GAB GBC GCA (KF) AOPT 2WAY TI
      3 XP YP ZP A1 A2 A3 MANGLE          - XP/YP/ZP are a point, so LENGTHs
      4 V1 V2 V3 D1 D2 D3 [DFAILM DFAILS] - directions and failure STRAINS
      5 TFAIL ALPH SOFT FBRT YCFAC DFAILT DFAILC EFS
      6 XC XT YC YT SC CRIT BETA          - strengths, i.e. stresses
      7 PFL EPSF EPSR TSMD SOFT2          - strains and factors
      8 SLIMT1..SLIMS NCYRED SOFTG        - factors
      9 LCXC LCXT LCYC LCYT LCSC DT       - strength as a function of strain rate

    ALPH is the same nonlinear-shear term as *MAT_022's ("see *MAT_022",
    p.2-424), which p.2-249 gives "in units of [stress^-3]" - a field that is
    easy to mistake for the 0..1 weight BETA on Card 6 and silently drop.

    TFAIL is the trap.  Its meaning switches on the VALUE, at 0.1:
    "GT.0.0.and.LE.0.1: element is deleted when its time step is smaller than
    the given value" but "GT.0.1: ... the quotient of the actual time step and
    the original time step".  So it is a TIME in the first band and a ratio in
    the second, and a conversion that pushes a time across 0.1 would silently
    turn it into a ratio - that case is refused rather than written out.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 6:
        ctx.error(f"*{block.name}: {len(data)} data cards, but Cards 1-6 are "
                  "always required (R16 Vol II p.2-420) - refusing.")
        return
    if not edit:
        if len(data) > 8:                       # Card 9 (rate-dependent)
            for fi in range(5):
                lc = _numint(kf, data[8], STD8, block.long, fi)
                if lc:
                    ctx.register_curve(lc, RATE, PRESSURE,
                                       block.name + " strength(strain rate)")
        return
    for fi in (1, 2, 3, 4):                                    # RO EA EB EC
        kf.scale_field(data[0], STD8, block.long, fi,
                       ctx.fac(DENSITY if fi == 1 else PRESSURE))
    for fi in range(4):                                        # GAB GBC GCA KF
        kf.scale_field(data[1], STD8, block.long, fi, ctx.fac(PRESSURE))
    for fi in range(3):                                        # XP YP ZP
        kf.scale_field(data[2], STD8, block.long, fi, ctx.fac(LENGTH))
    kf.scale_field(data[4], STD8, block.long, 1, ctx.fac(STRESS_M3))   # ALPH
    tfail = kf.get_number(data[4], STD8, block.long, 0)
    if tfail is not None and 0 < tfail <= Decimal("0.1"):
        ft = ctx.fac(TIME)
        if Fraction(tfail) * ft > Fraction(1, 10):
            ctx.error(
                f"*{block.name}: TFAIL={tfail} is a time-step threshold "
                "(R16 Vol II p.2-424: 'GT.0.0.and.LE.0.1'), but rescaling it "
                f"by {ft} puts it above 0.1, where the same field means a "
                "time-step RATIO instead. Set TFAIL manually in the target "
                "units.")
            return
        kf.scale_field(data[4], STD8, block.long, 0, ft)
    for fi in range(5):                                        # XC XT YC YT SC
        kf.scale_field(data[5], STD8, block.long, fi, ctx.fac(PRESSURE))
    if len(data) > 8:
        dt = kf.get_number(data[8], STD8, block.long, 5)
        if dt and dt > 0:                # strain-rate averaging window
            kf.scale_field(data[8], STD8, block.long, 5, ctx.fac(TIME))
    if len(data) > 9:
        ctx.warn(f"*{block.name}: {len(data) - 9} trailing card(s) beyond "
                 "Card 9 left unscaled - verify.")
    ctx.count(block.name)


def h_mat_simplified_rubber_foam(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-1210..2-1218 (*MAT_SIMPLIFIED_RUBBER/FOAM, MAT_181).
    The keyword name really does carry a slash; the parser keeps it because
    it splits keyword lines on whitespace, not on punctuation.

    Card 1 MID RO KM MU G SIGF REF PRTEN, Card 2 SGL SW ST LC/TBID TENSION
    RTYPE AVGOPT PR, optional Card 4 LCUNLD HU SHAPE STOL VISCO HISOUT and
    optional Card 5 (up to 12) Gi BETAi VFLAG.

    The curve is the interesting part.  The manual defines it once and
    unconditionally: LC gives "the force as a function of the actual change
    in the gauge length", and only *if* SGL, SW and ST are all 1.0 is it
    *also* readable as engineering stress against engineering strain.  So the
    honest axes are (LENGTH, FORCE), and scaling SGL/SW/ST as the lengths
    they are keeps that self-consistent: LS-DYNA forms eps = dl/SGL (both
    lengths, unchanged) and sigma = F/(SW*ST) (a stress, correctly scaled).
    A deck written with the unity-specimen idiom therefore comes out with
    SGL = SW = ST = 25.4 after in -> mm, which looks odd but is right.

    Two sign-switched fields: AVGOPT < 0 is "a time window/interval over
    which the strain rates are averaged", and PR < 0 is not a Poisson ratio
    at all but the viscosity coefficient beta of p^{n+1} = p^n exp(-beta dt)
    ..., i.e. a RATE.

    Only the bare and _TITLE spellings route here.  _WITH_FAILURE inserts its
    own Card 3 and _LOG_LOG_INTERPOLATION is a separate entry, so both stay
    unknown rather than being silently given this layout.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 2:
        ctx.error(f"*{block.name}: {len(data)} data cards, but Cards 1-2 are "
                  "always required (R16 Vol II p.2-1210) - refusing.")
        return
    lc = _numint(kf, data[1], STD8, block.long, 3)
    if not edit:
        if lc:
            ctx.register_curve(lc, LENGTH, FORCE,
                               block.name + " force(gauge length change)")
            # LC/TBID: the same field names a *DEFINE_TABLE whose own values
            # are strain rates; registering both is harmless because only the
            # matching keyword ever consumes the registration.
            ctx.register_table(lc, RATE, LENGTH, FORCE)
        if len(data) > 2:
            lcu = _numint(kf, data[2], STD8, block.long, 0)
            if lcu:
                ctx.register_curve(lcu, LENGTH, FORCE,
                                   block.name + " unloading force(length)")
        return
    for fi, dim in ((1, DENSITY), (2, PRESSURE), (4, PRESSURE), (5, PRESSURE)):
        kf.scale_field(data[0], STD8, block.long, fi, ctx.fac(dim))
    for fi in range(3):                                        # SGL SW ST
        kf.scale_field(data[1], STD8, block.long, fi, ctx.fac(LENGTH))
    avgopt = kf.get_number(data[1], STD8, block.long, 6)
    if avgopt and avgopt < 0:
        kf.scale_field(data[1], STD8, block.long, 6, ctx.fac(TIME))
    pr = kf.get_number(data[1], STD8, block.long, 7)
    if pr and pr < 0:
        kf.scale_field(data[1], STD8, block.long, 7, ctx.fac(RATE))
    for li in data[3:]:                                   # Prony series cards
        vflag = _numint(kf, li, STD8, block.long, 2) or 0
        if not vflag:      # VFLAG=1 -> normalised moduli, dimensionless
            kf.scale_field(li, STD8, block.long, 0, ctx.fac(PRESSURE))  # Gi
        kf.scale_field(li, STD8, block.long, 1, ctx.fac(RATE))          # BETAi
    ctx.count(block.name)


def h_mat_gas_mixture(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-1018..2-1024 (*MAT_GAS_MIXTURE / MAT_148).

    Card 1 MID IADIAB RUNIV PDV.  RUNIV switches the whole rest of the
    keyword between a per-MASS and a per-MOLE description of the gas species,
    and the two use different dimensions in the same columns:

      RUNIV = 0   Card 2 CVMASS1-8, Card 3 CPMASS1-8   - specific heats
      RUNIV != 0  Card 2 MOLWT1-8   (mass per mole)
                  Card 3 CPMOLE1-8, Card 4 B1-8, Card 5 C1-8
                  - molar heat capacity and its first/second-order
                    temperature coefficients

    The mole is an invariant amount of substance and, per kunit's usual
    convention, so is the temperature unit, so a per-mole-per-kelvin heat
    capacity scales as a plain ENERGY and RUNIV (J/(mol*K)) with it.  That
    keeps Remark 2's Cp(T) = [CPMOLE + B*T + C*T^2]/MOLWT consistent:
    ENERGY / MASS = (0,2,-2) = SPEC_HEAT, matching the RUNIV = 0 branch.

    (Do NOT calibrate against the worked example on p.2-1024 - it prints SI
    values of RUNIV and MOLWT next to a ton-mm-s density and is internally
    inconsistent by a factor of 1000.)
    """
    if not edit:
        return
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    runiv = kf.get_number(data[0], STD8, block.long, 2)
    kf.scale_field(data[0], STD8, block.long, 2, ctx.fac(MOMENT))     # RUNIV
    per_card = ([SPEC_HEAT, SPEC_HEAT] if not runiv
                else [MASS, MOMENT, MOMENT, MOMENT])
    if len(data) - 1 > len(per_card):
        ctx.warn(f"*{block.name}: {len(data) - 1 - len(per_card)} trailing "
                 f"card(s) beyond the {len(per_card)} species cards this "
                 f"RUNIV branch defines were left unscaled - verify.")
    for dim, li in zip(per_card, data[1:]):
        for fi in range(8):
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(dim))
    ctx.count(block.name)


def h_mat_057(block: Block, ctx) -> None:
    """Scan-time table registration for *MAT_LOW_DENSITY_FOAM (MAT_057).

    LCID (Card 1 field 3) may name a *DEFINE_TABLE instead of a curve; the
    table's value axis is then a family of discrete TEMPERATURES and every
    sub-curve is still nominal stress versus nominal strain (R16 Vol II
    p.2-437).  spec.curves can only call register_curve, so the table form is
    registered here with a dimensionless value axis.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    lcid = _numint(kf, data[0], STD8, block.long, 3)
    if lcid:
        ctx.register_table(lcid, DIMLESS, DIMLESS, PRESSURE)


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
            ctx.scan.sec_discrete_dro[secid] = dro
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
    return mid in ctx.scan.torsional_mats


def h_smat_spring_elastic(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    if not edit:
        ctx.scan.smat_blocks.append((kf, block, "S01"))
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
        ctx.scan.smat_blocks.append((kf, block, "S03"))
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
        ctx.scan.smat_blocks.append((kf, block, "S02"))
        return
    dim = ROT_DAMP if _smat_torsional(kf, block, ctx) else DAMP
    kf.scale_field(data[0], STD8, block.long, 1, ctx.fac(dim))
    ctx.count(block.name)


def h_smat_curve_mats(block: Block, ctx, edit: bool) -> None:
    """S04 (MID LCD LCR) / S05 (MID LCDR): curves carry the physics; their
    dims depend on translational vs torsional, resolved post-scan."""
    if not edit:
        kind = "S05" if "DAMPER" in block.name else "S04"
        ctx.scan.smat_blocks.append((ctx.kf, block, kind))
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


def h_define_vector(block: Block, ctx, edit: bool) -> None:
    """*DEFINE_VECTOR: VID XT YT ZT XH YH ZH CID, repeating.

    Normally the tail and head are two POINTS, so all six values are lengths.
    But a vector named by the VID field of
    *ALE_STRUCTURED_MESH_VOLUME_FILLING is not geometry at all: "Fields 2 to
    5 (XT, YT, ZT) of the *DEFINE_VECTOR card are used to define the initial
    translational velocities" (R16 Vol I p.4-125).  Scaling those as lengths
    is wrong by exactly the time-unit ratio - 1000x converting s to ms - and
    is silent, which is why the S-ALE handler records such ids at scan time
    and this handler consults the set.  A vector in that role that also
    carries a non-zero HEAD is genuinely ambiguous and is refused.
    """
    if not edit:
        return
    kf = ctx.kf
    for li in block.data:
        vid = _numint(kf, li, STD8, block.long, 0)
        if vid in ctx.sale_vel_vectors:
            if any(kf.get_number(li, STD8, block.long, fi)
                   for fi in (4, 5, 6)):
                ctx.error(f"*{block.name}: vector {vid} is referenced as the "
                          "VID of *ALE_STRUCTURED_MESH_VOLUME_FILLING, where "
                          "XT/YT/ZT are initial VELOCITIES (R16 Vol I "
                          "p.4-125), but it also carries a non-zero head "
                          "XH/YH/ZH - the same vector cannot be both a "
                          "velocity and a geometric direction; split it.")
                return
            for fi in (1, 2, 3):
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(VELOCITY))
            ctx.note(f"*{block.name}: vector {vid} feeds an S-ALE volume "
                     "filling, so XT/YT/ZT were scaled as VELOCITIES, not "
                     "lengths (R16 Vol I p.4-125).")
            continue
        for fi in range(1, 7):
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(LENGTH))
    ctx.count(block.name)


def h_define_box(block: Block, ctx, edit: bool) -> None:
    """*DEFINE_BOX: BOXID XMN XMX YMN YMX ZMN ZMX, normally real coordinates.

    A box referenced by an *ALE_STRUCTURED_MESH_VOLUME_FILLING card with
    GEOM = BOXCPT holds S-ALE CONTROL-POINT INDICES instead ("the box is
    defined using S-ALE control points"; the manual's own example on
    R16 Vol I p.4-129 is '*DEFINE_BOX 1 8 15 8 15 8 15', where 8 and 15 are
    indices).  Scaling those by the length factor would move the box to
    nonsense indices, so such ids are exempted.  GEOM = BOXCOR boxes are
    true coordinates and keep the normal scaling.
    """
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    if not data:
        return
    boxid = _numint(kf, data[0], STD8, block.long, 0)
    if boxid in ctx.sale_index_boxes:
        ctx.note(f"*{block.name}: box {boxid} is referenced with "
                 "GEOM=BOXCPT, so its six values are S-ALE control-point "
                 "INDICES, not coordinates (R16 Vol I p.4-126) - left "
                 "unchanged.")
    else:
        for fi in range(1, 7):
            kf.scale_field(data[0], STD8, block.long, fi, ctx.fac(LENGTH))
    if len(data) > 1:
        ctx.warn(f"*{block.name}: {len(data) - 1} trailing card(s) beyond the "
                 "single box card left unscaled - verify manually.")
    ctx.count(block.name)


def x_ale_ref_group(block: Block, ctx) -> None:
    """Guard for the one undocumented column of *ALE_REFERENCE_SYSTEM_GROUP.

    Card 2 field 5 is blank in the R16 and R17 card tables, but the manual's
    own Example 2 header (p.4-87) labels it SMOOTHVMX - a name that reads
    like a maximum smoothing VELOCITY.  Every documented example leaves it
    empty; a deck that writes there gets a loud warning rather than a silent
    pass-through."""
    kf = ctx.kf
    data = list(block.data)
    if len(data) > 1 and kf.get_number(data[1], STD8, block.long, 5):
        ctx.warn(f"*{block.name}: Card 2 field 6 is blank in the R16/R17 "
                 "card table but this deck writes a value there (the "
                 "manual's Example 2 header calls it SMOOTHVMX, which reads "
                 "like a velocity) - left UNSCALED, verify manually.")


def h_ale_structured_fsi(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.4-96..4-101 (*ALE_STRUCTURED_FSI[_ABEXT|_ABINT]).

    Card 1 LSTRSID ALESID LSTRSTYP ALESTYP - - - MCOUP is all ids and flags.
    Card 2 START END PFAC FRIC ILVL FLIP DECAY OFFSET: only START and END are
    times.  PFAC > 0 is a fraction of the estimated critical stiffness,
    FRIC > 0 a Coulomb coefficient, DECAY a factor that scales the FSI
    pressure at the gas front and OFFSET > 0 a fraction of the Lagrange
    segment thickness - all dimensionless.  Each of the three, when negative,
    names a curve (or, for FRIC, a *DEFINE_TABLE) that carries the physics
    instead, so the ids must survive untouched and the referenced data gets
    the dimensions registered here.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 2:
        if edit:
            ctx.count(block.name)
        return
    c2 = data[1]
    if not edit:
        pfac = kf.get_number(c2, STD8, block.long, 2)
        if pfac is not None and pfac < 0:
            ctx.register_curve(int(-pfac), LENGTH, PRESSURE,
                               block.name + " PFAC p(penetration)")
        fric = kf.get_number(c2, STD8, block.long, 3)
        if fric is not None and fric < 0:
            # table values are coupling pressures, each sub-curve is the
            # friction coefficient versus relative velocity
            ctx.register_table(int(-fric), PRESSURE, VELOCITY, DIMLESS)
        offset = kf.get_number(c2, STD8, block.long, 7)
        if offset is not None and offset < 0:
            ctx.register_curve(int(-offset), TIME, LENGTH,
                               block.name + " OFFSET(time)")
        return
    for fi in (0, 1):                                        # START END
        kf.scale_field(c2, STD8, block.long, fi, ctx.fac(TIME))
    if len(data) > 2:
        ctx.warn(f"*{block.name}: {len(data) - 2} trailing card(s) beyond "
                 "Card 2 left unscaled - verify manually.")
    ctx.count(block.name)


def h_ale_control_points(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.4-106..4-112 (*ALE_STRUCTURED_MESH_CONTROL_POINTS).

    Card 1 is a normal 8x10 card: CPID - ICASE SFO - OFFO.  SFO and OFFO
    apply as "Ordinate value = SFO x (Defined value + OFFO)" (Remark 1,
    p.4-108), so the length factor belongs on X and OFFO while SFO, a pure
    scale factor, must stay - scaling SFO instead would apply the factor
    twice.

    The point cards that follow use THREE 20-character fields, not the 8x10
    layout: N (the control-point index, never scaled), X (a position) and a
    third value whose dimension flips with ICASE - a dimensionless
    progressive-spacing RATIO for ICASE = 0 (dl(n+1) = dl(n)*(1+RATIO)) but
    the element LENGTH XL for ICASE = 1 or 2.  Blank fields are common
    (ICASE = 2 leaves X blank on all but the base node) and stay blank.
    """
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    if not data:
        return
    icase = _numint(kf, data[0], STD8, block.long, 2) or 0
    if icase not in (0, 1, 2):
        ctx.error(f"*{block.name}: ICASE={icase} is not documented "
                  "(0, 1 or 2, R16 Vol I p.4-107) - refusing to guess "
                  "whether the third column is a ratio or a length.")
        return
    kf.scale_field(data[0], STD8, block.long, 5, ctx.fac(LENGTH))   # OFFO
    for li in data[1:]:
        # In the documented 20-char layout the right-aligned integer N ends
        # in columns 11-20, so columns 1-10 are blank; a 10-char deck puts N
        # in columns 1-10 instead.
        widths = (20, 20, 20) if not kf.lines[li][:10].strip() else (10, 10, 10)
        kf.scale_field(li, widths, block.long, 1, ctx.fac(LENGTH))   # X
        if icase:
            kf.scale_field(li, widths, block.long, 2, ctx.fac(LENGTH))  # XL
    ctx.count(block.name)


# *ALE_STRUCTURED_MESH_VOLUME_FILLING Card 2 by GEOM (R16 Vol I p.4-126,
# R17 Vol I p.4-127).  Card 2 is GEOM IN/OUT E1 E2 E3 E4 E5 at 0-based
# 0..6; only the E fields listed here are dimensional.
_VF_GEOM: Dict[str, Dict[int, Dim]] = {
    "ALL": {},
    "PART": {3: LENGTH},        # E1 part id, E2 offset along the normals
    "PARTSET": {3: LENGTH},
    "SEGSET": {3: LENGTH},
    "PLANE": {},                # E1/E2 are node ids
    "CYLINDER": {4: LENGTH, 5: LENGTH},   # E1/E2 node ids, E3/E4 radii
    "BOXCOR": {},               # E1 is a *DEFINE_BOX of real coordinates
    "BOXCPT": {},               # E1 is a *DEFINE_BOX of control-point indices
    "ELLIPSOID": {3: LENGTH, 4: LENGTH, 5: LENGTH},  # rx ry rz
}


def h_ale_volume_filling(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.4-124..4-129 (*ALE_STRUCTURED_MESH_VOLUME_FILLING).

    Two cards per filling instruction, several instructions in sequence.
    Card 1 MSHID - AMMGTO - NSAMPLE - - VID: ids, a sampling count and an
    alphanumeric AMMG NAME (never numeric-parsed), plus VID - a
    *DEFINE_VECTOR whose XT/YT/ZT are INITIAL VELOCITIES here, recorded so
    h_define_vector scales them correctly.
    Card 2 GEOM IN/OUT E1..E5: E1-E5 are typed "I or F" and their meaning -
    hence their dimension - is selected by the alphanumeric GEOM keyword in
    field 1 of the same card, which is why no static table can express it.
    GEOM = BOXCPT additionally means the referenced *DEFINE_BOX holds
    control-point INDICES, recorded so h_define_box leaves it alone.

    SPHERE appears in the R16 prose list but has no row in the geometry
    table and is gone in R17, so it is refused rather than guessed.
    """
    kf = ctx.kf
    data = list(block.data)
    if len(data) % 2:
        ctx.error(f"*{block.name}: {len(data)} data cards is not a multiple "
                  "of the 2-card filling instruction (R16 Vol I p.4-124) - "
                  "the layout was not understood, refusing to guess.")
        return
    for i in range(0, len(data), 2):
        c1, c2 = data[i], data[i + 1]
        geom = kf.fields(c2, STD8, block.long)[0][0].strip().upper()
        dims = _VF_GEOM.get(geom)
        if dims is None:
            ctx.error(f"*{block.name}: GEOM={geom!r} is not in the R16/R17 "
                      "geometry table (ALL, PART, PARTSET, SEGSET, PLANE, "
                      "CYLINDER, BOXCOR, BOXCPT, ELLIPSOID; SPHERE is listed "
                      "in the R16 prose but has no card row and is gone in "
                      "R17) - refusing to guess the E1..E5 dimensions.")
            return
        if not edit:
            vid = _numint(kf, c1, STD8, block.long, 7)
            if vid:
                ctx.sale_vel_vectors.add(vid)
            if geom == "BOXCPT":
                boxid = _numint(kf, c2, STD8, block.long, 2)
                if boxid:
                    ctx.sale_index_boxes.add(boxid)
            continue
        for fi, dim in dims.items():
            kf.scale_field(c2, STD8, block.long, fi, ctx.fac(dim))
    if edit:
        ctx.count(block.name)


# *INITIAL_VOLUME_FRACTION_GEOMETRY Card 3 by CNTTYP (R16 Vol I
# p.28-140..28-144).  Note XOFFST moves from field 4 (CNTTYP=1) to field 3
# (CNTTYP=2), and the plane's direction cosines must never be scaled.
_IVFG_CNTTYP: Dict[int, Dict[int, Dim]] = {
    1: {3: LENGTH},                                  # SID STYPE NORMDIR XOFFST
    2: {2: LENGTH},                                  # SGSID NORMDIR XOFFST
    3: {0: LENGTH, 1: LENGTH, 2: LENGTH},            # X0 Y0 Z0 + cosines
    4: {i: LENGTH for i in range(8)},                # cone: 2 centres + 2 radii
    5: {i: LENGTH for i in range(6)},                # box: min/max corners
    6: {i: LENGTH for i in range(4)},                # sphere: centre + radius
}


def h_initial_volume_fraction_geometry(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.28-136..28-146 (*INITIAL_VOLUME_FRACTION_GEOMETRY).

    Card 1 FMSID FMIDTYP BAMMG NTRACE is ids and a sampling count (BAMMG may
    be an AMMG NAME string under S-ALE, so it is never numeric-parsed).  It
    is followed by PAIRS of cards until the next keyword: a container card
    CNTTYP FILLOPT FAMMG VX VY VZ - whose VX/VY/VZ are the easily-missed
    initial VELOCITY of that AMMG - and a geometry card whose whole layout is
    selected by CNTTYP (1 part/part-set surface, 2 segment set, 3 plane,
    4 cone, 5 box, 6 sphere, 7 *DEFINE_FUNCTION).

    CNTTYP = 7 is refused for the same reason *DEFINE_FUNCTION is a hard
    flag: a free-form expression of (x, y, z) cannot be auto-scaled.  An odd
    number of trailing cards means the pairing assumption is wrong (Remark 1,
    p.28-144, requires a Card 3 for every Card 2) and is refused too.
    """
    if not edit:
        return
    kf = ctx.kf
    data = list(block.data)
    if not data:
        return
    body = data[1:]
    if len(body) % 2:
        ctx.error(f"*{block.name}: {len(body)} cards after Card 1 is not a "
                  "multiple of the (container, geometry) pair required by "
                  "Remark 1 (R16 Vol I p.28-144) - refusing to guess.")
        return
    for i in range(0, len(body), 2):
        cont, geom = body[i], body[i + 1]
        cnttyp = _numint(kf, cont, STD8, block.long, 0) or 0
        if cnttyp == 7:
            ctx.error(f"*{block.name}: CNTTYP=7 fills the container from a "
                      "*DEFINE_FUNCTION (R16 Vol I p.28-144), whose "
                      "free-form expression cannot be auto-scaled - convert "
                      "this filling manually.")
            return
        dims = _IVFG_CNTTYP.get(cnttyp)
        if dims is None:
            ctx.error(f"*{block.name}: CNTTYP={cnttyp} is not documented "
                      "(1-7, R16 Vol I p.28-138) - refusing to guess.")
            return
        for fi in (3, 4, 5):                                  # VX VY VZ
            kf.scale_field(cont, STD8, block.long, fi, ctx.fac(VELOCITY))
        for fi, dim in dims.items():
            kf.scale_field(geom, STD8, block.long, fi, ctx.fac(dim))
    ctx.count(block.name)


def h_boundary_sale_mesh_face(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.5-126..5-128 (*BOUNDARY_SALE_MESH_FACE).

    Repeating cards BCTYPE MSHID NEGX POSX NEGY POSY NEGZ POSZ.  BCTYPE is
    alphanumeric (FIXED, NOFLOW, SYM, NONREFL, FLOWVEL, PRES, AMBIENT) and
    the six face fields are 0/1 flags for the flag-like BCTYPEs - but for
    PRES and FLOWVEL a value greater than zero IS the id of the
    *DEFINE_CURVE controlling that face ("faces loaded by time-dependent
    pressures" / "nodes constrained by time-dependent velocities"), so the
    curve dimensions have to be registered or the referenced curve is
    emitted unchanged with only an 'unreferenced' warning.  Nothing on the
    card is ever scaled.

    For AMBIENT the face value is a PACKED integer, MMG + 100*LCID1 +
    10000*LCID2; unpacking it reliably is not documented well enough to risk,
    so a packed value carrying curve ids is refused instead.
    """
    if edit:
        ctx.count(block.name)
        return
    kf = ctx.kf
    for li in block.data:
        bctype = kf.fields(li, STD8, block.long)[0][0].strip().upper()
        ydim = {"PRES": PRESSURE, "FLOWVEL": VELOCITY}.get(bctype)
        for fi in range(2, 8):
            v = _numint(kf, li, STD8, block.long, fi)
            if not v or v <= 0:
                continue
            if ydim is not None:
                ctx.register_curve(v, TIME, ydim,
                                   f"{block.name} {bctype} face")
            elif bctype == "AMBIENT" and v >= 100:
                ctx.error(f"*{block.name}: BCTYPE=AMBIENT face value {v} "
                          "packs load-curve ids as MMG + 100*LCID1 + "
                          "10000*LCID2 (R16 Vol I p.5-127); kunit does not "
                          "unpack it, so the referenced curves would be left "
                          f"unscaled - resolve them with --curve, or convert "
                          "this boundary manually.")
                return


def h_lagrange_in_solid(block: Block, ctx, edit: bool) -> None:
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) > 1:
        pfac = kf.get_number(data[1], STD8, block.long, 2)
        if pfac is not None and pfac < 0:
            lcid = int(-pfac)
            if not edit:
                ctx.scan.register_curve(lcid, LENGTH, PRESSURE, "CLIS PFAC curve")
    if edit:
        if len(data) > 1:
            kf.scale_field(data[1], STD8, block.long, 0, ctx.fac(TIME))
            kf.scale_field(data[1], STD8, block.long, 1, ctx.fac(TIME))
        ctx.count("CONSTRAINED_LAGRANGE_IN_SOLID")


def h_joint_stiffness_generalized(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.10-88..10-100 (*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED).

    Exactly four cards.  Card 1 JSID PIDA PIDB CIDA CIDB JID RPS - ids plus
    the relative penalty-stiffness factor, nothing dimensional.  Card 2b.1
    LCIDPH LCIDT LCIDPS DLCIDPH DLCIDT DLCIDPS - moment-versus-rotation and
    damping-moment-versus-rotation-rate curve ids (abscissae in RADIANS and
    radians per unit time, so DIMLESS and ANG_VEL; ordinates MOMENT).
    Card 2b.2 ESPH FMPH EST FMT ESPS FMPS - the ES* elastic stiffnesses are
    "per unit radian", i.e. MOMENT, and are scaled unconditionally; the FM*
    frictional moment limits are typed "F/I" and sign-overloaded: "LT.0:
    -FMxx is the load curve or table ID defining the yield moment as a
    function of rotation" (p.10-98), so a negative value is an id that must
    stay byte-identical.  Card 2b.3 holds six stop angles in degrees.

    The table form of a negative FM* requires JID on Card 1 ("a table permits
    the moment to also be a function of the joint reaction force and requires
    the specification of JID"), and kunit cannot resolve a table id through
    register_curve, so FM* < 0 with a non-zero JID is refused rather than
    guessed.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 4:
        ctx.error(f"*{block.name}: expected 4 cards (ids, curve ids, "
                  f"stiffness/friction, stop angles), found {len(data)}.")
        return
    jid = _numint(kf, data[0], STD8, block.long, 5) or 0
    if not edit:
        for fi in (0, 1, 2):                       # LCIDPH LCIDT LCIDPS
            lcid = _numint(kf, data[1], STD8, block.long, fi)
            if lcid:
                ctx.register_curve(lcid, DIMLESS, MOMENT,
                                   block.name + " moment(rotation)")
        for fi in (3, 4, 5):                       # DLCIDPH DLCIDT DLCIDPS
            lcid = _numint(kf, data[1], STD8, block.long, fi)
            if lcid:
                ctx.register_curve(lcid, ANG_VEL, MOMENT,
                                   block.name + " damping moment(rate)")
    for fi in (1, 3, 5):                           # FMPH FMT FMPS
        v = kf.get_number(data[2], STD8, block.long, fi)
        if v is None or v >= 0:
            continue
        if jid:
            ctx.error(f"*{block.name}: FM field {fi + 1} = {v} is a negative "
                      f"curve OR table id and JID={jid} is set, so the "
                      "manual's table form is enabled (R16 Vol I p.10-98) - "
                      "kunit cannot resolve a *DEFINE_TABLE reference here; "
                      "convert this joint stiffness manually.")
            return
        if not edit:
            ctx.register_curve(int(-v), DIMLESS, MOMENT,
                               block.name + " yield moment(rotation)")
    if not edit:
        return
    for fi in (0, 2, 4):                           # ESPH EST ESPS
        kf.scale_field(data[2], STD8, block.long, fi, ctx.fac(MOMENT))
    for fi in (1, 3, 5):                           # FMPH FMT FMPS
        v = kf.get_number(data[2], STD8, block.long, fi)
        if v is not None and v < 0:
            ctx.note(f"*{block.name}: FM field {fi + 1} = {v} is a curve id "
                     "(R16 Vol I p.10-98) - left unchanged, the referenced "
                     "curve is scaled instead.")
            continue
        kf.scale_field(data[2], STD8, block.long, fi, ctx.fac(MOMENT))
    if len(data) > 4:
        ctx.warn(f"*{block.name}: {len(data) - 4} trailing card(s) beyond the "
                 "4-card GENERALIZED layout left unscaled - verify manually.")
    ctx.count(block.name)


def h_shell_in_solid(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.10-178..10-180 (*CONSTRAINED_SHELL_IN_SOLID[_PENALTY]).

    Card 1 SHSID SSID SHSTYP SSTYP holds ids only.  The optional Card 2 is
    START END - - - PSSF: normally two coupling times, with PSSF a penalty
    spring stiffness SCALE factor (the stiffness itself is derived from the
    geometric mean of the shell and solid bulk moduli, so PSSF is a pure
    ratio).  But END = -9999 is a sentinel: "If END = -9999, START is
    interpreted as the curve or table ID defining multiple pairs of
    start-time and end-time" (p.10-180).  Scaling that pair would both
    destroy the curve reference and turn -9999 into a value LS-DYNA reads
    under the *other* branch ("negative END indicates that coupling is
    inactive during dynamic relaxation"), so both fields are left alone and
    the referenced curve - whose two axes are BOTH times - is scaled instead.
    """
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if len(data) < 2:                    # Card 2 is optional and often absent
        if edit:
            ctx.count(block.name)
        return
    end = kf.get_number(data[1], STD8, block.long, 1)
    if end is not None and end == -9999:
        start = _numint(kf, data[1], STD8, block.long, 0)
        if not edit:
            if start:
                ctx.register_curve(start, TIME, TIME,
                                   block.name + " start/end time pairs")
        else:
            ctx.note(f"*{block.name}: END=-9999 makes START={start} a curve "
                     "id holding start/end time pairs (R16 Vol I p.10-180) - "
                     "both fields left unchanged, the curve is scaled.")
    elif edit:
        for fi in (0, 1):                                    # START END
            kf.scale_field(data[1], STD8, block.long, fi, ctx.fac(TIME))
    if edit:
        if len(data) > 2:
            ctx.warn(f"*{block.name}: {len(data) - 2} trailing card(s) beyond "
                     "Card 2 left unscaled - verify manually.")
        ctx.count(block.name)


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
                ctx.scan.register_curve(lcid, TIME, ydim, block.name)
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
                ctx.scan.register_curve(lcid, TIME, PRESSURE, block.name)
        else:
            kf.scale_field(li, STD8, block.long, 3, ctx.fac(TIME))  # DEATH
            kf.scale_field(li, STD8, block.long, 4, ctx.fac(TIME))  # BIRTH
    if edit:
        ctx.count(block.name)


def h_icfd_prescribed_temp(block: Block, ctx, edit: bool) -> None:
    """R16 Vol III p.7-28 (*ICFD_BOUNDARY_PRESCRIBED_TEMP), repeating cards
    PID LCID SF DEATH BIRTH.  The LCID curve is temperature versus time, so
    only its abscissa is dimensional; DEATH/BIRTH are times and SF carries
    the temperature amplitude against a normalised curve.

    Registering the ordinate as TEMP is what makes a SHARED curve safe: real
    ICFD decks point one unit-amplitude curve at both a prescribed velocity
    and a prescribed temperature, and without this registration the curve
    would silently be scaled by the velocity factor alone - multiplying the
    imposed fluid temperature by that factor.  With both demands registered
    h_define_curve refuses the deck instead (or, if every ordinate is zero,
    downgrades it to a warning)."""
    kf = ctx.kf
    for li in block.data:
        lcid = _numint(kf, li, STD8, block.long, 1)
        if not edit:
            if lcid:
                ctx.register_curve(lcid, TIME, TEMP, block.name)
        else:
            kf.scale_field(li, STD8, block.long, 3, ctx.fac(TIME))  # DEATH
            kf.scale_field(li, STD8, block.long, 4, ctx.fac(TIME))  # BIRTH
    if edit:
        ctx.count(block.name)


def h_icfd_initial(block: Block, ctx, edit: bool) -> None:
    """R16 Vol III p.7-141 (*ICFD_INITIAL), repeating cards PID Vx Vy Vz T P
    - DFUNC.  DFUNC reinterprets the SAME columns on the SAME card: with
    DFUNC = 1 "all previous flags for initial velocity, pressure and
    temperature now refer to *DEFINE_FUNCTION IDs", i.e. fields 1-5 become
    bare integer ids that must not be rescaled.  Such a deck also needs
    *DEFINE_FUNCTION, which is already a hard flag, but the refusal is made
    explicit here rather than left to depend on that."""
    if not edit:
        return
    kf = ctx.kf
    for li in block.data:
        dfunc = _numint(kf, li, STD8, block.long, 7) or 0
        if dfunc:
            ctx.error(f"*{block.name}: DFUNC={dfunc} makes the velocity, "
                      "temperature and pressure columns *DEFINE_FUNCTION ids "
                      "(R16 Vol III p.7-141) - they must not be rescaled and "
                      "the functions themselves cannot be auto-converted; "
                      "convert this initial condition manually.")
            return
        for fi in (1, 2, 3):                                  # Vx Vy Vz
            kf.scale_field(li, STD8, block.long, fi, ctx.fac(VELOCITY))
        kf.scale_field(li, STD8, block.long, 5, ctx.fac(PRESSURE))     # P
        if kf.get_number(li, STD8, block.long, 4):                     # T
            ctx.note(f"*{block.name}: temperature field left unchanged "
                     "(temperatures are never rescaled)")
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
            ctx.scan.register_curve(lcid, TIME, DIMLESS, block.name + " LCIDSF")
        for fi in tfields:
            v = kf.get_number(li, STD8, block.long, fi)
            if v is not None and v < 0 and fi in neg_curve:
                if not edit:
                    ctx.scan.register_curve(int(-v), TIME, TIME,
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
                    ctx.scan.register_curve(int(-v), TIME, dim,
                                       f"{block.name} (negative field "
                                       f"{fi + 1})")
                continue
            if edit and dim is not DIMLESS:
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(dim))
    if edit:
        ctx.count(block.name)


def h_blast(block: Block, ctx, edit: bool) -> None:
    """*LOAD_BLAST_ENHANCED (R16 Vol I p.33-16..33-21) and legacy *LOAD_BLAST.

    Card 2 CFM/CFL/CFT/CFP (p.33-18) convert model units to ConWep's
    lbm/ft/ms/psi when UNIT=5.
    """
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
                    s, rel = format_fixed(Decimal(str(v)), w)
                    kf.max_fmt_err = max(kf.max_fmt_err, rel)
                    kf.set_field(li2, STD8, block.long, fi, s)
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


def h_load_gravity_part(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.33-57..33-58 (*LOAD_GRAVITY_PART[_SET]): repeating cards
    PID/PSID DOF LC ACCEL LCDR STGA STGR.  ACCEL is the gravity acceleration;
    LC and LCDR are load curves of a dimensionless FACTOR versus time that
    multiplies ACCEL (Remark 1: "Curve LC gives factor as a function of
    time"), so only ACCEL rescales.  STGA/STGR are construction-stage ids.
    ACCEL values also feed unit detection (they sit near 9.80665 m/s^2)."""
    kf = ctx.kf
    for li in block.data:
        if not edit:
            for fi in (2, 4):                             # LC, LCDR
                lcid = _numint(kf, li, STD8, block.long, fi)
                if lcid:
                    ctx.scan.register_curve(lcid, TIME, DIMLESS,
                                       block.name + " factor curve")
            v = kf.get_number(li, STD8, block.long, 3)
            if v:
                ctx.scan.probes["gravity_accels"].append(abs(float(v)))
        else:
            kf.scale_field(li, STD8, block.long, 3, ctx.fac(ACCEL))
    if edit:
        ctx.count(block.name)


def _exact_pow10(f: Fraction) -> Optional[int]:
    """k when f == 10**k exactly, else None."""
    num, den = f.numerator, f.denominator
    if den != 1:
        if num != 1:
            return None
        k = _exact_pow10(Fraction(den))
        return -k if k is not None else None
    k = 0
    while num % 10 == 0:
        num //= 10
        k += 1
    return k if num == 1 else None


def _scale_sn_fields(block: Block, ctx, li: int, lcid: int,
                     fis: Tuple[int, int, int], what: str) -> None:
    """Rescale one A/B/STHRES record of a predefined S-N equation.

    R16 Vol II p.2-94..2-96 and Vol I p.23-73..23-79 (identical equations):
      LCID=-1:  N*S^b = a          ->  [a] = stress^b,  b dimensionless
      LCID=-2:  log(S) = a - b*log(N) -> a shifts by log10(stress factor),
                                         b dimensionless (log-log slope)
      LCID=-3:  S = a*N^b          ->  [a] = stress,    b dimensionless
      LCID=-4:  S = a - b*log(N)   ->  [a] = [b] = stress
    with S the stress amplitude/range and N the cycle count.  STHRES is a
    threshold stress in every form.  SNTYPE (range vs amplitude) and LTYPE
    (interpolation) never change the dimensions.
    """
    kf = ctx.kf
    a_fi, b_fi, s_fi = fis
    fs = ctx.fac(PRESSURE)
    kf.scale_field(li, STD8, block.long, s_fi, fs)        # STHRES
    if lcid == -3:
        kf.scale_field(li, STD8, block.long, a_fi, fs)
    elif lcid == -4:
        kf.scale_field(li, STD8, block.long, a_fi, fs)
        kf.scale_field(li, STD8, block.long, b_fi, fs)
    elif lcid == -1:
        b = kf.get_number(li, STD8, block.long, b_fi)
        if b is None or b == int(b):
            kf.scale_field(li, STD8, block.long, a_fi, fs ** int(b or 0))
        else:
            ctx.error(f"*{block.name}: {what} uses LCID=-1 (N*S^b = a) with "
                      f"non-integer B={b}; A scales by (stress factor)^B, "
                      "which cannot be applied exactly - convert this S-N "
                      "definition manually.")
    elif lcid == -2:
        k = _exact_pow10(fs)
        if k is None:
            ctx.error(f"*{block.name}: {what} uses LCID=-2 (log(S) = a - "
                      "b*log(N)); A shifts by log10(stress factor), which is "
                      f"not an integer for this unit pair (factor {fs}) - "
                      "convert this S-N definition manually.")
            return
        v = kf.get_number(li, STD8, block.long, a_fi)
        if v is not None and k:
            w = 20 if block.long else 10
            s, rel = format_fixed(v + k, w)
            kf.max_fmt_err = max(kf.max_fmt_err, rel)
            kf.set_field(li, STD8, block.long, a_fi, s)
    else:
        ctx.error(f"*{block.name}: {what} has LCID={lcid}, which is not a "
                  "documented S-N equation id (-1..-4) - refusing to guess.")


def h_mat_add_fatigue(block: Block, ctx, edit: bool) -> None:
    """R16 Vol II p.2-92..2-97 (*MAT_ADD_FATIGUE[_EN]).

    <BLANK> option, LCID > 0 (Card 1a MID LCID LTYPE - - - SNLIMT SNTYPE):
    the S-N data lives in *DEFINE_CURVE LCID with abscissa N (cycles to
    failure, dimensionless) and ordinate S (stress) per Remark 1 (p.2-96).
    LCID < 0 (Card 1b MID LCID LTYPE A B STHRES SNLIMT SNTYPE, plus up to 7
    segment cards "- - - Ai Bi STHRESi", p.2-95) selects a predefined
    equation handled by _scale_sn_fields.
    EN option (Card 1c MID KP NP SIGMAF EPSP BP CP): KP (cyclic strength
    coefficient) and SIGMAF (fatigue strength coefficient) are stresses; NP,
    BP, CP are exponents and EPSP a strain (dimensionless).  Optional Card 2b
    E PR carries a modulus."""
    kf = ctx.kf
    data = _strip_title(block, list(block.data))
    if not data:
        return
    if "EN" in block.name.split("_"):
        if edit:
            for fi in (1, 3):                             # KP, SIGMAF
                kf.scale_field(data[0], STD8, block.long, fi,
                               ctx.fac(PRESSURE))
            if len(data) > 1:                             # Card 2b: E PR
                kf.scale_field(data[1], STD8, block.long, 0,
                               ctx.fac(PRESSURE))
            ctx.count(block.name)
        return
    lcid = _numint(kf, data[0], STD8, block.long, 1)
    if lcid is None:
        lcid = -1                                         # manual default
    if lcid > 0:
        if not edit:
            ctx.scan.register_curve(lcid, DIMLESS, PRESSURE,
                               block.name + " S-N (cycles vs stress)")
            return
        if len(data) > 1:
            ctx.warn(f"*{block.name}: LCID={lcid} > 0 but {len(data) - 1} "
                     "extra card(s) present - segment cards are only read "
                     "for LCID < 0 (R16 Vol II p.2-92); left unchanged, "
                     "verify.")
        ctx.count(block.name)
        return
    if not edit:
        return
    for i, li in enumerate(data):
        _scale_sn_fields(block, ctx, li, lcid, (3, 4, 5),
                         "Card 1b" if i == 0 else f"segment card {i}")
    ctx.count(block.name)


# *FREQUENCY_DOMAIN_RANDOM_VIBRATION base-quantity by VAFLAG (R16 Vol I
# p.23-66); the PSD ordinate is (base quantity)^2 per (cycles/time).
_RV_BASE_DIM: Dict[int, Optional[Dim]] = {
    0: None,                        # no random vibration analysis
    1: ACCEL, 11: ACCEL,            # base / enforced acceleration
    2: PRESSURE, 3: PRESSURE,       # random pressure / plane wave
    8: FORCE,                       # nodal force
    9: VELOCITY, 12: VELOCITY,      # base / enforced velocity
    10: LENGTH, 13: LENGTH,         # base / enforced displacement
}
_RV_PSD_DIM: Dict[Dim, Dim] = {
    ACCEL: ACCEL_PSD, VELOCITY: VEL_PSD, LENGTH: DISP_PSD,
    PRESSURE: PRES_PSD, FORCE: FORCE_PSD,
}


def h_freq_random_vibration(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_RANDOM_VIBRATION[_FATIGUE], R16 Vol I p.23-63..79.

    R16 layout: Card1 MDMIN MDMAX FNMIN FNMAX RESTRT (FNMIN/FNMAX are natural
    frequencies); Card2 DAMPF LCDAM LCTYP DMPMAS DMPSTF DMPTYP (DAMPF is the
    modal damping ratio; DMPMAS/DMPSTF are the Rayleigh alpha [1/time] and
    beta [time]; LCDAM's abscissa is a frequency for LCTYP=0 or a mode number
    for LCTYP=1, ordinate dimensionless); Card3 VAFLAG METHOD UNIT UMLT VAPSD
    VARMS NAPSD NCPSD; Card4 LDTYP IPANELU IPANELV TEMPER - LDFLAG ICOARSE
    TCOARSE; NAPSD Card5s SID STYPE DOF LDPSD LDVEL LDFLW LDSPN CID; NCPSD
    Card6s LOAD_I LOAD_J LCTYP2 LDPSD1 LDPSD2.  FATIGUE adds Card7 MFTG NFTG
    STRTYP TEXPOS STRSF INFTG SRANGE (TEXPOS = exposure time; STRSF/SRANGE
    are dimensionless factors), NFTG Card7.1s (7.1a PID LCID PTYPE LTYPE - -
    - SNLIMT for LCID>0, 7.1b PID LCID PTYPE - A B STHRES SNLIMT for LCID<0,
    equations as in _scale_sn_fields) and INFTG Card7.2 filename cards.

    Legacy (LS-DYNA R8..R10, written by LS-PrePost <= 4.5 and identified by
    its $# header comments) instead holds MFTG/RESTRM/INFTG on Card1 fields
    6-8, NFTG on Card3 field 8 and TEXPOS/SNTYPE/STRSF on Card4 fields 5/7/8;
    those Card1/Card4 fields are structurally blank in R16 and NCPSD cannot
    be negative, so any number there identifies the legacy layout.  Only
    NFTG=-999 (S-N data from *MAT_ADD_FATIGUE) is modelled for it: the
    legacy in-block S-N card layout is not documented in the R16 manual.

    All frequencies are model-unit cycles/time (the companion output card
    *DATABASE_FREQUENCY_BINARY documents FMIN/FMAX as "cycles/time", R16
    Vol I p.16-98, and Remark 7 p.23-76 defines acceleration as
    [length]/[time]^2 for UNIT=0).  UNIT != 0 declares g-based PSD input;
    the manual does not define what frequency unit those g^2/Hz ordinates
    are per when the model time unit is not the second, so it is refused.
    VAFLAG 4-7 (shock/progressive/reverberant/TBL waves) add an optional
    PREF card and phase-velocity/decay curves that are not modelled; LDTYP=1
    (SPL in dB) references an SI-flavoured default PREF - both refused."""
    kf = ctx.kf
    data = list(block.data)
    fatigue = "FATIGUE" in block.name.split("_")
    if len(data) < 4:
        ctx.error(f"*{block.name}: expected at least Cards 1-4, found "
                  f"{len(data)} data card(s).")
        return
    c1, c2, c3, c4 = data[:4]

    def num(li, fi):
        return kf.get_number(li, STD8, block.long, fi)

    def iv(li, fi):
        return _numint(kf, li, STD8, block.long, fi) or 0

    vaflag = iv(c3, 0)
    unit = iv(c3, 2)
    napsd_raw = _numint(kf, c3, STD8, block.long, 6)
    napsd = 1 if napsd_raw is None else max(napsd_raw, 0)  # default 1
    f7 = _numint(kf, c3, STD8, block.long, 7)              # NCPSD | legacy NFTG
    ldtyp = iv(c4, 0)

    if vaflag in (4, 5, 6, 7):
        ctx.error(f"*{block.name}: VAFLAG={vaflag} (shock/progressive/"
                  "reverberant/TBL wave loading) is not modelled - it adds "
                  "an optional PREF card (changing the card layout) and "
                  "phase-velocity / decay curves (LDVEL/LDFLW/LDSPN) - "
                  "convert manually.")
        return
    if vaflag not in _RV_BASE_DIM:
        ctx.error(f"*{block.name}: VAFLAG={vaflag} is not documented "
                  "(R16 Vol I p.23-66) - refusing to guess.")
        return
    if unit != 0:
        ctx.error(f"*{block.name}: UNIT={unit} declares g-based acceleration "
                  "input (R16 Vol I p.23-66..67); the manual does not define "
                  "which frequency unit the g^2-per-frequency PSD ordinates "
                  "are per when the model time unit is not the second, so "
                  "the PSD curve cannot be rescaled safely - use model "
                  "units (UNIT=0) or convert manually.")
        return
    if ldtyp == 1:
        ctx.error(f"*{block.name}: LDTYP=1 (SPL input in dB) depends on a "
                  "reference pressure with an SI-flavoured default "
                  "(2.0E-5, Card 3.1 p.23-67) - not modelled, convert "
                  "manually.")
        return
    if ldtyp not in (0, 2):
        ctx.error(f"*{block.name}: LDTYP={ldtyp} is not documented "
                  "(R16 Vol I p.23-68) - refusing to guess.")
        return

    legacy = fatigue and (any(num(c1, fi) for fi in (5, 6, 7))
                          or (f7 or 0) < 0 or bool(num(c4, 4)))
    base = _RV_BASE_DIM[vaflag]
    psd = _RV_PSD_DIM.get(base)

    # ── card-count plan ──────────────────────────────────────────────────
    load_cards = data[4:4 + napsd]
    if legacy:
        nftg = f7 or 0
        if nftg != -999:
            ctx.error(f"*{block.name}: legacy (R8/R10) layout with "
                      f"NFTG={nftg}: only NFTG=-999 (S-N data from "
                      "*MAT_ADD_FATIGUE) is modelled - the legacy in-block "
                      "S-N card layout is not documented in the R16 manual; "
                      "port the deck to the R16 layout or convert manually.")
            return
        if iv(c1, 7):
            ctx.error(f"*{block.name}: legacy (R8/R10) layout with "
                      f"INFTG={iv(c1, 7)} appends initial-damage database "
                      "cards whose legacy layout cannot be verified - "
                      "convert manually.")
            return
        expected = 4 + napsd
        cross_cards, c7, sn_cards = [], None, []
    else:
        ncpsd = f7 or 0
        if ncpsd < 0:
            ctx.error(f"*{block.name}: NCPSD={ncpsd} is invalid (cross-PSD "
                      "count cannot be negative).")
            return
        cross_cards = data[4 + napsd:4 + napsd + ncpsd]
        expected = 4 + napsd + ncpsd
        c7, sn_cards = None, []
        if fatigue:
            if len(data) <= expected:
                ctx.error(f"*{block.name}: FATIGUE option requires Card 7 "
                          "after the load cards (R16 Vol I p.23-71) - "
                          "not found.")
                return
            c7 = data[expected]
            nftg_raw = _numint(kf, c7, STD8, block.long, 1)
            nftg = 1 if nftg_raw is None else nftg_raw     # default 1
            inftg = max(iv(c7, 5), 0)
            n_sn = max(nftg, 0) if nftg != -999 else 0
            sn_cards = data[expected + 1:expected + 1 + n_sn]
            expected += 1 + n_sn + inftg                   # + 7.2 filenames
    if len(data) != expected:
        ctx.error(f"*{block.name}: found {len(data)} data cards but the "
                  f"flags imply {expected} (NAPSD={napsd}"
                  + ("" if legacy else f", NCPSD={f7 or 0}")
                  + ") - the layout was not understood, refusing to guess.")
        return

    # ── scan pass: curve registration ────────────────────────────────────
    if not edit:
        lcdam = _numint(kf, c2, STD8, block.long, 1)
        if lcdam:
            lctyp = iv(c2, 2)
            xdim = {0: FREQ, 1: DIMLESS}.get(lctyp)
            if xdim is None:
                ctx.error(f"*{block.name}: LCTYP={lctyp} is not documented "
                          "(R16 Vol I p.23-65).")
            else:
                ctx.scan.register_curve(lcdam, xdim, DIMLESS,
                                   block.name + " LCDAM")
        for li in load_cards:
            ldpsd = _numint(kf, li, STD8, block.long, 3)
            if ldpsd and base is not None:
                if ldtyp == 0:
                    ctx.scan.register_curve(ldpsd, FREQ, psd,
                                       block.name + " LDPSD")
                else:                                      # LDTYP=2: history
                    ctx.scan.register_curve(ldpsd, TIME, base,
                                       block.name + " LDPSD (time history)")
            for fi, fname in ((4, "LDVEL"), (5, "LDFLW"), (6, "LDSPN")):
                sub = _numint(kf, li, STD8, block.long, fi)
                if sub:
                    ctx.warn(f"*{block.name}: {fname}={sub} is only read "
                             "for wave loading (VAFLAG=5..7) and is ignored "
                             f"for VAFLAG={vaflag} - curve left unresolved.")
        for li in cross_cards:
            lctyp2 = iv(li, 2)
            if base is None:
                continue
            ld1 = _numint(kf, li, STD8, block.long, 3)
            ld2 = _numint(kf, li, STD8, block.long, 4)
            if ld1:
                ctx.scan.register_curve(ld1, FREQ, psd, block.name + " LDPSD1")
            if ld2:
                ydim = psd if lctyp2 == 0 else DIMLESS     # phase angle
                ctx.scan.register_curve(ld2, FREQ, ydim, block.name + " LDPSD2")
        for li in sn_cards:
            sn_lcid = _numint(kf, li, STD8, block.long, 1) or 0
            if sn_lcid > 0:
                ctx.scan.register_curve(sn_lcid, DIMLESS, PRESSURE,
                                   block.name + " S-N (cycles vs stress)")
        return

    # ── edit pass: rescale fields ────────────────────────────────────────
    for fi in (2, 3):                                      # FNMIN FNMAX
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(FREQ))
    kf.scale_field(c2, STD8, block.long, 3, ctx.fac(FREQ))  # DMPMAS (alpha)
    kf.scale_field(c2, STD8, block.long, 4, ctx.fac(TIME))  # DMPSTF (beta)
    if num(c4, 3):
        ctx.note(f"*{block.name}: TEMPER left unchanged (temperatures are "
                 "never rescaled)")
    if legacy:
        kf.scale_field(c4, STD8, block.long, 4, ctx.fac(TIME))   # TEXPOS
    if c7 is not None:
        kf.scale_field(c7, STD8, block.long, 3, ctx.fac(TIME))   # TEXPOS
    for li in sn_cards:
        sn_lcid = _numint(kf, li, STD8, block.long, 1) or 0
        if sn_lcid < 0:
            _scale_sn_fields(block, ctx, li, sn_lcid, (4, 5, 6), "Card 7.1b")
    ctx.count(block.name + (" (legacy R8/R10 layout)" if legacy else ""))


def _blank_field(kf: KFile, li: int, widths, long, fi: int) -> bool:
    """True when field `fi` of a fixed-format card holds nothing at all.

    Several frequency-domain cards are identified by a structurally BLANK
    first column (*FREQUENCY_DOMAIN_SSD Card 3, *..._ACOUSTIC_BEM Card 4,
    *..._ACOUSTIC_FEM Card 2), which a written 0 would not satisfy.
    """
    fl = kf.fields(li, widths, long)
    return fi >= len(fl) or not fl[fi][0].strip()


def _prev_header(kf: KFile, li: int) -> str:
    """Nearest preceding '$' comment line, upper-cased ('' if none).

    LS-PrePost labels every card it writes with a '$#' header naming the
    variables; that label is the only remaining signal when a structurally
    blank card is dropped from block.data (blank lines are not data lines).
    """
    for j in range(li - 1, -1, -1):
        s = kf.lines[j].strip()
        if not s:
            continue
        if s.startswith("$"):
            return s.upper()
        return ""
    return ""


# *FREQUENCY_DOMAIN_SSD Card 7 excitation ordinate by VAD (R16 Vol I
# p.23-111..23-112).  0/1 are force and pressure, 2-4 base motion, 5-7 the
# large-mass enforced equivalents, 8-11 the rotational set, 12-14 the
# DIRECT-only enforced motions.
_SSD_VAD_DIM: Dict[int, Dim] = {
    0: FORCE, 1: PRESSURE,
    2: VELOCITY, 3: ACCEL, 4: LENGTH,
    5: VELOCITY, 6: ACCEL, 7: LENGTH,
    8: MOMENT, 9: ANG_VEL, 10: ANG_ACCEL, 11: DIMLESS,
    12: VELOCITY, 13: ACCEL, 14: LENGTH,
}


def h_freq_ssd(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_SSD{_ERP,_FATIGUE,_DIRECT,_FRF}, R16 Vol I
    p.23-104..23-116.

    Card 1 MDMIN MDMAX FNMIN FNMAX RESTMD RESTDP LCFLAG RELATV - FNMIN/FNMAX
    are the natural frequencies admitted to the modal superposition.
    Card 2 DAMPF LCDAM LCTYP DMPMAS DMPSTF DMPFLG - DAMPF is a modal damping
    RATIO, DMPMAS the mass-proportional Rayleigh alpha (1/time) and DMPSTF
    the stiffness-proportional beta (time) of C = alpha*M + beta*K
    (Remark 5c, p.23-114).
    Card 3 <blank> ISTRESS MEMORY NERP STRTYP NOUT NOTYP NOVA - entirely
    flags; its first column is structurally blank, which is what identifies
    it.
    ERP adds Card 4 RO C ERPRLF ERPREF RADEFF (density, sound speed, a
    dimensionless radiation loss factor that is a curve id when negative, and
    the ERP reference POWER the dB scale is referred to) plus NERP copies of
    Card 5 PID PTYP.
    Card 7 NID NTYP DOF VAD LC1 LC2 SF VID repeats to the end of the block;
    SF is a plain multiplier.  FATIGUE appends Card 8 LCFTG after them, so it
    is the LAST data line of the block.

    Only the excitation curves carry physics that a table cannot express:
    LC1's ordinate follows VAD and LC2 is a phase angle in degrees when
    LCFLAG = 0 but the imaginary part - same dimension as LC1 - when
    LCFLAG = 1 (Remark 11, p.23-115).
    """
    kf = ctx.kf
    opts = block.name.split("_")
    erp = "ERP" in opts
    fatigue = "FATIGUE" in opts
    data = list(block.data)
    if len(data) < 3:
        ctx.error(f"*{block.name}: expected at least Cards 1-3, found "
                  f"{len(data)} data card(s).")
        return
    c1, c2 = data[0], data[1]

    # ── locate the excitation cards ──────────────────────────────────────
    if _blank_field(kf, data[2], STD8, block.long, 0):
        start = 3                                  # data[2] is Card 3
    elif "NID" in _prev_header(kf, data[2]) and "VAD" in _prev_header(
            kf, data[2]):
        # Legacy LS-PrePost short form writes Card 3 as an entirely empty
        # line, which is not a data line - the excitation cards start here.
        start = 2
        ctx.note(f"*{block.name}: Card 3 is written as an empty line "
                 "(legacy LS-PrePost short form) and carries no data - the "
                 "excitation cards were located from the '$#' header.")
    else:
        ctx.error(f"*{block.name}: the third data card has a non-blank first "
                  "column, but R16 Card 3 begins with a blank field "
                  "(p.23-107).  This is either a pre-R9 deck whose Card 3 "
                  "holds FMIN/FMAX/NFREQ (frequencies that must be rescaled) "
                  "or a shifted layout - refusing to guess; convert this "
                  "block manually.")
        return
    if erp:
        if len(data) <= start:
            ctx.error(f"*{block.name}: the ERP option requires Card 4 "
                      "(RO C ERPRLF ERPREF RADEFF) - not found.")
            return
        c4 = data[start]
        nerp = max(_numint(kf, data[2], STD8, block.long, 3) or 0, 0)
        start += 1 + nerp                          # Card 4 + NERP x Card 5
    else:
        c4 = None
    c8 = None
    if fatigue and len(data) > start:
        c8 = data[-1]                              # LCFTG is the last line
    loads = data[start:len(data) - (1 if c8 is not None else 0)]

    lcflag = _numint(kf, c1, STD8, block.long, 6) or 0

    # ── scan pass: curve registration ────────────────────────────────────
    if not edit:
        lcdam = _numint(kf, c2, STD8, block.long, 1)
        if lcdam:
            lctyp = _numint(kf, c2, STD8, block.long, 2) or 0
            xdim = {0: FREQ, 1: DIMLESS}.get(lctyp)
            if xdim is None:
                ctx.error(f"*{block.name}: LCTYP={lctyp} is not documented "
                          "(R16 Vol I p.23-106).")
            else:
                ctx.register_curve(lcdam, xdim, DIMLESS,
                                   block.name + " LCDAM")
        if c4 is not None:
            erprlf = kf.get_number(c4, STD8, block.long, 2)
            if erprlf is not None and erprlf < 0:
                ctx.register_curve(int(-erprlf), FREQ, DIMLESS,
                                   block.name + " ERPRLF(frequency)")
        for li in loads:
            vad = _numint(kf, li, STD8, block.long, 3) or 0
            ydim = _SSD_VAD_DIM.get(vad)
            if ydim is None:
                ctx.error(f"*{block.name}: VAD={vad} is not documented "
                          "(R16 Vol I p.23-111) - refusing to guess.")
                return
            lc1 = _numint(kf, li, STD8, block.long, 4)
            lc2 = _numint(kf, li, STD8, block.long, 5)
            if lc1:
                ctx.register_curve(lc1, FREQ, ydim, block.name + " LC1")
            if lc2:
                # LCFLAG=0: LC2 is a phase angle in degrees; LCFLAG=1: it is
                # the imaginary part and carries LC1's dimension.
                ctx.register_curve(lc2, FREQ,
                                   ydim if lcflag == 1 else DIMLESS,
                                   block.name + " LC2")
        if c8 is not None:
            lcftg = _numint(kf, c8, STD8, block.long, 0)
            if lcftg:
                ctx.register_curve(lcftg, FREQ, TIME,
                                   block.name + " LCFTG (duration)")
        return

    # ── edit pass ────────────────────────────────────────────────────────
    for fi in (2, 3):                                       # FNMIN FNMAX
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(FREQ))
    kf.scale_field(c2, STD8, block.long, 3, ctx.fac(FREQ))   # DMPMAS (alpha)
    kf.scale_field(c2, STD8, block.long, 4, ctx.fac(TIME))   # DMPSTF (beta)
    if c4 is not None:
        kf.scale_field(c4, STD8, block.long, 0, ctx.fac(DENSITY))   # RO
        kf.scale_field(c4, STD8, block.long, 1, ctx.fac(VELOCITY))  # C
        kf.scale_field(c4, STD8, block.long, 3, ctx.fac(POWER))     # ERPREF
    ctx.count(block.name)


def h_freq_frf(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_FRF (blank option), R16 Vol I p.23-41..23-56.

    Card 1a N1 N1TYP DOF1 VAD1 VID1 FNMAX MDMIN MDMAX - FNMAX is the maximum
    NATURAL frequency taken into the FRF sum.  VAD1 = 12 (torsion) inserts
    Card 1a.1 (N11 N11TYP) here, shifting everything after it.
    Card 2 DAMPF LCDAM LCTYP DMPMAS DMPSTF - the Rayleigh pair again (alpha
    at field 3 is 1/time, beta at field 4 is a time); note there is no
    DMPFLG, unlike *FREQUENCY_DOMAIN_SSD.
    Card 3 N2 N2TYP DOF2 VAD2 VID2 RELATV - ids and flags only.
    Card 4 FMIN FMAX NFREQ FSPACE LCFREQ RESTRT OUTPUT - FMIN/FMAX are the
    FRF output frequencies, documented as cycles/time.  Card 4 is the last
    card of the blank option.

    The excitation is a unit load, so there are no excitation curves; LCDAM
    is a modal damping ratio versus frequency (LCTYP=0) or mode number
    (LCTYP=1).  LCFREQ is refused for the same reason h_database_frequency
    refuses it: the manual never says which curve column holds the output
    frequencies.
    """
    kf = ctx.kf
    data = list(block.data)
    if len(data) < 4:
        ctx.error(f"*{block.name}: expected 4 cards (1a, 2, 3, 4), found "
                  f"{len(data)} - the SUBCASE option is a separate keyword.")
        return
    vad1 = _numint(kf, data[0], STD8, block.long, 3) or 0
    off = 1 if vad1 == 12 else 0            # Card 1a.1 (N11 N11TYP)
    if len(data) < 4 + off:
        ctx.error(f"*{block.name}: VAD1=12 (torsion) inserts Card 1a.1 "
                  "(R16 Vol I p.23-44), so 5 cards are expected - found "
                  f"{len(data)}.")
        return
    c2, c4 = data[1 + off], data[3 + off]
    lcfreq = _numint(kf, c4, STD8, block.long, 4)
    if lcfreq:
        ctx.error(f"*{block.name}: LCFREQ={lcfreq} output-frequency curves "
                  "are not modelled (the manual does not document which "
                  "curve column holds the frequencies) - use FMIN/FMAX/NFREQ "
                  "or convert manually.")
        return
    if not edit:
        lcdam = _numint(kf, c2, STD8, block.long, 1)
        if lcdam:
            lctyp = _numint(kf, c2, STD8, block.long, 2) or 0
            xdim = {0: FREQ, 1: DIMLESS}.get(lctyp)
            if xdim is None:
                ctx.error(f"*{block.name}: LCTYP={lctyp} is not documented "
                          "(R16 Vol I p.23-46).")
            else:
                ctx.register_curve(lcdam, xdim, DIMLESS,
                                   block.name + " LCDAM")
        return
    kf.scale_field(data[0], STD8, block.long, 5, ctx.fac(FREQ))   # FNMAX
    kf.scale_field(c2, STD8, block.long, 3, ctx.fac(FREQ))        # DMPMAS
    kf.scale_field(c2, STD8, block.long, 4, ctx.fac(TIME))        # DMPSTF
    for fi in (0, 1):                                             # FMIN FMAX
        kf.scale_field(c4, STD8, block.long, fi, ctx.fac(FREQ))
    if len(data) > 4 + off:
        ctx.warn(f"*{block.name}: {len(data) - 4 - off} trailing card(s) "
                 "beyond Card 4 left unscaled - verify manually.")
    ctx.count(block.name)


# *FREQUENCY_DOMAIN_RESPONSE_SPECTRUM Card 6a spectrum by LCTYP (R16 Vol I
# p.23-85..23-86): 0-4 are plotted against natural FREQUENCY, 5-9 the same
# five quantities against natural PERIOD, 10-12 are base motion TIME
# histories.
_RS_LCTYP: Dict[int, Tuple[Dim, Dim]] = {
    0: (FREQ, VELOCITY), 1: (FREQ, ACCEL), 2: (FREQ, LENGTH),
    3: (FREQ, FORCE), 4: (FREQ, PRESSURE),
    5: (TIME, VELOCITY), 6: (TIME, ACCEL), 7: (TIME, LENGTH),
    8: (TIME, FORCE), 9: (TIME, PRESSURE),
    10: (TIME, VELOCITY), 11: (TIME, ACCEL), 12: (TIME, LENGTH),
}


def h_freq_response_spectrum(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_RESPONSE_SPECTRUM (blank OPTION1), R16 Vol I
    p.23-80..23-92.

    Card 1 MDMIN MDMAX FNMIN FNMAX RESTRT MCOMB RELATV MPRS - FNMIN/FNMAX are
    the natural frequencies admitted to the modal superposition.  Three flags
    on that card insert optional cards BEFORE the damping card, which is why
    a fixed table cannot find it: MCOMB=99 adds Cards 2.1 (MCOMB1 MCOMB2) and
    2.2 (W1 W2), MPRS=2 adds Card 3 (R40), and the MISSING_MASS_CORRECTION
    option adds Card 4 (ZPA + an A70 file name) whose first field is a
    zero-period ACCELERATION.
    Card 5 DAMPF LCDAMP LDTYP DMPMAS DMPSTF - the Rayleigh pair (alpha 1/time
    at field 3, beta time at field 4).
    Card 6a LCTYP DOF LC/TBID SF VID LNID LNTYP INFLAG repeats to the end of
    the block, one per input spectrum; SF is a plain multiplier and LCTYP
    picks the spectrum's axes.

    The DDAM option is a separate keyword name and stays unknown: its
    Card 6b carries a 6-way UNIT declaration and its STD=-1 coefficients are
    defined against a modal weight in kips.
    """
    kf = ctx.kf
    opts = block.name.split("_")
    data = list(block.data)
    if not data:
        return
    c1 = data[0]
    mcomb = _numint(kf, c1, STD8, block.long, 5) or 0
    mprs = _numint(kf, c1, STD8, block.long, 7) or 0
    idx = 1
    if mcomb == 99:
        idx += 2                                   # Cards 2.1 and 2.2
    if mprs == 2:
        idx += 1                                   # Card 3 (R40)
    zpa_card = None
    if "MISSING" in opts and "MASS" in opts and "CORRECTION" in opts:
        zpa_card = data[idx] if idx < len(data) else None
        idx += 1                                   # Card 4 (ZPA + FILENAME)
    if idx >= len(data):
        ctx.error(f"*{block.name}: the flags on Card 1 (MCOMB={mcomb}, "
                  f"MPRS={mprs}) imply the damping card at data line "
                  f"{idx + 1}, but the block has only {len(data)}.")
        return
    c5 = data[idx]
    spectra = data[idx + 1:]

    if not edit:
        lcdamp = _numint(kf, c5, STD8, block.long, 1)
        if lcdamp:
            ldtyp = _numint(kf, c5, STD8, block.long, 2) or 0
            xdim = {0: FREQ, 1: DIMLESS}.get(ldtyp)
            if xdim is None:
                ctx.error(f"*{block.name}: LDTYP={ldtyp} is not documented "
                          "(R16 Vol I p.23-84).")
            else:
                ctx.register_curve(lcdamp, xdim, DIMLESS,
                                   block.name + " LCDAMP")
        for li in spectra:
            lctyp = _numint(kf, li, STD8, block.long, 0) or 0
            axes = _RS_LCTYP.get(lctyp)
            if axes is None:
                ctx.error(f"*{block.name}: LCTYP={lctyp} is not documented "
                          "(R16 Vol I p.23-85) - refusing to guess.")
                return
            lcid = _numint(kf, li, STD8, block.long, 2)
            if lcid:
                ctx.register_curve(lcid, axes[0], axes[1],
                                   block.name + " input spectrum")
                # LC/TBID may name a *DEFINE_TABLE whose value axis is a
                # critical-damping ratio (dimensionless) and whose member
                # curves are the spectra themselves.
                ctx.register_table(lcid, DIMLESS, axes[0], axes[1])
        return

    for fi in (2, 3):                                       # FNMIN FNMAX
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(FREQ))
    if zpa_card is not None:
        kf.scale_field(zpa_card, STD8, block.long, 0, ctx.fac(ACCEL))  # ZPA
    kf.scale_field(c5, STD8, block.long, 3, ctx.fac(FREQ))   # DMPMAS (alpha)
    kf.scale_field(c5, STD8, block.long, 4, ctx.fac(TIME))   # DMPSTF (beta)
    ctx.count(block.name)


# *FREQUENCY_DOMAIN_ACOUSTIC_BEM Card 4.1 curve ordinate by BEMTYP (R16
# Vol I p.23-12..23-13); the abscissa is always a frequency.
_BEM_TYPE_DIM: Dict[int, Dim] = {
    0: DIMLESS, 1: DIMLESS,          # velocity from the transient run
    2: PRESSURE,                     # real/imaginary pressure
    3: VELOCITY,                     # real/imaginary normal velocity
    4: ACOUST_IMP,                   # real/imaginary acoustic impedance
    5: DIMLESS,                      # absorption coefficient
}


def h_freq_acoustic_bem(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_ACOUSTIC_BEM{_ATV,_MATV,_POWER,_HALF_SPACE,
    _PANEL_CONTRIBUTION}, R16 Vol I p.23-4..23-17.

    Card 1 RO C FMIN FMAX NFREQ DTOUT TSTART PREF - fluid density, sound
    speed, output frequency band, the binary-output time interval and start
    time, and the reference pressure the dB scale is referred to.  C < 0
    means |C| is a *FREQUENCY_DOMAIN_ACOUSTIC_SOUND_SPEED curve id and must
    never be scaled.
    Card 2 NSIDEXT TYPEXT NSIDINT TYPINT FFTWIN TRSLT IPFILE IUNITS - flags,
    plus IUNITS, a unit-system DECLARATION that has to be rewritten together
    with the values (see below).  FFTWIN = 5 or 8 inserts Card 2.1a
    (T_HOLD DECAY, T_HOLD a hold-off TIME or, when negative, a per-panel
    curve id); FFTWIN = 7 inserts Card 2.1b (ALPHA, the dimensionless Kaiser
    shape parameter).  Both shift every later card.
    Card 3 holds iteration counts and RELATIVE solver tolerances.
    Card 4 <blank> NBC RESTRT IEDGE NOEL NFRUP VELOUT DBA - note the blank
    first column: NBC, the number of Card 4.1 lines, sits at field 2.
    Card 4.1 SSID SSTYPE NORM BEMTYP LC1 LC2 - ids and flags; BEMTYP picks
    the LC1/LC2 curve ordinate and BEMTYP < 0 makes |BEMTYP| itself a
    normal-velocity-amplitude curve id.
    Card 5 (PANEL_CONTRIBUTION) is a node-set id and Card 6 (HALF_SPACE) a
    *DEFINE_PLANE id plus the R17 reflection coefficient RC - both follow the
    NBC Card 4.1 lines, Card 5 before Card 6.
    """
    kf = ctx.kf
    opts = block.name.split("_")
    data = list(block.data)
    if len(data) < 4:
        ctx.error(f"*{block.name}: expected at least Cards 1-4, found "
                  f"{len(data)} data card(s).")
        return
    c1, c2 = data[0], data[1]
    fftwin = _numint(kf, c2, STD8, block.long, 4) or 0
    hold_card = None
    idx = 2
    if fftwin in (5, 8):
        hold_card = data[idx] if idx < len(data) else None
        idx += 1                                   # Card 2.1a T_HOLD DECAY
    elif fftwin == 7:
        idx += 1                                   # Card 2.1b ALPHA
    idx += 1                                       # Card 3
    if idx >= len(data):
        ctx.error(f"*{block.name}: FFTWIN={fftwin} implies Card 4 at data "
                  f"line {idx + 1}, but the block has only {len(data)}.")
        return
    c4 = data[idx]
    if not _blank_field(kf, c4, STD8, block.long, 0):
        ctx.error(f"*{block.name}: Card 4 must begin with a blank field "
                  "(NBC sits in the second column, R16 Vol I p.23-10) but "
                  f"data line {idx + 1} does not - the card layout was not "
                  "understood, refusing to guess.")
        return
    nbc_raw = _numint(kf, c4, STD8, block.long, 1)
    nbc = 1 if nbc_raw is None else max(nbc_raw, 0)          # default 1
    bc_cards = data[idx + 1:idx + 1 + nbc]
    expected = idx + 1 + nbc
    if "PANEL_CONTRIBUTION" in block.name:
        expected += 1
    if "HALF_SPACE" in block.name:
        expected += 1
    if len(data) != expected:
        ctx.error(f"*{block.name}: found {len(data)} data cards but the "
                  f"flags imply {expected} (FFTWIN={fftwin}, NBC={nbc}) - "
                  "the layout was not understood, refusing to guess.")
        return
    c = kf.get_number(c1, STD8, block.long, 1)

    if not edit:
        if c is not None and c < 0:
            ctx.register_curve(int(-c), FREQ, VELOCITY,
                               block.name + " complex sound speed")
        if hold_card is not None:
            th = kf.get_number(hold_card, STD8, block.long, 0)
            if th is not None and th < 0:
                ctx.register_curve(int(-th), DIMLESS, TIME,
                                   block.name + " T_HOLD(panel)")
        for li in bc_cards:
            bemtyp = _numint(kf, li, STD8, block.long, 3) or 0
            if bemtyp < 0:
                ctx.register_curve(-bemtyp, FREQ, VELOCITY,
                                   block.name + " |BEMTYP| velocity")
                continue
            ydim = _BEM_TYPE_DIM.get(bemtyp)
            if ydim is None:
                ctx.error(f"*{block.name}: BEMTYP={bemtyp} is not documented "
                          "(R16 Vol I p.23-12) - refusing to guess.")
                return
            for fi in (4, 5):                                # LC1 LC2
                lcid = _numint(kf, li, STD8, block.long, fi)
                if lcid and ydim is not DIMLESS:
                    ctx.register_curve(lcid, FREQ, ydim,
                                       block.name + " LC1/LC2")
        return

    # ── edit pass ────────────────────────────────────────────────────────
    kf.scale_field(c1, STD8, block.long, 0, ctx.fac(DENSITY))    # RO
    if c is not None and c > 0:
        kf.scale_field(c1, STD8, block.long, 1, ctx.fac(VELOCITY))
    for fi in (2, 3):                                            # FMIN FMAX
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(FREQ))
    for fi in (5, 6):                                        # DTOUT TSTART
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(TIME))
    kf.scale_field(c1, STD8, block.long, 7, ctx.fac(PRESSURE))   # PREF
    if hold_card is not None:
        th = kf.get_number(hold_card, STD8, block.long, 0)
        if th is None or th >= 0:
            kf.scale_field(hold_card, STD8, block.long, 0, ctx.fac(TIME))
    # IUNITS: 0 means "no unit change requested" and stays 0; any other value
    # declares the deck's system and must be remapped to the destination's.
    iunits = _numint(kf, c2, STD8, block.long, 7) or 0
    if iunits:
        src_sys = BEM_UNIT_SYSTEMS.get(iunits)
        if src_sys is None:
            ctx.error(f"*{block.name}: IUNITS={iunits} is not a documented "
                      "value (0-4, R16 Vol I p.23-8) - refusing to guess "
                      "what unit system the acoustic input declares.")
            return
        if ctx.src is not None and src_sys != ctx.src:
            ctx.warn(f"*{block.name}: IUNITS={iunits} declares "
                     f"{src_sys.key} but the deck is being converted as "
                     f"{ctx.src.key} - the flag and the model units disagree "
                     "in the SOURCE deck; the values are scaled as MODEL "
                     "units and IUNITS is rewritten. Verify.")
        new_units = BEM_UNITS.get((ctx.dst.mass, ctx.dst.length, ctx.dst.time))
        if new_units is None:
            supported = ", ".join(f"{v}={s.key}"
                                  for v, s in sorted(BEM_UNIT_SYSTEMS.items()))
            ctx.error(f"*{block.name}: IUNITS={iunits} declares the acoustic "
                      "input's unit system, but the IUNITS table (R16 Vol I "
                      f"p.23-8) has no value for {ctx.dst.key}. Supported "
                      f"destinations: {supported}. Pick one of those, or set "
                      "IUNITS=0 in the source deck and convert manually.")
            return
        w = 20 if block.long else 10
        kf.set_field(c2, STD8, block.long, 7, str(new_units).rjust(w))
        ctx.count(block.name + f" (IUNITS->{new_units})")
        return
    ctx.count(block.name)


# *FREQUENCY_DOMAIN_ACOUSTIC_FEM Card 4 boundary excitation by VAD (R16
# Vol I p.23-24).  The x1 spellings are amplitude + PHASE (the second curve
# is an angle in degrees), the x2 spellings real + imaginary (both curves
# carry the physical dimension).
_FEM_VAD_DIM: Dict[int, Dim] = {
    11: VELOCITY, 12: VELOCITY,
    21: ACCEL, 22: ACCEL,
    31: LENGTH, 32: LENGTH,
    41: ACOUST_IMP, 42: ACOUST_IMP,
    51: PRESSURE, 52: PRESSURE,
}


def h_freq_acoustic_fem(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_ACOUSTIC_FEM{_EIGENVALUE,_MODAL}, R16 Vol I
    p.23-20..23-27.

    Card MDL (MODAL option only, and only when the fifth column is empty)
    MDMIN MDMAX FNMIN FNMAX precedes Card 1 - fields 2 and 3 are natural
    frequencies.
    Card 1 RO C FMIN FMAX NFREQ DTOUT TSTART PREF, identical in meaning to
    the BEM Card 1, including the C < 0 sound-speed-curve rule.
    Card 2 <blank> FFTWIN MTXDMP RESTRT and Card 3 PID PTYP are flags and
    ids.  Card 4 SID STYP VAD DOF LCID1 LCID2 SF VID repeats once per
    boundary condition and Card 5 NID NTYP IPFILE DBA closes the block
    (absent under EIGENVALUE).  The manual gives NO count field for the
    Card 4 repeats, so they are located by counting backwards from the end
    of the block - the last data card is Card 5.

    Only Card 1 (and Card MDL) carry dimensional fields; SF on Card 4 is a
    plain multiplier on the curves.  VAD selects the LCID1/LCID2 ordinate.
    """
    kf = ctx.kf
    opts = block.name.split("_")
    data = list(block.data)
    if not data:
        return
    idx = 0
    if ("MODAL" in opts
            and _blank_field(kf, data[0], STD8, block.long, 4)):
        if edit:                                        # Card MDL FNMIN FNMAX
            for fi in (2, 3):
                kf.scale_field(data[0], STD8, block.long, fi, ctx.fac(FREQ))
        idx = 1
    if len(data) <= idx:
        ctx.error(f"*{block.name}: Card 1 not found.")
        return
    c1 = data[idx]
    if len(data) > idx + 1 and not _blank_field(kf, data[idx + 1], STD8,
                                                block.long, 0):
        ctx.error(f"*{block.name}: Card 2 must begin with a blank field "
                  "(FFTWIN sits in the second column, R16 Vol I p.23-22) but "
                  f"data line {idx + 2} does not - the card layout was not "
                  "understood, refusing to guess.")
        return
    rest = data[idx + 3:]                       # everything after Card 3
    if "EIGENVALUE" in opts:
        bc_cards = rest                         # Card 5 is absent
    else:
        bc_cards = rest[:-1]                    # the last card is Card 5
    c = kf.get_number(c1, STD8, block.long, 1)

    if not edit:
        if c is not None and c < 0:
            ctx.register_curve(int(-c), FREQ, VELOCITY,
                               block.name + " complex sound speed")
        for li in bc_cards:
            vad = _numint(kf, li, STD8, block.long, 2) or 0
            if vad in (0, 1, 2):                # SSD / transient / opening
                continue
            ydim = _FEM_VAD_DIM.get(vad)
            if ydim is None:
                ctx.error(f"*{block.name}: VAD={vad} is not documented "
                          "(R16 Vol I p.23-24) - refusing to guess.")
                return
            lcid1 = _numint(kf, li, STD8, block.long, 4)
            lcid2 = _numint(kf, li, STD8, block.long, 5)
            if lcid1:
                ctx.register_curve(lcid1, FREQ, ydim, block.name + " LCID1")
            if lcid2:
                # odd VAD -> LCID2 is a phase angle in degrees; even VAD ->
                # it is the imaginary part and carries LCID1's dimension.
                ctx.register_curve(lcid2, FREQ,
                                   DIMLESS if vad % 2 else ydim,
                                   block.name + " LCID2")
        return

    kf.scale_field(c1, STD8, block.long, 0, ctx.fac(DENSITY))    # RO
    if c is not None and c > 0:
        kf.scale_field(c1, STD8, block.long, 1, ctx.fac(VELOCITY))
    for fi in (2, 3):                                            # FMIN FMAX
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(FREQ))
    for fi in (5, 6):                                        # DTOUT TSTART
        kf.scale_field(c1, STD8, block.long, fi, ctx.fac(TIME))
    kf.scale_field(c1, STD8, block.long, 7, ctx.fac(PRESSURE))   # PREF
    ctx.count(block.name)


def h_freq_incident_wave(block: Block, ctx, edit: bool) -> None:
    """*FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE, R16 Vol I p.23-37..23-38.

    Repeating cards TYPE MAG XC YC ZC, one per incident wave, running to the
    next keyword.  TYPE flips the meaning of three fields on the same card:
    "XC, YC, ZC: Direction cosines for the plane wave (TYPE = 1) or
    coordinates of the point source for the spherical wave (TYPE = 2)".
    MAG < 0 makes |MAG| a frequency-dependent magnitude curve id.

    For TYPE = 1 the magnitude is the plane-wave pressure amplitude A of
    p = A*exp(-ik(ax+by+cz)) - a PRESSURE.  For TYPE = 2 Remark 2 gives
    p(r) = A*exp(-ikr)/r, which makes A a pressure TIMES a length, but the
    manual never states the units of MAG for that branch.  Rather than pick
    between two defensible readings on a field nobody could confirm, the
    spherical branch is refused.
    """
    kf = ctx.kf
    for li in block.data:
        wtype = _numint(kf, li, STD8, block.long, 0)
        wtype = 1 if wtype is None else wtype
        mag = kf.get_number(li, STD8, block.long, 1)
        if wtype not in (1, 2):
            ctx.error(f"*{block.name}: TYPE={wtype} is not documented "
                      "(1 = plane wave, 2 = spherical wave, R16 Vol I "
                      "p.23-37) - refusing to guess.")
            return
        if wtype == 2 and mag is not None and mag > 0:
            ctx.error(f"*{block.name}: TYPE=2 (spherical wave) with "
                      f"MAG={mag}. Remark 2 (R16 Vol I p.23-38) defines "
                      "p(r) = A*exp(-ikr)/r, which makes MAG a pressure "
                      "TIMES a length, but the manual never states its "
                      "units - kunit refuses to pick between "
                      "pressure*length and plain pressure. Convert this "
                      "incident wave manually (the XC/YC/ZC point-source "
                      "coordinates are lengths).")
            return
        if mag is not None and mag < 0:
            if not edit:
                ctx.register_curve(int(-mag), FREQ, PRESSURE,
                                   block.name + " magnitude(frequency)")
        elif edit and wtype == 1:
            kf.scale_field(li, STD8, block.long, 1, ctx.fac(PRESSURE))
        if edit and wtype == 2:
            for fi in (2, 3, 4):                          # XC YC ZC
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(LENGTH))
    if edit:
        ctx.count(block.name)


def h_database_frequency_ascii(block: Block, ctx, edit: bool) -> None:
    """*DATABASE_FREQUENCY_ASCII_{OPTION1}_{OPTION2}, R16 Vol I
    p.16-91..16-93.

    One card - FMIN FMAX NFREQ FSPACE LCFREQ - shared by every OPTION1
    (NODOUT_SSD, ELOUT_SSD, NODFOR_SSD, SECFORC_SSD, NODOUT_PSD, ELOUT_PSD);
    OPTION1 only selects which binout database the frequencies apply to.
    FMIN/FMAX are documented verbatim as "(cycles/time)".  Under the SUBCASE
    option the card repeats once per *FREQUENCY_DOMAIN_SSD_SUBCASE loading
    case (Remark 8), so every data line is scaled.

    LCFREQ names a curve listing the output frequencies, but the manual never
    says which curve column holds them - refused here exactly as
    h_database_frequency refuses it for the binary card.
    """
    kf = ctx.kf
    for li in block.data:
        lcfreq = _numint(kf, li, STD8, block.long, 4)
        if lcfreq:
            ctx.error(f"*{block.name}: LCFREQ={lcfreq} output-frequency "
                      "curves are not modelled (the manual does not document "
                      "which curve column holds the frequencies, R16 Vol I "
                      "p.16-93) - use FMIN/FMAX/NFREQ or convert manually.")
            continue
        if edit:
            for fi in (0, 1):                                # FMIN FMAX
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(FREQ))
    if edit:
        ctx.count(block.name)


def h_database_frequency(block: Block, ctx, edit: bool) -> None:
    """R16 Vol I p.16-94..16-100 (*DATABASE_FREQUENCY_BINARY_OPTION1_
    {OPTION2}).  Card 1a BINARY SF - - PSETID holds flags plus the
    dimensionless RMS-combination scale factor SF; only D3PSD and D3SSD read
    the extra Card 2b FMIN FMAX NFREQ FSPACE LCFREQ, whose FMIN/FMAX are
    output frequencies explicitly documented as cycles/time (p.16-98).
    LCFREQ names a curve listing the output frequencies, but the manual does
    not say which curve column carries them, so decks using it are refused.
    The SUMMATION option replaces Card 1 with filename cards, D3ACC appends
    node-id cards and BINARY=3 (D3RMS/D3SPCM) appends an ISTATE FILENAME
    card - nothing dimensional in any of those."""
    kf = ctx.kf
    parts = block.name.split("_")
    if "SUMMATION" in parts:
        return
    opt1 = parts[3] if len(parts) > 3 else ""
    if opt1 not in ("D3PSD", "D3SSD"):
        return
    for li in block.data[1:]:              # Card 2b (repeats under SUBCASE)
        lcfreq = _numint(kf, li, STD8, block.long, 4)
        if lcfreq:
            ctx.error(f"*{block.name}: LCFREQ={lcfreq} output-frequency "
                      "curves are not modelled (the manual does not document "
                      "which curve column holds the frequencies, R16 Vol I "
                      "p.16-98) - use FMIN/FMAX/NFREQ or convert manually.")
            continue
        if edit:
            for fi in (0, 1):                              # FMIN FMAX
                kf.scale_field(li, STD8, block.long, fi, ctx.fac(FREQ))
    if edit:
        ctx.count(block.name)


def h_cnrb_inertia(block: Block, ctx, edit: bool) -> None:
    if not edit:
        return
    kf = ctx.kf
    d = _strip_title(block, list(block.data))
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


def x_part_inertia(block: Block, ctx) -> None:
    """Edit-time check: IRCS=1 (R16 Vol I p.37-13) inserts an extra local-
    coordinate-system card after the inertia card, which the fixed 5-card
    group cannot model - refuse rather than scale shifted cards."""
    kf = ctx.kf
    data = list(block.data)
    for i in range(2, len(data), 5):            # card 3 of each part
        if (_numint(kf, data[i], STD8, block.long, 4) or 0):
            ctx.error(f"*{block.name}: IRCS != 0 adds a local-coordinate "
                      "card the fixed layout cannot model - convert this "
                      "part manually.")
            return


def x_scan_part(block: Block, ctx) -> None:
    """Collect (pid, secid, mid) so DRO can classify discrete materials.

    Works for every *PART variant with a Spec: the ids card is always the
    second card of the repeating group (after the heading), and the group
    length gives the per-part period (2 for PART, 3 for PART_CONTACT,
    5 for PART_INERTIA)."""
    kf = ctx.kf
    kind, spec = resolve(block.name)
    step = (len(spec.group) if kind == "spec" and spec.group else 2)
    data = list(block.data)
    for i in range(1, len(data), step):
        pid = _numint(kf, data[i], STD8, block.long, 0)
        secid = _numint(kf, data[i], STD8, block.long, 1)
        mid = _numint(kf, data[i], STD8, block.long, 2)
        if pid:
            ctx.scan.part_links.append((pid, secid, mid))


CUSTOM: Dict[str, Callable] = {
    "DEFINE_CURVE": h_define_curve,
    "DEFINE_TABLE": h_define_table,
    "DEFINE_BOX": h_define_box,
    "DEFINE_VECTOR": h_define_vector,
    "ALE_STRUCTURED_FSI": h_ale_structured_fsi,
    "ALE_STRUCTURED_FSI_ABEXT": h_ale_structured_fsi,
    "ALE_STRUCTURED_FSI_ABINT": h_ale_structured_fsi,
    "ALE_STRUCTURED_MESH_CONTROL_POINTS": h_ale_control_points,
    "ALE_STRUCTURED_MESH_VOLUME_FILLING": h_ale_volume_filling,
    "BOUNDARY_SALE_MESH_FACE": h_boundary_sale_mesh_face,
    "INITIAL_VOLUME_FRACTION_GEOMETRY": h_initial_volume_fraction_geometry,
    "ELEMENT_SPH": h_element_sph,
    "ELEMENT_SPH_VOLUME": h_element_sph,
    "LOAD_BLAST_ENHANCED": h_blast,
    "LOAD_BLAST": h_blast,
    "LOAD_GRAVITY_PART": h_load_gravity_part,
    "LOAD_GRAVITY_PART_SET": h_load_gravity_part,
    "MAT_ADD_FATIGUE": h_mat_add_fatigue,
    "MAT_ADD_FATIGUE_EN": h_mat_add_fatigue,
    "FREQUENCY_DOMAIN_RANDOM_VIBRATION": h_freq_random_vibration,
    "FREQUENCY_DOMAIN_RANDOM_VIBRATION_FATIGUE": h_freq_random_vibration,
    # The SSD options that add no card of their own (DIRECT, FRF) share the
    # base layout; ERP and FATIGUE are handled inside the handler.  SUBCASE
    # interleaves character-data cards and stays unknown.
    "FREQUENCY_DOMAIN_SSD": h_freq_ssd,
    "FREQUENCY_DOMAIN_SSD_ERP": h_freq_ssd,
    "FREQUENCY_DOMAIN_SSD_FATIGUE": h_freq_ssd,
    "FREQUENCY_DOMAIN_SSD_FRF": h_freq_ssd,
    "FREQUENCY_DOMAIN_SSD_DIRECT": h_freq_ssd,
    "FREQUENCY_DOMAIN_SSD_DIRECT_FREQUENCY_DEPENDENT": h_freq_ssd,
    "FREQUENCY_DOMAIN_FRF": h_freq_frf,
    "FREQUENCY_DOMAIN_RESPONSE_SPECTRUM": h_freq_response_spectrum,
    "FREQUENCY_DOMAIN_RESPONSE_SPECTRUM_MISSING_MASS_CORRECTION":
        h_freq_response_spectrum,
    "FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE": h_freq_incident_wave,
    "AIRBAG_SIMPLE_AIRBAG_MODEL": h_airbag_simple,
    "AIRBAG_SIMPLE_PRESSURE_VOLUME": h_airbag_simple,
    "LOAD_THERMAL_LOAD_CURVE": h_load_thermal_load_curve,
    "LOAD_THERMAL_VARIABLE": h_load_thermal_variable,
    "LOAD_THERMAL_VARIABLE_NODE": h_load_thermal_variable_node,
    "BOUNDARY_TEMPERATURE_NODE": h_boundary_temperature,
    "BOUNDARY_TEMPERATURE_SET": h_boundary_temperature,
    "BOUNDARY_CONVECTION_SET": h_boundary_convection,
    "BOUNDARY_RADIATION_SET": h_boundary_radiation,
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
    "MAT_FRAZER_NASH_RUBBER_MODEL": h_mat_frazer_nash,
    "MAT_INV_HYPERBOLIC_SIN": h_mat_inv_hyperbolic_sin,
    "MAT_SPOTWELD": h_mat_spotweld,
    "MAT_THERMAL_ISOTROPIC": h_mat_thermal_isotropic,
    "MAT_GAS_MIXTURE": h_mat_gas_mixture,
    "MAT_FABRIC": h_mat_fabric,
    "MAT_ENHANCED_COMPOSITE_DAMAGE": h_mat_enhanced_composite_damage,
    "MAT_SIMPLIFIED_RUBBER/FOAM": h_mat_simplified_rubber_foam,
    "ICFD_BOUNDARY_PRESCRIBED_VEL": h_icfd_prescribed_vel,
    "ICFD_BOUNDARY_PRESCRIBED_PRE": h_icfd_prescribed_pre,
    "ICFD_BOUNDARY_PRESCRIBED_TEMP": h_icfd_prescribed_temp,
    "ICFD_CONTROL_TIME": h_icfd_control_time,
    "ICFD_INITIAL": h_icfd_initial,
    "MESH_BL": h_mesh_bl,
    "CONSTRAINED_LAGRANGE_IN_SOLID": h_lagrange_in_solid,
    "CONSTRAINED_NODAL_RIGID_BODY_INERTIA": h_cnrb_inertia,
    "CONSTRAINED_JOINT_STIFFNESS_GENERALIZED": h_joint_stiffness_generalized,
    "CONSTRAINED_SHELL_IN_SOLID": h_shell_in_solid,
    "CONSTRAINED_SHELL_IN_SOLID_PENALTY": h_shell_in_solid,
    "INITIAL_STRESS_SHELL": h_initial_stress_shell,
    "INITIAL_STRESS_SHELL_SET": h_initial_stress_shell,
    "INITIAL_STRESS_SOLID": h_initial_stress_solid,
    "INITIAL_STRESS_SOLID_SET": h_initial_stress_solid,
}


# ─────────────────────────────────────────────────────────────────────────────
# Prefix-router rules.  resolve() first consults the exact-match tables (HARD /
# CUSTOM / SPEC / WHITELIST) and WHITELIST_PREFIXES; only a name that matches
# none of those is walked past this ORDERED list, and the FIRST rule whose
# prefix matches the raw keyword name decides the verdict.  Order is
# significant - e.g. DATABASE_FREQUENCY_BINARY must be tried before
# DATABASE_FREQUENCY_, and both before DATABASE_.  To add a keyword family,
# insert a rule at the correct precedence point; do not reorder existing ones.
# Each rule is (prefix, resolver) where resolver(name) -> (kind, payload).
# ─────────────────────────────────────────────────────────────────────────────

def _rule_load_body(name):
    # GENERALIZED / POROUS / GENERALIZED_SET_NODE... have different layouts
    # (N1 N2 LCID DRLCID XC YC ZC + accel card) - not modelled
    if name.rsplit("_", 1)[-1] in ("X", "Y", "Z", "RX", "RY", "RZ"):
        return "custom", h_load_body
    return "unknown", None


def _rule_prescribed_motion(name):
    # SET_BOX / SET_EDGE_UVW / SET_FACE_XYZ / SET_LINE / SET_POINT_UVW
    # insert extra geometry cards the plain layout cannot model
    if any(t in name for t in ("_BOX", "_EDGE", "_FACE", "_LINE", "_POINT")):
        return "unknown", None
    return "custom", h_prescribed_motion


def _rule_rigidwall_planar(name):
    return "custom", h_rigidwall_planar


def _rule_contact(name):
    # families whose card 2/3 layouts differ from the 3D penalty
    # layout h_contact models must not be routed there.
    # MORTAR contacts use the same 3D penalty-card layout h_contact
    # models (Cards 1-3 SSID/MSID.., FS FD DC VC.., SFS SFM SST MST..),
    # only adding optional MPP/advanced cards that h_contact tolerates -
    # so they are NOT excluded here (R16 Vol I *CONTACT_..._MORTAR); the
    # MPP spellings only PREPEND their own dimensionless cards, which
    # h_contact's MPP plan handles.  The THERMAL option however inserts a
    # THRM card between the mandatory and optional cards (card-order
    # table, R16 Vol I p.11-6..11-7) and is not modelled - keep unknown.
    # *CONTACT_DRAWBEAD's own Card 4.1 is modelled, but the _BENDING and
    # _INITIALIZE options insert Cards 4.2 / 4.3 ahead of everything that
    # follows (p.11-53..11-56) and are not.
    if (name.startswith("CONTACT_TIEBREAK")
            or "DAMPING" in name or "THERMAL" in name
            or ("DRAWBEAD" in name
                and ("BENDING" in name or "INITIALIZE" in name))
            or name.startswith("CONTACT_2D") or "ENTITY" in name
            or "GEBOD" in name or "INTERIOR" in name
            or "GUIDED_CABLE" in name or "COUPLING" in name
            or "AUTO_MOVE" in name):
        return "unknown", None
    return "custom", h_contact


def _rule_database_frequency_binary(name):
    return "custom", h_database_frequency


def _rule_database_frequency_ascii(name):
    return "custom", h_database_frequency_ascii


def _rule_database_frequency(name):
    return "unknown", None      # other variants: layout not modelled


def _rule_rigidwall_geometric(name):
    return "custom", h_rigidwall_geometric


def _rule_freq_acoustic_bem(name):
    return "custom", h_freq_acoustic_bem


def _rule_freq_acoustic_fem(name):
    return "custom", h_freq_acoustic_fem


def _rule_database_binary(name):
    tail = name[len("DATABASE_BINARY_"):]
    if tail in ("D3DUMP", "RUNRSF", "D3DRLF",   # cycle intervals
                "D3PROP"):                      # flags only
        return "white", None
    if tail in ("D3PLOT", "D3PART", "D3THDT", "INTFOR", "FSIFOR",
                "BLSTFOR", "D3MEAN"):
        return "custom", h_database_dt
    return "unknown", None


def _rule_database(name):
    # only the ASCII output-file keywords have 'DT' as field 0
    # (R16 Vol I *DATABASE_OPTION table); anything else (e.g.
    # NODAL_FORCE_GROUP: NSID CID, TRACER: TIME X Y Z) must not have
    # its first field blindly scaled as a time
    if name[len("DATABASE_"):] in _DATABASE_ASCII:
        return "custom", h_database_dt
    return "unknown", None


def _rule_control(name):
    return "soft", ("not in the dimension table - left unchanged; "
                    "most CONTROL cards are flags, but verify")


_ROUTER_RULES: List[Tuple[str, Callable[[str], Tuple[str, object]]]] = [
    ("LOAD_BODY_", _rule_load_body),
    ("BOUNDARY_PRESCRIBED_MOTION", _rule_prescribed_motion),
    ("RIGIDWALL_PLANAR", _rule_rigidwall_planar),
    ("RIGIDWALL_GEOMETRIC", _rule_rigidwall_geometric),
    ("CONTACT_", _rule_contact),
    ("FREQUENCY_DOMAIN_ACOUSTIC_BEM", _rule_freq_acoustic_bem),
    ("FREQUENCY_DOMAIN_ACOUSTIC_FEM", _rule_freq_acoustic_fem),
    # DATABASE_ family: most-specific prefix first (see order note above)
    ("DATABASE_FREQUENCY_BINARY", _rule_database_frequency_binary),
    ("DATABASE_FREQUENCY_ASCII", _rule_database_frequency_ascii),
    ("DATABASE_FREQUENCY_", _rule_database_frequency),
    ("DATABASE_BINARY_", _rule_database_binary),
    ("DATABASE_", _rule_database),
    ("CONTROL_", _rule_control),
]


def resolve(name: str):
    """Classify a keyword. Returns (kind, payload):
    kind in {spec, custom, white, soft, hard, unknown}.

    Precedence: the exact-match tables and the _TITLE/_ID + _MAT_ALIASES
    normalisation are consulted first, then the ordered prefix _ROUTER_RULES."""
    base = name
    for opt in ("_TITLE", "_ID"):
        if base.endswith(opt):
            base = base[: -len(opt)]
    base = _MAT_ALIASES.get(base, base)
    if name in HARD_FLAGS or base in HARD_FLAGS:
        return "hard", HARD_FLAGS.get(name) or HARD_FLAGS[base]
    if base in CUSTOM:
        return "custom", CUSTOM[base]
    if base in SPECS:
        return "spec", SPECS[base]
    if base in WHITELIST or name in WHITELIST:
        return "white", None
    for p in WHITELIST_PREFIXES:
        if name.startswith(p):
            return "white", None
    for prefix, rule in _ROUTER_RULES:
        if name.startswith(prefix):
            return rule(name)
    return "unknown", None


# scan-time extras for keywords that already have a Spec
SCAN_EXTRA: Dict[str, Callable] = {
    "MAT_PIECEWISE_LINEAR_PLASTICITY": h_mat_024,
    "MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY": h_mat_024,
    "MAT_LOW_DENSITY_FOAM": h_mat_057,
    "PART": x_scan_part,
    "PART_INERTIA": x_scan_part,
    "PART_CONTACT": x_scan_part,
}
# edit-time extras (warnings that need field values)
EDIT_EXTRA: Dict[str, Callable] = {
    "ALE_REFERENCE_SYSTEM_GROUP": x_ale_ref_group,
    "MAT_ELASTIC": x_mat_001,
    "MAT_ELASTIC_FLUID": x_mat_001,
    "MAT_JOHNSON_COOK": x_mat_015,
    "MAT_POWER_LAW_PLASTICITY": x_mat_018,
    "PART_INERTIA": x_part_inertia,
}


def register_keyword(name: str, spec: Optional[Spec] = None, *,
                     alias: object = None,
                     scan_extra: Optional[Callable] = None,
                     edit_extra: Optional[Callable] = None) -> None:
    """One-stop registration hook for a new table-driven keyword.

    Bundles the steps a new keyword normally needs so future additions have a
    single obvious entry point instead of touching four module-level dicts by
    hand:

      * ``spec``       -> installed in ``SPECS`` under ``name`` (the *resolve()*
                          verdict becomes ("spec", spec));
      * ``alias``      -> one numeric/short spelling (str) or several (iterable
                          of str) mapped to ``name`` in ``_MAT_ALIASES`` so
                          e.g. MAT_024 resolves to the canonical MAT_ name;
      * ``scan_extra`` -> a scan-time hook added to ``SCAN_EXTRA`` for ``name``;
      * ``edit_extra`` -> an edit-time hook added to ``EDIT_EXTRA`` for ``name``.

    Every argument except ``name`` is optional, so the helper also serves
    custom-handler-only keywords (pass ``alias``/extras with no ``spec``).
    It only mutates the registry dicts - it does not alter resolve()'s
    precedence, which still consults these dicts in their fixed order.
    """
    if spec is not None:
        SPECS[name] = spec
    if alias is not None:
        aliases = (alias,) if isinstance(alias, str) else tuple(alias)
        for a in aliases:
            _MAT_ALIASES[a] = name
    if scan_extra is not None:
        SCAN_EXTRA[name] = scan_extra
    if edit_extra is not None:
        EDIT_EXTRA[name] = edit_extra


# ── demonstration: two existing materials wired through register_keyword() ────
# These produce SPECS / _MAT_ALIASES entries byte-identical to the inline
# literal forms they replace (verified by comparing registry contents before
# and after the refactor); they show the intended single-hook registration.
register_keyword("MAT_NULL",
                 Spec(cards=[C({1: DENSITY, 2: PRESSURE, 3: VISCOSITY,
                                6: PRESSURE})], probe={"ro": (0, 1)}),
                 alias="MAT_009")
register_keyword("MAT_VACUUM",
                 Spec(cards=[C({1: DENSITY})], probe={"ro": (0, 1)}),
                 alias="MAT_140")
