"""Unit systems, physical dimensions and exact conversion factors.

A unit system is an (M, L, T) triple. Every dimensional quantity carries a
signature (a, b, c) of exponents, and its conversion factor from system S to
system D is (Ms/Md)^a * (Ls/Ld)^b * (Ts/Td)^c. Unit sizes are stored as exact
`Fraction`s so metric<->metric factors are exact powers of ten.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Dict, Tuple

Dim = Tuple[int, int, int]  # (mass, length, time) exponents

# ── dimension signatures ─────────────────────────────────────────────────────
DIMLESS   : Dim = (0, 0, 0)
MASS      : Dim = (1, 0, 0)
LENGTH    : Dim = (0, 1, 0)
TIME      : Dim = (0, 0, 1)
AREA      : Dim = (0, 2, 0)
VOLUME    : Dim = (0, 3, 0)
L4        : Dim = (0, 4, 0)      # area moment of inertia
VELOCITY  : Dim = (0, 1, -1)
ACCEL     : Dim = (0, 1, -2)
ANG_VEL   : Dim = (0, 0, -1)     # rad/time (rad dimensionless)
ANG_ACCEL : Dim = (0, 0, -2)
RATE      : Dim = (0, 0, -1)     # strain rate, Cowper-Symonds C
FREQ      : Dim = (0, 0, -1)
DENSITY   : Dim = (1, -3, 0)
PRESSURE  : Dim = (1, -1, -2)    # stress, modulus, energy/volume
FORCE     : Dim = (1, 1, -2)
MOMENT    : Dim = (1, 2, -2)
ENERGY    : Dim = (1, 2, -2)
MASS_AREA : Dim = (1, -2, 0)     # mass per unit area
MASS_LEN  : Dim = (1, -1, 0)     # mass per unit length (beam NSM)
INERTIA   : Dim = (1, 2, 0)      # mass moment of inertia
STIFF     : Dim = (1, 0, -2)     # translational stiffness force/length
VISCOSITY : Dim = (1, -1, -1)    # Pa*s; also pressure impulse (stress*time)
DC_FRIC   : Dim = (0, -1, 1)     # contact friction decay coeff (1/velocity)
DAMP      : Dim = (1, 0, -1)     # translational damping force/velocity
ROT_DAMP  : Dim = (1, 2, -1)     # rotational damping moment/(rad/time)
STIFF_LEN : Dim = (1, -2, -2)    # stiffness per length (tiebreak CN, stress/length)
STRESS_M3 : Dim = (-3, 3, 6)     # 1/stress^3 (MAT_022 ALPH nonlinear shear term)
PWR_VOL   : Dim = (1, -1, -3)    # power/volume: EOS energy-deposition rate
                                 # dE/dt, E being energy per reference volume
SPEC_HEAT : Dim = (0, 2, -2)     # specific heat, ASSUMING both systems share
                                 # the same temperature unit (K or degC)
THERM_COND: Dim = (1, 1, -3)     # thermal conductivity W/(m*K), same
                                 # temperature-unit assumption as SPEC_HEAT
# surface tension (force/length) shares STIFF's (1, 0, -2) signature
# power-spectral densities: (base quantity)^2 per frequency = q^2 * time
ACCEL_PSD : Dim = (0, 2, -3)     # acceleration PSD, accel^2/(cycles/time)
VEL_PSD   : Dim = (0, 2, -1)     # velocity PSD
DISP_PSD  : Dim = (0, 2, 1)      # displacement PSD
PRES_PSD  : Dim = (2, -2, -3)    # pressure PSD
FORCE_PSD : Dim = (2, 2, -3)     # force PSD

# Sentinel: temperature values are never rescaled (K<->degC offsets make that
# unsafe); fields marked TEMP are classified and reported, not converted.
TEMP = "TEMP"

DIM_NAMES = {
    DIMLESS: "dimensionless", MASS: "mass", LENGTH: "length", TIME: "time",
    AREA: "area", VOLUME: "volume", L4: "length^4", VELOCITY: "velocity",
    ACCEL: "acceleration", RATE: "1/time", DENSITY: "density",
    PRESSURE: "pressure/stress", FORCE: "force", MOMENT: "moment/energy",
    MASS_AREA: "mass/area", MASS_LEN: "mass/length", INERTIA: "mass inertia",
    STIFF: "stiffness | energy/area | surf.tens.", VISCOSITY: "viscosity (P*t)",
    DC_FRIC: "1/velocity", ANG_ACCEL: "1/time^2", DAMP: "damping F/v",
    ROT_DAMP: "damping M/(rad/t)", STIFF_LEN: "stiffness/length",
    STRESS_M3: "1/stress^3", PWR_VOL: "power/volume",
    SPEC_HEAT: "specific heat (same temp unit)",
    THERM_COND: "thermal conductivity (same temp unit)",
    ACCEL_PSD: "PSD accel^2/freq", VEL_PSD: "PSD vel^2/freq",
    DISP_PSD: "PSD disp^2/freq", PRES_PSD: "PSD pressure^2/freq",
    FORCE_PSD: "PSD force^2/freq",
}

# names accepted by --curve LCID=<x>:<y> overrides
DIM_BY_NAME = {
    "none": DIMLESS, "dimensionless": DIMLESS, "strain": DIMLESS,
    "mass": MASS, "length": LENGTH, "disp": LENGTH, "displacement": LENGTH,
    "time": TIME, "area": AREA, "volume": VOLUME,
    "velocity": VELOCITY, "vel": VELOCITY, "accel": ACCEL,
    "acceleration": ACCEL, "angvel": ANG_VEL, "angaccel": ANG_ACCEL,
    "rate": RATE, "freq": FREQ, "frequency": FREQ, "density": DENSITY,
    "pressure": PRESSURE, "stress": PRESSURE, "modulus": PRESSURE,
    "force": FORCE, "moment": MOMENT, "energy": ENERGY,
    "stiffness": STIFF, "damping": DAMP, "viscosity": VISCOSITY,
    "powervol": PWR_VOL,
    "accelpsd": ACCEL_PSD, "velpsd": VEL_PSD, "disppsd": DISP_PSD,
    "prespsd": PRES_PSD, "forcepsd": FORCE_PSD,
}


def parse_dim_name(name: str) -> Dim:
    try:
        return DIM_BY_NAME[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown dimension name {name!r} (choose from: "
            f"{', '.join(sorted(DIM_BY_NAME))})") from None

# ── base unit sizes (exact) ──────────────────────────────────────────────────
_LB = Fraction(45359237, 10**8)          # lbm in kg (exact by definition)
_G0 = Fraction(980665, 10**5)            # standard gravity m/s^2 (exact)
_FT = Fraction(3048, 10**4)              # ft in m
_IN = Fraction(254, 10**4)               # in in m

MASS_UNITS: Dict[str, Fraction] = {      # size in kg
    "kg": Fraction(1), "g": Fraction(1, 1000), "ton": Fraction(1000),
    "lb": _LB, "slug": _LB * _G0 / _FT, "slinch": _LB * _G0 / _IN,
}
LENGTH_UNITS: Dict[str, Fraction] = {    # size in m
    "m": Fraction(1), "mm": Fraction(1, 1000), "cm": Fraction(1, 100),
    "in": _IN, "ft": _FT,
}
TIME_UNITS: Dict[str, Fraction] = {      # size in s
    "s": Fraction(1), "ms": Fraction(1, 1000), "us": Fraction(1, 10**6),
}

_MASS_ALIASES = {"kg": "kg", "kilogram": "kg", "g": "g", "gram": "g",
                 "ton": "ton", "tonne": "ton", "t": "ton", "mg": "ton",
                 "megagram": "ton", "metric_ton": "ton",
                 "lb": "lb", "lbm": "lb", "pound": "lb",
                 "slug": "slug", "slinch": "slinch", "blob": "slinch",
                 "lbf-s2/in": "slinch", "lbfs2/in": "slinch"}
_LEN_ALIASES = {"m": "m", "meter": "m", "metre": "m", "mm": "mm",
                "millimeter": "mm", "millimetre": "mm", "cm": "cm",
                "centimeter": "cm", "centimetre": "cm",
                "in": "in", "inch": "in", "ft": "ft", "foot": "ft"}
_TIME_ALIASES = {"s": "s", "sec": "s", "second": "s", "ms": "ms",
                 "millisecond": "ms", "us": "us", "µs": "us",
                 "microsecond": "us"}


@dataclass(frozen=True)
class UnitSystem:
    mass: str      # canonical unit names
    length: str
    time: str

    @property
    def mass_kg(self) -> Fraction:   return MASS_UNITS[self.mass]
    @property
    def length_m(self) -> Fraction:  return LENGTH_UNITS[self.length]
    @property
    def time_s(self) -> Fraction:    return TIME_UNITS[self.time]
    @property
    def pressure_pa(self) -> Fraction:
        return self.mass_kg / (self.length_m * self.time_s ** 2)
    @property
    def force_n(self) -> Fraction:
        return self.mass_kg * self.length_m / self.time_s ** 2

    @property
    def key(self) -> str:
        return f"{self.mass}-{self.length}-{self.time}"

    def describe(self) -> str:
        p = float(self.pressure_pa)
        f = float(self.force_n)
        p_lbl = {1.0: "Pa", 1e6: "MPa", 1e9: "GPa", 1e11: "Mbar",
                 1e3: "kPa"}.get(round(p, 6), f"{p:g} Pa")
        f_lbl = {1.0: "N", 1e3: "kN", 1e6: "MN", 1e-3: "mN",
                 1e-6: "uN"}.get(round(f, 9), f"{f:g} N")
        try:
            if abs(p / 6894.757293168361 - 1) < 1e-9:
                p_lbl = "psi"
            if abs(f / 4.4482216152605 - 1) < 1e-9:
                f_lbl = "lbf"
        except ZeroDivisionError:
            pass
        return (f"{self.key}  (pressure = {p_lbl}, force = {f_lbl})")


PRESETS: Dict[str, UnitSystem] = {
    s.key: s for s in (
        UnitSystem("kg", "m", "s"),
        UnitSystem("ton", "mm", "s"),
        UnitSystem("kg", "mm", "ms"),
        UnitSystem("g", "mm", "ms"),
        UnitSystem("g", "cm", "us"),
        UnitSystem("g", "cm", "ms"),
        UnitSystem("kg", "cm", "ms"),
        UnitSystem("kg", "cm", "s"),
        UnitSystem("kg", "mm", "s"),
        UnitSystem("slinch", "in", "s"),
        UnitSystem("slug", "ft", "s"),
        UnitSystem("lb", "ft", "s"),
    )
}


def parse_system(spec: str) -> UnitSystem:
    """Parse 'ton-mm-s', 'Mg,mm,s', 'kg m s' ... into a UnitSystem."""
    tokens = [t for t in spec.replace(",", "-").replace(" ", "-").split("-") if t]
    if len(tokens) != 3:
        raise ValueError(f"unit system {spec!r}: expected MASS-LENGTH-TIME")
    m, l, t = (tok.strip().lower() for tok in tokens)
    try:
        return UnitSystem(_MASS_ALIASES[m], _LEN_ALIASES[l], _TIME_ALIASES[t])
    except KeyError as e:
        raise ValueError(f"unit system {spec!r}: unknown unit {e.args[0]!r} "
                         f"(mass: {sorted(set(_MASS_ALIASES))}; "
                         f"length: {sorted(set(_LEN_ALIASES))}; "
                         f"time: {sorted(set(_TIME_ALIASES))})") from None


def factor(dim: Dim, src: UnitSystem, dst: UnitSystem) -> Fraction:
    """Exact multiplier converting a value of dimension `dim` from src to dst."""
    a, b, c = dim
    return ((src.mass_kg / dst.mass_kg) ** a
            * (src.length_m / dst.length_m) ** b
            * (src.time_s / dst.time_s) ** c)


def apply_factor(value: Decimal, f: Fraction) -> Decimal:
    """value * f at high precision (exact for power-of-ten factors)."""
    if f == 1:
        return value
    with localcontext() as ctx:
        ctx.prec = 34
        return value * Decimal(f.numerator) / Decimal(f.denominator)


# ── *LOAD_BLAST_ENHANCED UNIT flag support ───────────────────────────────────
# Built-in UNIT values (LS-DYNA R16 manual, *LOAD_BLAST_ENHANCED):
BLAST_BUILTIN_UNITS: Dict[Tuple[str, str, str], int] = {
    ("lb", "ft", "s"): 1,
    ("kg", "m", "s"): 2,
    ("slinch", "in", "s"): 3,
    ("g", "cm", "us"): 4,
    ("kg", "mm", "ms"): 6,
    ("ton", "mm", "s"): 7,
    ("g", "mm", "ms"): 8,
}
BLAST_UNIT_SYSTEMS = {v: UnitSystem(*k) for k, v in BLAST_BUILTIN_UNITS.items()}


# ── *MAT_CSCM_CONCRETE (MAT_159) UNITS flag support ──────────────────────────
# UNITS values from the R16 manual Vol II p.2-1084 (*MAT_159 Card 3); the flag
# declares the unit system that FPC (pressure) and DAGG (length) are given in:
#   EQ.0: GPa, mm, msec, kg/mm3, kN      EQ.1: MPa, mm, msec, g/mm3, N
#   EQ.2: MPa, mm, sec, Mg/mm3, N        EQ.3: Psi, inch, sec, lbf-s2/in4, lbf
#   EQ.4: Pa, m, sec, kg/m3, N
CSCM_UNITS: Dict[Tuple[str, str, str], int] = {
    ("kg", "mm", "ms"): 0,
    ("g", "mm", "ms"): 1,
    ("ton", "mm", "s"): 2,
    ("slinch", "in", "s"): 3,
    ("kg", "m", "s"): 4,
}
CSCM_UNIT_SYSTEMS = {v: UnitSystem(*k) for k, v in CSCM_UNITS.items()}


def blast_unit5_factors(sys: UnitSystem) -> Tuple[float, float, float, float]:
    """CFM/CFL/CFT/CFP for UNIT=5: model unit -> ConWep's lbm/ft/ms/psi.

    Per the R16 manual (Vol I p.33-18): CFM = pounds per mass unit, CFL =
    feet per length unit, CFT = MILLIseconds per time unit, CFP = psi per
    pressure unit.
    """
    psi_pa = _LB * _G0 / _IN ** 2
    return (float(sys.mass_kg / _LB),
            float(sys.length_m / _FT),
            float(sys.time_s / Fraction(1, 1000)),
            float(sys.pressure_pa / psi_pa))
