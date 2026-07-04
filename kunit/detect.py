"""Unit-system auto-detection.

Evidence, strongest first:
  1. material densities (steel ~7850 kg/m^3, aluminium ~2700, water ~1000...)
  2. elastic moduli (steel ~2.1e11 Pa) - fixes the time unit via c=sqrt(E/rho)
  3. detonation velocities (HE burn D ~ 4000-9000 m/s)
  4. gravity-shaped *LOAD_BODY curve ordinates (9.80665 m/s^2)
  5. header comments ('Unit system : m, kg, sec, Pa')
Each candidate system gets a score; the ranked table plus the winning system
are returned. Ambiguity (top two closer than 20%) -> caller should demand an
explicit --from. Includes are followed (tolerantly) so evidence in child
files counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .convert import load_tree, scan
from .parser import STD8, parse_number
from .units import PRESETS, UnitSystem, parse_system

_DENSITY_ANCHORS = [7850.0, 7800.0, 7830.0, 8900.0, 2700.0, 4500.0,
                    1000.0, 2400.0, 1200.0, 7200.0, 1630.0]  # kg/m^3 (1630=TNT)
_MODULUS_ANCHORS = [2.1e11, 2.0e11, 1.93e11, 1.1e11, 7.0e10, 6.9e10,
                    3.0e9, 1.0e9]                            # Pa
_G0 = 9.80665

_HEADER_RE = re.compile(r"\bunit\s*system|\bunits?\s*[:=]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-zµ]+", re.IGNORECASE)
# stamp written by convert(); the deck is now in the *destination* system
_KUNIT_HDR_RE = re.compile(r"\bkunit:\s*converted\s+from\s+\S+\s+to\s+(\S+)",
                           re.IGNORECASE)


@dataclass
class Verdict:
    system: Optional[UnitSystem]
    ranked: List[Tuple[float, UnitSystem]]
    evidence: List[str]
    ambiguous: bool

    def table(self) -> str:
        out = ["score  system"]
        for s, sys in self.ranked[:6]:
            out.append(f"{s:5.1f}  {sys.describe()}")
        return "\n".join(out)


def _near(value: float, anchors, tol: float) -> bool:
    return any(abs(value / a - 1.0) <= tol for a in anchors if a)


def _log_band(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def detect(path: str, follow_includes: bool = True) -> Verdict:
    files, _inc = load_tree(path, follow_includes, strict=False)
    ctx = scan(files, None, {"follow_includes": follow_includes})
    evidence: List[str] = []

    # probe values declared by the schema (density / modulus / det velocity)
    ros: List[float] = []
    es: List[float] = []
    dets: List[float] = []
    from .schema import resolve
    for kf in files:
        for block in kf.blocks:
            kind, payload = resolve(block.name)
            if kind != "spec" or not payload.probe:
                continue
            data = list(block.data)
            opts = block.name.split("_")
            if "TITLE" in opts or "ID" in opts:
                data = data[1:]
            for key, (ci, fi) in payload.probe.items():
                if ci < len(data):
                    v = kf.get_number(data[ci], STD8, block.long, fi)
                    if v:
                        {"ro": ros, "e": es, "d": dets}[key].append(float(v))

    # gravity ordinates from *LOAD_BODY curves
    gravities: List[float] = []
    for lcid in ctx.probes["gravity_lcids"]:
        for ckf, cb in ctx.curve_blocks.get(lcid, []):
            data = list(cb.data)
            if "TITLE" in cb.name.split("_"):
                data = data[1:]
            ords = []
            for li in data[1:]:
                v = ckf.get_number(li, (20, 20), cb.long, 1)
                if v:
                    ords.append(abs(float(v)))
            if ords and max(ords) > 0 and min(ords) / max(ords) > 0.99:
                gravities.append(max(ords))

    # header comment declaration (main file only)
    header_sys: Optional[UnitSystem] = None
    for ln in files[0].lines[:80]:
        if not ln.lstrip().startswith("$"):
            continue
        km = _KUNIT_HDR_RE.search(ln)
        if km:
            try:
                header_sys = parse_system(km.group(1))
                evidence.append(f"kunit header declares {header_sys.key}: "
                                f"{ln.strip()!r}")
            except ValueError:
                pass
            break
        if _HEADER_RE.search(ln):
            toks = [t.lower() for t in _TOKEN_RE.findall(ln)]
            m = next((t for t in toks if t in ("kg", "g", "ton", "tonne", "mg",
                                               "lbm", "lb", "slug", "slinch")), None)
            l = next((t for t in toks if t in ("mm", "cm", "m", "in", "inch",
                                               "ft", "foot")), None)
            t = next((t for t in toks if t in ("s", "sec", "ms", "us", "µs")), None)
            if m and l and t:
                try:
                    header_sys = parse_system(f"{m}-{l}-{t}")
                    evidence.append(f"header comment declares {header_sys.key}: "
                                    f"{ln.strip()!r}")
                except ValueError:
                    pass
            break

    ranked: List[Tuple[float, UnitSystem]] = []
    for sys in PRESETS.values():
        fm, fl, ft = float(sys.mass_kg), float(sys.length_m), float(sys.time_s)
        s = 0.0
        for ro in ros:
            d_si = ro * fm / fl ** 3
            if _near(d_si, _DENSITY_ANCHORS, 0.03):
                s += 5
            elif _log_band(d_si, 700, 25000):
                s += 2
            elif _log_band(d_si, 50, 700) or _log_band(d_si, 25000, 1e5):
                s += 0.3
        for e in es:
            e_si = e * fm / (fl * ft ** 2)
            if _near(e_si, _MODULUS_ANCHORS, 0.05):
                s += 5
            elif _log_band(e_si, 5e8, 5e11):
                s += 2
            elif _log_band(e_si, 1e6, 5e8):
                s += 0.5
        for d in dets:
            d_si = d * fl / ft
            if _log_band(d_si, 3500, 10000):
                s += 3
        for g in gravities:
            g_si = g * fl / ft ** 2
            if abs(g_si / _G0 - 1.0) < 0.02:
                s += 6
            elif abs(g_si / _G0 - 1.0) < 0.2:
                s += 2
        if header_sys is not None and sys == header_sys:
            s += 8
        ranked.append((s, sys))

    ranked.sort(key=lambda kv: -kv[0])
    for ro in ros[:4]:
        evidence.append(f"material density {ro:g}")
    for e in es[:4]:
        evidence.append(f"elastic modulus {e:g}")
    for d in dets[:2]:
        evidence.append(f"detonation velocity {d:g}")
    for g in gravities[:2]:
        evidence.append(f"gravity-curve ordinate {g:g}")
    if len(files) > 1:
        evidence.append(f"evidence gathered across {len(files)} files "
                        "(includes followed)")

    best_s, best = ranked[0]
    second_s = ranked[1][0] if len(ranked) > 1 else 0.0
    ambiguous = best_s <= 0 or (second_s > 0 and second_s / best_s > 0.8)
    return Verdict(system=None if best_s <= 0 else best, ranked=ranked,
                   evidence=evidence, ambiguous=ambiguous)
