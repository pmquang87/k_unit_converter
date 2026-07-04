import os
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import ConvertError, convert, detect, factor, parse_system
from kunit.parser import KFile, parse_number
from kunit.units import (ACCEL, DENSITY, LENGTH, MASS, PRESSURE, TIME,
                         VELOCITY, blast_unit5_factors)

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")


def F(*vals, w=10):
    """Fixed-width card line from values."""
    return "".join(str(v).rjust(w) for v in vals)


DECK_SI = """$ Unit system : m, kg, sec, Pa
*KEYWORD
*TITLE
synthetic test deck
*CONTROL_TERMINATION
     0.006         0       0.0       0.01.000000E8         0
*DATABASE_GLSTAT
2.00000E-5         0         0         1
*SECTION_SHELL
         1         2       1.0         2         1         0         0         1
      0.05      0.05      0.05      0.05       0.0       0.0       0.0         0
*MAT_PLASTIC_KINEMATIC
         1    7850.02.10000E11       0.31.200000E91.10000E10       0.0
       0.0       0.0    0.0015       0.0
*MAT_RIGID_TITLE
rigid
         2    7850.02.10000E11       0.3       0.0       0.0       0.0
       0.0         0         0         0       0.0       0.0       0.0
       0.0       0.0       0.0       0.0       0.0       0.0         0
*PART
plate
         1         1         1         0         0         0         0         0
*LOAD_BODY_Z
         1      -1.0         0       0.0       0.0       0.0         0
*DEFINE_CURVE_TITLE
Weight
         1         0       1.0       1.0       0.0       0.0         0         0
                 0.0                9.81
                 1.0                9.81
*LOAD_BLAST_ENHANCED
         1      50.0       2.5       0.0       0.1       0.0         2         1
       0.0       0.0       0.0       0.0         01.00000E20         0
*ELEMENT_MASS
       1     101            50.0       0
       2     102             5.5       0
*NODE
       1             1.7        1.654231       0.2799996       0       0
       2            -1.6             0.0      7.85000-2       0       0
*END
"""


def _write(text, name="deck.k", d=None):
    d = d or tempfile.mkdtemp(prefix="kunit_test_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _lines(path):
    with open(path, newline="") as fh:
        return fh.read().split("\n")


class UnitsTests(unittest.TestCase):
    def test_exact_metric_factors(self):
        self.assertEqual(factor(LENGTH, SI, TON), Fraction(1000))
        self.assertEqual(factor(MASS, SI, TON), Fraction(1, 1000))
        self.assertEqual(factor(DENSITY, SI, TON), Fraction(1, 10**12))
        self.assertEqual(factor(PRESSURE, SI, TON), Fraction(1, 10**6))
        self.assertEqual(factor(TIME, SI, TON), Fraction(1))
        self.assertEqual(factor((1, 1, -2), SI, TON), Fraction(1))  # force N

    def test_blast_unit5_factors(self):
        cfm, cfl, cft, cfp = blast_unit5_factors(TON)
        self.assertAlmostEqual(cfm, 2204.62262, places=4)
        self.assertAlmostEqual(cfl, 3.28084e-3, places=8)
        self.assertAlmostEqual(cft, 1000.0)
        self.assertAlmostEqual(cfp, 145.0377377, places=4)

    def test_parse_aliases(self):
        self.assertEqual(parse_system("Mg,mm,s"), TON)
        self.assertEqual(parse_system("tonne mm sec"), TON)


class ParserTests(unittest.TestCase):
    def test_eless_exponent(self):
        self.assertEqual(parse_number("7.85000-9"), Decimal("7.85000E-9"))
        self.assertEqual(parse_number(" 2.1E+11 "), Decimal("2.1E+11"))
        self.assertIsNone(parse_number("   "))


class DetectTests(unittest.TestCase):
    def test_detect_si(self):
        v = detect(_write(DECK_SI))
        self.assertFalse(v.ambiguous)
        self.assertEqual(v.system, SI)

    def test_detect_ton_after_convert(self):
        p = _write(DECK_SI)
        out = p + ".ton.k"
        convert(p, SI, TON, out, self_check=False)
        v = detect(out)
        self.assertEqual(v.system, TON)

    # weak-evidence deck: density off every anchor (+2), modulus near one
    # (+5) - total 7, so a wrongly credited header bonus (+8) flips the
    # verdict.  Guards the '$ kunit: converted from X to Y' stamp parsing.
    DECK_WEAK_SI = """*KEYWORD
*MAT_ELASTIC
         1    2580.07.20000E10      0.33
*NODE
       1             1.7        1.654231       0.2799996       0       0
*END
"""

    def test_kunit_stamp_declares_destination(self):
        p = _write(self.DECK_WEAK_SI)
        out = p + ".ton.k"
        convert(p, SI, TON, out, self_check=False)
        v = detect(out)
        self.assertEqual(v.system, TON)
        self.assertTrue(any("kunit header declares ton-mm-s" in e
                            for e in v.evidence), v.evidence)

    def test_self_check_ok_with_weak_evidence(self):
        p = _write(self.DECK_WEAK_SI)
        out = p + ".ton.k"
        ctx = convert(p, SI, TON, out, self_check=True)
        self.assertTrue(ctx.self_check.startswith("OK"), ctx.self_check)
        self.assertEqual(ctx.warnings, [])


class ConvertTests(unittest.TestCase):
    def _conv(self, text=DECK_SI, **kw):
        p = _write(text)
        out = p + ".out.k"
        kw.setdefault("self_check", False)
        ctx = convert(p, SI, TON, out, **kw)
        return _lines(out), ctx

    def test_node_and_mass_scaling(self):
        lines, _ = self._conv()
        kf_i = lines.index("*NODE")
        self.assertAlmostEqual(float(lines[kf_i + 1][8:24]), 1700.0)
        self.assertAlmostEqual(float(lines[kf_i + 1][24:40]), 1654.231)
        self.assertAlmostEqual(float(lines[kf_i + 2][40:56]), 78.5, places=4)
        em_i = lines.index("*ELEMENT_MASS")
        self.assertAlmostEqual(float(lines[em_i + 1][16:32]), 0.05)
        self.assertAlmostEqual(float(lines[em_i + 2][16:32]), 0.0055)
        self.assertEqual(lines[em_i + 1][16:32].rstrip(),
                         lines[em_i + 1][16:30])

    def test_material_and_section(self):
        lines, _ = self._conv()
        mi = lines.index("*MAT_PLASTIC_KINEMATIC")
        card = lines[mi + 1]
        self.assertAlmostEqual(float(card[10:20]), 7.85e-9)
        self.assertAlmostEqual(float(card[20:30]), 2.1e5)
        self.assertAlmostEqual(float(card[40:50]), 1200.0)
        si = lines.index("*SECTION_SHELL")
        self.assertAlmostEqual(float(lines[si + 2][0:10]), 50.0)

    def test_curve_scaled_as_acceleration(self):
        lines, _ = self._conv()
        ci = lines.index("Weight")
        self.assertAlmostEqual(float(lines[ci + 2][20:40]), 9810.0)
        self.assertAlmostEqual(float(lines[ci + 3][0:20]), 1.0)

    def test_blast_builtin_unit7(self):
        lines, _ = self._conv()
        bi = lines.index("*LOAD_BLAST_ENHANCED")
        c1 = lines[bi + 1]
        self.assertAlmostEqual(float(c1[10:20]), 0.05)
        self.assertAlmostEqual(float(c1[20:30]), 2500.0)
        self.assertEqual(int(c1[60:70]), 7)

    def test_blast_forced_unit5(self):
        lines, _ = self._conv(blast_unit=5)
        bi = lines.index("*LOAD_BLAST_ENHANCED")
        self.assertEqual(int(lines[bi + 1][60:70]), 5)
        c2 = lines[bi + 2]
        self.assertAlmostEqual(float(c2[0:10]), 2204.62262, places=3)
        self.assertAlmostEqual(float(c2[20:30]), 1000.0)

    def test_blast_unit5_card2_rounds_to_field_width(self):
        # CFL for ton-mm-s is 1 mm in ft = 1/304.8 = 0.0032808399... (R16
        # Vol I p.33-18: CFL = feet per length unit).  The 10-char field must
        # hold the correctly ROUNDED shortest form, 0.00328084 - not
        # 0.00328083, the truncation of the 9-significant-digit form.
        lines, _ = self._conv(blast_unit=5)
        bi = lines.index("*LOAD_BLAST_ENHANCED")
        c2 = lines[bi + 2]
        self.assertEqual(c2[10:20], "0.00328084")
        self.assertEqual(c2[0:10], "2204.62262")
        self.assertEqual(c2[30:40], "145.037738")

    def test_round_trip(self):
        p = _write(DECK_SI)
        mid = p + ".ton.k"
        back = p + ".back.k"
        convert(p, SI, TON, mid, self_check=False)
        convert(mid, TON, SI, back, self_check=False)
        kf_a, kf_b = KFile(p), KFile(back)
        self.assertEqual(len(kf_a.lines), len(kf_b.lines) - 4)
        na = kf_a.lines[kf_a.lines.index("*NODE") + 1]
        nb = kf_b.lines[kf_b.lines.index("*NODE") + 1]
        for sl in (slice(8, 24), slice(24, 40), slice(40, 56)):
            self.assertAlmostEqual(float(na[sl]), float(nb[sl]), places=6)

    def test_unknown_keyword_refused(self):
        deck = DECK_SI.replace("*END", "*AIRBAG_SIMPLE_PRESSURE_VOLUME\n"
                               "         1\n*END")
        p = _write(deck)
        with self.assertRaises(ConvertError):
            convert(p, SI, TON, p + ".o.k", self_check=False)
        ctx = convert(p, SI, TON, p + ".o.k", allow_unknown=True,
                      self_check=False)
        self.assertIn("AIRBAG_SIMPLE_PRESSURE_VOLUME", ctx.unknown)

    def test_parameter_field_refused(self):
        deck = DECK_SI.replace(
            "         1    7850.02.10000E11",
            "         1    &dens 2.10000E11")
        p = _write(deck)
        with self.assertRaises(Exception):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_free_format_line(self):
        deck = DECK_SI.replace(
            "       1             1.7        1.654231       0.2799996       0       0",
            "1, 1.7, 1.654231, 0.2799996, 0, 0")
        lines, _ = self._conv(deck)
        ni = lines.index("*NODE")
        toks = lines[ni + 1].split(",")
        self.assertAlmostEqual(float(toks[1]), 1700.0)
        self.assertAlmostEqual(float(toks[3]), 279.9996)

    def test_eos_gruneisen_card2(self):
        # R16 Vol II p.1-15: Card2 is V0 (dimensionless), blank, LCID
        # (energy-deposition rate dE/dt vs time, power/volume ordinates)
        deck = """*KEYWORD
*EOS_GRUNEISEN_TITLE
water eos
         1    1480.0      1.97       0.0       0.0      0.11       0.03.00000E+5
       0.0                   7
*EOS_GRUNEISEN
         2     150.0      1.97       0.0       0.0       0.0       0.0       0.0
       0.0
*DEFINE_CURVE
         7         0       1.0       1.0       0.0       0.0         0         0
                 0.0          1.00000E6
                 1.0          1.00000E6
*END
"""
        lines, ctx = self._conv(deck)
        self.assertFalse(any("EOS_GRUNEISEN" in w for w in ctx.warnings),
                         ctx.warnings)
        ei = lines.index("water eos")
        c1 = lines[ei + 1]
        self.assertAlmostEqual(float(c1[10:20]), 1.48e6)     # C x1000
        self.assertAlmostEqual(float(c1[70:80]), 0.3)        # E0 x1e-6
        c2 = lines[ei + 2]
        self.assertAlmostEqual(float(c2[0:10]), 0.0)         # V0 unscaled
        self.assertEqual(int(c2[20:30]), 7)                  # LCID untouched
        # V0-only Card2 (no LCID) must pass silently too
        e2 = lines.index("*EOS_GRUNEISEN")
        self.assertAlmostEqual(float(lines[e2 + 1][10:20]), 1.5e5)
        self.assertAlmostEqual(float(lines[e2 + 2][0:10]), 0.0)
        # curve 7 ordinates carry dE/dt: x1e-6 for kg-m-s -> ton-mm-s
        di = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[di + 2][0:20]), 0.0)
        self.assertAlmostEqual(float(lines[di + 2][20:40]), 1.0)
        self.assertAlmostEqual(float(lines[di + 3][0:20]), 1.0)


class NewFeatureTests(unittest.TestCase):
    def test_self_check_and_roundtrip(self):
        p = _write(DECK_SI)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, verify_roundtrip=True)
        self.assertTrue(ctx.self_check.startswith("OK"), ctx.self_check)
        self.assertTrue(ctx.roundtrip.startswith("OK"), ctx.roundtrip)

    def test_dry_run_writes_nothing(self):
        p = _write(DECK_SI)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, dry_run=True, self_check=False)
        self.assertFalse(os.path.exists(out))
        self.assertIn("DRY RUN - no files were written", ctx.notes)

    def test_curve_override(self):
        deck = ("*KEYWORD\n*DEFINE_CURVE\n"
                + F(5, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", w=20) + F("9.81", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)     # unreferenced
        li = _lines(out)
        self.assertAlmostEqual(float(li[li.index("*DEFINE_CURVE") + 2][20:40]),
                               9.81)
        self.assertTrue(any("lcid=5" in w for w in ctx.warnings))
        out2 = p + ".o2.k"
        convert(p, SI, TON, out2, self_check=False,
                curve_overrides={5: (TIME, ACCEL)})
        li = _lines(out2)
        self.assertAlmostEqual(float(li[li.index("*DEFINE_CURVE") + 2][20:40]),
                               9810.0)

    def test_table_subcurves_following_form(self):
        deck = ("*KEYWORD\n*MAT_PIECEWISE_LINEAR_PLASTICITY\n"
                + F(1, 7850.0, "2.1E11", 0.3, "3.5E8", 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 100, 0, 0) + "\n"
                "*DEFINE_TABLE\n" + F(100, 1.0, 0.0) + "\n"
                + F("0.0", w=20) + "\n" + F("100.0", w=20) + "\n"
                "*DEFINE_CURVE\n" + F(101, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", w=20) + F("3.5E8", w=20) + "\n"
                "*DEFINE_CURVE\n" + F(102, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", w=20) + F("4.5E8", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        c1 = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[c1 + 2][20:40]), 350.0)  # stress
        self.assertAlmostEqual(float(lines[c1 + 5][20:40]), 450.0)
        self.assertFalse(any("unreferenced" in w for w in ctx.warnings))
        # rate values: SI->kg-mm-ms scales 1/time by 1e-3
        out2 = p + ".o2.k"
        convert(p, SI, parse_system("kg-mm-ms"), out2, self_check=False)
        ti = _lines(out2).index("*DEFINE_TABLE")
        self.assertAlmostEqual(float(_lines(out2)[ti + 3][0:20]), 0.1)

    def test_table_pair_form(self):
        deck = ("*KEYWORD\n*MAT_PIECEWISE_LINEAR_PLASTICITY\n"
                + F(1, 7850.0, "2.1E11", 0.3, "3.5E8", 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 100, 0, 0) + "\n"
                "*DEFINE_TABLE\n" + F(100, 1.0, 0.0) + "\n"
                + F("0.0", w=20) + F(101, w=20) + "\n"
                "*SET_NODE_LIST\n" + F(9) + "\n" + F(1, 2) + "\n"
                "*DEFINE_CURVE\n" + F(101, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", w=20) + F("3.5E8", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        c1 = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[c1 + 2][20:40]), 350.0)

    def test_johnson_cook(self):
        deck = ("*KEYWORD\n*MAT_JOHNSON_COOK\n"
                + F(1, 7850.0, "8.0E10", "2.1E11", 0.29, 0.0, 0.0, 0.0) + "\n"
                + F("3.5E8", "2.75E8", 0.36, 0.022, 1.09, 1793.0, 293.0, 1.0) + "\n"
                + F(477.0, "-1.0E9", 2.0, 0, 0.05, 3.44, -2.12, 0.002) + "\n"
                + F(0.61, 0, 0.0, 0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        mi = lines.index("*MAT_JOHNSON_COOK")
        self.assertAlmostEqual(float(lines[mi + 1][10:20]), 7.85e-9)
        self.assertAlmostEqual(float(lines[mi + 1][20:30]), 8.0e4)   # G
        self.assertAlmostEqual(float(lines[mi + 2][0:10]), 350.0)    # A
        self.assertAlmostEqual(float(lines[mi + 2][50:60]), 1793.0)  # TM
        self.assertAlmostEqual(float(lines[mi + 2][60:70]), 293.0)   # TR
        self.assertAlmostEqual(float(lines[mi + 3][0:10]), 4.77e8)   # CP
        self.assertAlmostEqual(float(lines[mi + 3][10:20]), -1000.0) # PC
        self.assertTrue(any("temperature" in n for n in ctx.notes))

    def test_discrete_spring_dro(self):
        deck = ("*KEYWORD\n"
                "*SECTION_DISCRETE\n" + F(100, 1, 0.0, 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0) + "\n"
                "*SECTION_DISCRETE\n" + F(200, 0, 0.0, 2.0, 0.01, 0.0) + "\n"
                + F(0.0, 0.0) + "\n"
                "*PART\ntorsional\n" + F(10, 100, 1, 0, 0, 0, 0, 0) + "\n"
                "*PART\ntranslational\n" + F(11, 200, 2, 0, 0, 0, 0, 0) + "\n"
                "*MAT_SPRING_ELASTIC\n" + F(1, 1000.0) + "\n"
                "*MAT_SPRING_ELASTIC\n" + F(2, 1000.0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        mats = [i for i, ln in enumerate(lines)
                if ln == "*MAT_SPRING_ELASTIC"]
        # torsional: N*m/rad -> N*mm/rad (x1e3); translational: N/m -> N/mm (x1e-3)
        self.assertAlmostEqual(float(lines[mats[0] + 1][10:20]), 1.0e6)
        self.assertAlmostEqual(float(lines[mats[1] + 1][10:20]), 1.0)
        # translational section: V0 & CL scaled; torsional untouched
        secs = [i for i, ln in enumerate(lines) if ln == "*SECTION_DISCRETE"]
        self.assertAlmostEqual(float(lines[secs[1] + 1][30:40]), 2000.0)
        self.assertAlmostEqual(float(lines[secs[1] + 1][40:50]), 10.0)

    def test_rigidwall_planar_moving(self):
        deck = ("*KEYWORD\n*RIGIDWALL_PLANAR_MOVING\n"
                + F(0, 0, 0, 0.1, 0.0, "1.0E20", 1.0) + "\n"
                + F(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 5.0) + "\n"
                + F(100.0, 10.0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        ri = lines.index("*RIGIDWALL_PLANAR_MOVING")
        self.assertAlmostEqual(float(lines[ri + 1][30:40]), 100.0)   # offset
        self.assertAlmostEqual(float(lines[ri + 2][20:30]), 1000.0)  # zt
        self.assertAlmostEqual(float(lines[ri + 2][70:80]), 5000.0)  # wvel
        self.assertAlmostEqual(float(lines[ri + 3][0:10]), 0.1)      # mass
        self.assertAlmostEqual(float(lines[ri + 3][10:20]), 10000.0) # v0

    def test_contact_tiebreak(self):
        deck = ("*KEYWORD\n*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_TIEBREAK\n"
                + F(1, 2, 3, 3, 0, 0, 0, 0) + "\n"
                + F(0.2, 0.2, 0.0, 0.0, 20.0, 0, 0.0, "1.0E20") + "\n"
                + F(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0) + "\n"
                + F(6, "4.0E8", "4.0E8", 0.001, 0.0, 0.0, 1.0, 0.0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        c4 = lines.index("*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_TIEBREAK") + 4
        self.assertAlmostEqual(float(lines[c4][10:20]), 400.0)  # NFLS MPa
        self.assertTrue(any("TIEBREAK PARAM" in w for w in ctx.warnings))

    def test_initial_stress_solid(self):
        deck = ("*KEYWORD\n*INITIAL_STRESS_SOLID\n"
                + F(1, 1, 0, 0, 0, 0, 0, 0) + "\n"
                + F("-1.0E6", "-1.0E6", "-1.0E6", 0.0, 0.0, 0.0, 0.0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        ii = lines.index("*INITIAL_STRESS_SOLID")
        self.assertAlmostEqual(float(lines[ii + 2][0:10]), -1.0)
        bad = deck.replace(F(1, 1, 0, 0, 0, 0, 0, 0), F(1, 1, 2, 0, 0, 0, 0, 0))
        with self.assertRaises(ConvertError):
            convert(_write(bad), SI, TON, out + "2", self_check=False)

    def test_curve_smooth(self):
        deck = ("*KEYWORD\n*DEFINE_CURVE_SMOOTH\n"
                + F(7, 0, 0.5, 0.0, 0.1, 0.01, 5.0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        li = _lines(out)
        ln = li[li.index("*DEFINE_CURVE_SMOOTH") + 1]
        self.assertAlmostEqual(float(ln[20:30]), 500.0)   # DIST
        self.assertAlmostEqual(float(ln[40:50]), 0.1)     # TEND unchanged
        self.assertAlmostEqual(float(ln[60:70]), 5000.0)  # VMAX

    def test_includes_refused_without_flag(self):
        d = tempfile.mkdtemp(prefix="kunit_inc_")
        _write("*KEYWORD\n*NODE\n"
               "       1             1.0             0.0             0.0\n"
               "*END\n", "child.k", d)
        p = _write("*KEYWORD\n*INCLUDE\nchild.k\n*END\n", "parent.k", d)
        with self.assertRaises(ConvertError):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_includes_followed(self):
        d = tempfile.mkdtemp(prefix="kunit_inc_")
        _write("*KEYWORD\n*NODE\n"
               "       1             1.0             0.0             0.0\n"
               "*END\n", "child.k", d)
        p = _write("*KEYWORD\n*INCLUDE\nchild.k\n"
                   "*CONTROL_TERMINATION\n" + F(0.01) + "\n*END\n",
                   "parent.k", d)
        out = os.path.join(d, "parent_ton.k")
        ctx = convert(p, SI, TON, out, follow_includes=True, self_check=False)
        child_out = os.path.join(d, "child__ton-mm-s.k")
        self.assertTrue(os.path.exists(child_out))
        self.assertIn("child__ton-mm-s.k",
                      "\n".join(_lines(out)))
        ci = _lines(child_out)
        ni = ci.index("*NODE")
        self.assertAlmostEqual(float(ci[ni + 1][8:24]), 1000.0)
        self.assertEqual(len(ctx.written), 2)

    def test_includes_in_place_with_backup(self):
        d = tempfile.mkdtemp(prefix="kunit_inc_")
        c = _write("*KEYWORD\n*NODE\n"
                   "       1             1.0             0.0             0.0\n"
                   "*END\n", "child.k", d)
        p = _write("*KEYWORD\n*INCLUDE\nchild.k\n*END\n", "parent.k", d)
        convert(p, SI, TON, p, follow_includes=True, self_check=False)
        self.assertTrue(os.path.exists(p + ".orig_kg-m-s"))
        self.assertTrue(os.path.exists(c + ".orig_kg-m-s"))
        ci = _lines(c)
        self.assertAlmostEqual(float(ci[ci.index("*NODE") + 1][8:24]), 1000.0)
        # include reference unchanged for in-place conversion
        self.assertIn("child.k", "\n".join(_lines(p)))

    def test_keyword_line_trailing_text(self):
        deck = ("*KEYWORD MEMORY=800000000 NCPU=4\n"
                "*ELEMENT_SOLID (TEN NODES FORMAT)\n"
                + F(1, 1, w=8) + "\n"
                + F(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, w=8) + "\n"
                "*NODE\n"
                "       1             1.0             0.0             0.0\n"
                "*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        self.assertEqual(ctx.unknown, {})
        li = _lines(out)
        ni = [i for i, ln in enumerate(li) if ln.startswith("*NODE")][0]
        self.assertAlmostEqual(float(li[ni + 1][8:24]), 1000.0)
        # ten-node connectivity untouched
        self.assertIn(F(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, w=8), li)

    def test_initial_velocity_node(self):
        deck = ("*KEYWORD\n*INITIAL_VELOCITY_NODE\n"
                + F(7, 10.0, 0.0, -5.0, 0.0, 2.0, 0.0, 0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        li = _lines(out)
        ln = li[li.index("*INITIAL_VELOCITY_NODE") + 1]
        self.assertAlmostEqual(float(ln[10:20]), 10000.0)  # vx mm/s
        self.assertAlmostEqual(float(ln[30:40]), -5000.0)  # vz
        self.assertAlmostEqual(float(ln[50:60]), 2.0)      # vyr rad/s unchanged

    def test_mat_composite_damage(self):
        deck = ("*KEYWORD\n*MAT_COMPOSITE_DAMAGE_TITLE\nGrEP\n"
                + F(1, 1450.0, "1.3E11", "4.5E10", "4.5E10", 0.15, 0.15, 0.25) + "\n"
                + F("8.0E9", "8.0E9", "8.0E9", "2.0E9", 2.0, 1, 0) + "\n"
                + F(1.5, -2.5, 0.0, 1.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0) + "\n"
                + F("1.2E8", "2.5E9", "2.5E8", "2.5E7", "1.0E-27", "3.0E7",
                    "2.0E7", "2.0E7") + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        c1 = lines[lines.index("GrEP") + 1]
        self.assertAlmostEqual(float(c1[10:20]), 1.45e-9)   # RO
        self.assertAlmostEqual(float(c1[20:30]), 1.3e5)     # EA
        self.assertAlmostEqual(float(c1[40:50]), 4.5e4)     # EC
        self.assertAlmostEqual(float(c1[50:60]), 0.15)      # PRBA unchanged
        c2 = lines[lines.index("GrEP") + 2]
        self.assertAlmostEqual(float(c2[0:10]), 8000.0)     # GAB
        self.assertAlmostEqual(float(c2[30:40]), 2000.0)    # KFAIL
        self.assertAlmostEqual(float(c2[40:50]), 2.0)       # AOPT unchanged
        c3 = lines[lines.index("GrEP") + 3]
        self.assertAlmostEqual(float(c3[0:10]), 1500.0)     # XP
        self.assertAlmostEqual(float(c3[10:20]), -2500.0)   # YP
        self.assertAlmostEqual(float(c3[30:40]), 1.0)       # A1 unchanged
        c4 = lines[lines.index("GrEP") + 4]
        self.assertAlmostEqual(float(c4[60:70]), 45.0)      # BETA unchanged
        c5 = lines[lines.index("GrEP") + 5]
        self.assertAlmostEqual(float(c5[0:10]), 120.0)      # SC
        self.assertAlmostEqual(float(c5[10:20]), 2500.0)    # XT
        self.assertAlmostEqual(float(c5[40:50]), 1.0e-9)    # ALPH stress^-3
        self.assertAlmostEqual(float(c5[50:60]), 30.0)      # SN

    def test_mat_composite_damage_probe(self):
        deck = ("*KEYWORD\n*MAT_022\n"
                + F(1, 7850.0, "2.1E11", "2.1E11", "2.1E11", 0.3, 0.3, 0.3) + "\n"
                + F("8.0E10", "8.0E10", "8.0E10", 0.0, 0.0, 0, 0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + "\n"
                + F("1.2E8", "2.5E9", "2.5E8", "2.5E7", 0.0, 0.0, 0.0, 0.0)
                + "\n*END\n")
        v = detect(_write(deck))
        self.assertEqual(v.system, SI)
        self.assertTrue(any("material density 7850" in e for e in v.evidence),
                        v.evidence)

    def test_part_composite(self):
        deck = ("*KEYWORD\n*PART_COMPOSITE\nTopSheet\n"
                + F(12, 2, 0.0, 0.0, 1.5, 0, 0, 0) + "\n"
                + F(1, 0.001, 0.0, 0, 1, 0.001, 90.0, 0) + "\n"
                + F(1, 0.001, 45.0, 0, 1, 0.001, -45.0, 0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        pi = lines.index("*PART_COMPOSITE")
        hdr = lines[pi + 2]
        self.assertEqual(int(hdr[0:10]), 12)                  # PID untouched
        self.assertAlmostEqual(float(hdr[40:50]), 1.5e-9)     # MAREA mass/area
        ply1, ply2 = lines[pi + 3], lines[pi + 4]
        self.assertEqual(int(ply1[0:10]), 1)                  # MID1 untouched
        self.assertAlmostEqual(float(ply1[10:20]), 1.0)       # THICK1 -> mm
        self.assertAlmostEqual(float(ply1[50:60]), 1.0)       # THICK2 -> mm
        self.assertAlmostEqual(float(ply1[60:70]), 90.0)      # B2 degrees
        self.assertAlmostEqual(float(ply2[20:30]), 45.0)      # B1 degrees
        self.assertAlmostEqual(float(ply2[60:70]), -45.0)     # B2 degrees

    def test_part_composite_long_optcard(self):
        deck = ("*KEYWORD\n*PART_COMPOSITE_LONG\nSheet\n"
                + "OPTCARD".ljust(10) + F(103) + "\n"
                + F(13, 2, 0.0, 0.0, 0.0, 0, 0, 0) + "\n"
                + F(1, 0.002, 30.0, 0, 5, 0.83) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        pi = lines.index("*PART_COMPOSITE_LONG")
        self.assertTrue(lines[pi + 2].startswith("OPTCARD"))  # card kept as-is
        self.assertEqual(int(lines[pi + 3][0:10]), 13)        # PID after OPTCARD
        ply = lines[pi + 4]
        self.assertAlmostEqual(float(ply[10:20]), 2.0)        # THICK1 -> mm
        self.assertAlmostEqual(float(ply[20:30]), 30.0)       # B1 degrees
        self.assertAlmostEqual(float(ply[40:50]), 5.0)        # PLYID1 untouched
        self.assertAlmostEqual(float(ply[50:60]), 0.83)       # SHRFAC1 untouched

    def test_part_composite_tshell_refused(self):
        deck = ("*KEYWORD\n*PART_COMPOSITE_TSHELL\nSheet\n"
                + F(12, 1, 1.0) + "\n" + F(1, 0.001, 0.0, 0) + "\n*END\n")
        p = _write(deck)
        with self.assertRaises(ConvertError):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    @staticmethod
    def _cscm_deck(kw="*MAT_CSCM_CONCRETE_TITLE", units=4):
        title = "concreteCSCM\n" if kw.endswith("_TITLE") else ""
        return ("*KEYWORD\n" + kw + "\n" + title
                + F(4, 2320.0, 2, 0.0, 1, 1.1, 10.0, 0) + "\n"
                + F(0.0) + "\n"
                + F("3.044E7", 0.0254, units) + "\n*END\n")

    def test_mat_cscm_concrete_units_remap(self):
        p = _write(self._cscm_deck())
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False, verify_roundtrip=True)
        lines = _lines(out)
        c1 = lines[lines.index("concreteCSCM") + 1]
        self.assertAlmostEqual(float(c1[10:20]), 2.32e-9)   # RO
        self.assertEqual(int(c1[20:30]), 2)                 # NPLOT flag
        self.assertEqual(int(c1[40:50]), 1)                 # IRATE flag
        self.assertAlmostEqual(float(c1[50:60]), 1.1)       # ERODE strain
        self.assertAlmostEqual(float(c1[60:70]), 10.0)      # RECOV ratio
        c2 = lines[lines.index("concreteCSCM") + 2]
        self.assertAlmostEqual(float(c2[0:10]), 0.0)        # PRED damage
        c3 = lines[lines.index("concreteCSCM") + 3]
        self.assertAlmostEqual(float(c3[0:10]), 30.44)      # FPC Pa -> MPa
        self.assertAlmostEqual(float(c3[10:20]), 25.4)      # DAGG m -> mm
        self.assertEqual(int(c3[20:30]), 2)                 # UNITS 4 -> 2
        self.assertTrue(ctx.roundtrip.startswith("OK"), ctx.roundtrip)
        self.assertEqual(ctx.warnings, [])

    def test_mat_159_concrete_alias(self):
        p = _write(self._cscm_deck(kw="*MAT_159_CONCRETE"))
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        lines = _lines(out)
        c3 = lines[lines.index("*MAT_159_CONCRETE") + 3]
        self.assertAlmostEqual(float(c3[0:10]), 30.44)
        self.assertEqual(int(c3[20:30]), 2)

    def test_mat_cscm_concrete_no_target_units_aborts(self):
        p = _write(self._cscm_deck())
        with self.assertRaisesRegex(ConvertError, "no value for g-cm-us"):
            convert(p, SI, parse_system("g-cm-us"), p + ".o.k",
                    self_check=False)

    def test_mat_cscm_concrete_units_mismatch_warns(self):
        p = _write(self._cscm_deck(units=2))   # flag says ton-mm-s, deck is SI
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        self.assertTrue(any("UNITS=2 declares ton-mm-s" in w
                            for w in ctx.warnings), ctx.warnings)

    def test_mat_cscm_user_defined_refused(self):
        deck = ("*KEYWORD\n*MAT_CSCM\n"
                + F(4, 2320.0, 2, 0.0, 1, 1.1, 10.0, 0) + "\n"
                + F(0.0) + "\n"
                + F("1.15E10", "1.28E10", "1.44E7", 0.31, "1.05E7",
                    "1.93E-2", 1.0, 0.0) + "\n"
                + F(0.74, "1.1E-3", 0.17, 0.07, 0.66, "1.6E-3", 0.16, 0.07) + "\n"
                + F(5.0, "9.0E7", 0.05, "2.5E-10", "3.5E-19") + "\n"
                + F(100.0, 6100.0, 0.1, 61.0, 61.0, 5.0, 1.0, 0.0) + "\n"
                + F("1.0E-4", 0.78, "6.2E-5", 0.48, 21.0, 21.0, 1.0, 1.0)
                + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "user-defined MAT_159"):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_gui_imports(self):
        import kunit.gui  # noqa: F401  (no Tk instantiation)


class IcfdTests(unittest.TestCase):
    """ICFD / *MESH keywords (R16 Vol III), water in kg-m-s."""

    DECK = ("*KEYWORD\n"
            "*ICFD_CONTROL_TIME\n"
            + F(0.5, 0.0, 1.0, 0, -50.0, 0.0, 0.001, "1.00000E28") + "\n"
            + F(0.02) + "\n"
            + F(0, 0, 0) + "\n"
            + F(1, 0.001, 1.0, 0, 0.0, 0.0, 0.002) + "\n"
            "*ICFD_CONTROL_OUTPUT\n"
            + F(4, 0, 0.05, 1, 0, 0) + "\n"
            "*ICFD_DATABASE_DRAG\n"
            + F(4, 0, 0.01, 0, 10, 0, 0) + "\n"
            "*ICFD_MAT_TITLE\n"
            "water\n"
            + F(1, 1, 1000.0, 0.001, 0.0728, 3) + "\n"
            + F(4182.0, 0.6, 0.0, 0.85, 0, 0) + "\n"
            "*ICFD_PART\n"
            + F(1, 1, 1) + "\n"
            "*ICFD_PART_VOL\n"
            + F(5, 1, 1) + "\n"
            + F(1, 2, 3, 4, 0, 0, 0, 0) + "\n"
            "*ICFD_SECTION\n"
            + F(1) + "\n"
            "*ICFD_BOUNDARY_NONSLIP\n"
            + F(4) + "\n"
            "*ICFD_BOUNDARY_FREESLIP\n"
            + F(3) + "\n"
            "*ICFD_BOUNDARY_PRESCRIBED_VEL\n"
            + F(1, 1, 1, 1, 1.0, 0, "1.00000E28", 0.0) + "\n"
            "*ICFD_BOUNDARY_PRESCRIBED_VEL\n"
            + F(1, 2, 1, 2, 1.0, 0, "1.00000E28", 0.0) + "\n"
            "*ICFD_BOUNDARY_PRESCRIBED_PRE\n"
            + F(2, 2, 1.0, "1.00000E28", 0.0) + "\n"
            "*MESH_BL\n"
            + F(4, 3, 0.5, 0.05, 2) + "\n"       # BLST=2: BLTH is a length
            + F(5, 3, 1.2, 0.001, 3) + "\n"      # BLST=3: BLFE is a length
            + F(6, -7, 0.5, 0.05, 2) + "\n"      # negative NELTH: curve ref
            "*MESH_SURFACE_NODE\n"
            "     330        1.281916       0.7851647             0.0\n"
            "*MESH_SURFACE_ELEMENT\n"
            "     318       1     537     538       0       0\n"
            "*MESH_VOLUME\n"
            + F(1) + "\n"
            + F(1, 2, 3, 4, 0, 0, 0, 0) + "\n"
            "*DEFINE_CURVE_TITLE\n"
            "Vel_inlet\n"
            + F(1, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
            + F("0.0", "50.0", w=20) + "\n"
            + F("1.0", "50.0", w=20) + "\n"
            "*DEFINE_CURVE_TITLE\n"
            "Zero\n"
            + F(2, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
            + F("0.0", "0.0", w=20) + "\n"
            + F("2.0", "0.0", w=20) + "\n"
            "*DEFINE_CURVE_TITLE\n"
            "STscale\n"
            + F(3, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
            + F("0.0", "1.0", w=20) + "\n"
            + F("1.0", "1.0", w=20) + "\n"
            "*END\n")

    def test_detect_icfd_si(self):
        v = detect(_write(self.DECK))
        self.assertEqual(v.system, SI)
        self.assertFalse(v.ambiguous, v.table())
        self.assertTrue(any("ICFD fluid density 1000" in e
                            for e in v.evidence), v.evidence)

    def test_icfd_to_ton(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, verify_roundtrip=True)
        lines = _lines(out)
        # ICFD_MAT: RO, VIS, ST scaled; STSFLCID id untouched; HC/TC card
        mi = lines.index("water") + 1
        self.assertAlmostEqual(float(lines[mi][20:30]), 1e-9)      # RO
        self.assertAlmostEqual(float(lines[mi][30:40]), 1e-9)      # VIS
        self.assertAlmostEqual(float(lines[mi][40:50]), 7.28e-5)   # ST
        self.assertEqual(int(lines[mi][50:60]), 3)                 # STSFLCID
        self.assertAlmostEqual(float(lines[mi + 1][0:10]), 4.182e9)  # HC
        self.assertAlmostEqual(float(lines[mi + 1][10:20]), 0.6)     # TC
        # velocity curve ordinates x1000; zero/scale-factor curves unchanged
        ci = lines.index("Vel_inlet")
        self.assertAlmostEqual(float(lines[ci + 2][20:40]), 50000.0)
        zi = lines.index("Zero")
        self.assertAlmostEqual(float(lines[zi + 2][20:40]), 0.0)
        si = lines.index("STscale")
        self.assertAlmostEqual(float(lines[si + 2][20:40]), 1.0)
        # MESH_SURFACE_NODE coordinates x1000 (i8 + 3e16 layout)
        ni = lines.index("*MESH_SURFACE_NODE") + 1
        self.assertEqual(lines[ni][0:8], "     330")
        self.assertAlmostEqual(float(lines[ni][8:24]), 1281.916)
        self.assertAlmostEqual(float(lines[ni][24:40]), 785.1647)
        # MESH_BL: BLST=2 scales BLTH, BLST=3 scales BLFE, negative untouched
        bi = lines.index("*MESH_BL")
        self.assertAlmostEqual(float(lines[bi + 1][20:30]), 500.0)  # BLTH
        self.assertAlmostEqual(float(lines[bi + 1][30:40]), 0.05)   # BLFE
        self.assertAlmostEqual(float(lines[bi + 2][20:30]), 1.2)    # BLTH
        self.assertAlmostEqual(float(lines[bi + 2][30:40]), 1.0)    # BLFE
        self.assertEqual(int(lines[bi + 3][10:20]), -7)             # NELTH
        # MESH_SURFACE_ELEMENT connectivity untouched
        ei = lines.index("*MESH_SURFACE_ELEMENT") + 1
        self.assertEqual(lines[ei], "     318       1     537     538"
                                    "       0       0")
        # the all-zero shared curve downgrades the dim conflict to a warning
        self.assertTrue(any("lcid=2" in w and "immaterial" in w
                            for w in ctx.warnings), ctx.warnings)
        self.assertTrue(ctx.self_check.startswith("OK"), ctx.self_check)
        self.assertTrue(ctx.roundtrip.startswith("OK"), ctx.roundtrip)

    def test_icfd_control_time_to_ms(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, parse_system("kg-mm-ms"), out, self_check=False)
        lines = _lines(out)
        ti = lines.index("*ICFD_CONTROL_TIME")
        c1 = lines[ti + 1]
        self.assertAlmostEqual(float(c1[0:10]), 500.0)     # TTM
        self.assertAlmostEqual(float(c1[40:50]), -50.0)    # DTMIN curve ref
        self.assertAlmostEqual(float(c1[60:70]), 1.0)      # DTINIT
        self.assertAlmostEqual(float(c1[70:80]), 1e31)     # TDEATH
        self.assertAlmostEqual(float(lines[ti + 2][0:10]), 20.0)   # DTT
        c4 = lines[ti + 4]
        self.assertAlmostEqual(float(c4[10:20]), 1.0)      # DTDR
        self.assertAlmostEqual(float(c4[60:70]), 2.0)      # DTINITDR
        oi = lines.index("*ICFD_CONTROL_OUTPUT")
        self.assertAlmostEqual(float(lines[oi + 1][20:30]), 50.0)  # DTOUT
        di = lines.index("*ICFD_DATABASE_DRAG")
        self.assertAlmostEqual(float(lines[di + 1][20:30]), 10.0)  # DTOUT
        vi = lines.index("*ICFD_BOUNDARY_PRESCRIBED_VEL")
        self.assertAlmostEqual(float(lines[vi + 1][60:70]), 1e31)  # DEATH

    def test_icfd_vel_vad4_refused(self):
        deck = ("*KEYWORD\n*ICFD_BOUNDARY_PRESCRIBED_VEL\n"
                + F(1, 1, 4, 9, 1.0, 0, 0.0, 0.0) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "VAD=4"):
            convert(p, SI, TON, p + ".o.k", self_check=False)


class SphTests(unittest.TestCase):
    """SPH bird-strike keywords (R16 Vol I p.12-530, p.19-136, p.41-108;
    Vol II p.2-145), gelatin bird + fluid head in kg-m-s."""

    DECK = ("*KEYWORD\n"
            "*CONTROL_SPH\n"
            + F(1, 1, "1.00000E20", 3, 1500, 0, 0.002, "1.00000E15") + "\n"
            + F(0, 0, 0, 0, 0, 0, 0, 100) + "\n"
            "*SECTION_SPH_TITLE\n"
            "bird\n"
            + F(2, 1.2, 0.2, 2.0, 0.005, "1.00000E20", 0.001) + "\n"
            "*MAT_NULL_TITLE\n"
            "bird\n"
            + F(1, 938.0, -0.09974, 0.0027, 1.1, 0.8, 0.0, 0.0) + "\n"
            "*MAT_ELASTIC_FLUID_TITLE\n"
            "Head\n"
            + F(3, 2600.0, "8.50000E8", 0.24, 0.0, 0.0, "2.20000E9") + "\n"
            + F(0.1, "1.00000E20") + "\n"
            "*ELEMENT_SPH\n"
            "  193725       8    2.220812e-04\n"
            "  193726       9   -5.000000e-08\n"
            "*ELEMENT_SPH_VOLUME\n"
            "  193727       8    4.000000e-08\n"
            "*END\n")

    def test_detect_gelatin_bird_si(self):
        v = detect(_write(self.DECK))
        self.assertEqual(v.system, SI)
        self.assertFalse(v.ambiguous, v.table())
        self.assertTrue(any("material density 938" in e for e in v.evidence),
                        v.evidence)

    def test_sph_to_ton(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, verify_roundtrip=True)
        lines = _lines(out)
        # CONTROL_SPH: DT/START are times (factor 1 here), MAXV x1000
        ci = lines.index("*CONTROL_SPH")
        c1 = lines[ci + 1]
        self.assertEqual(c1[20:30], "1.00000E20")           # DT untouched
        self.assertAlmostEqual(float(c1[60:70]), 0.002)     # START
        self.assertAlmostEqual(float(c1[70:80]), 1.0e18)    # MAXV mm/s
        self.assertEqual(int(lines[ci + 2][70:80]), 100)    # ISYMP percent
        # SECTION_SPH: CSLH/HMIN/HMAX factors stay, SPHINI m -> mm
        si = lines.index("bird") + 1
        self.assertAlmostEqual(float(lines[si][10:20]), 1.2)   # CSLH
        self.assertAlmostEqual(float(lines[si][20:30]), 0.2)   # HMIN
        self.assertAlmostEqual(float(lines[si][30:40]), 2.0)   # HMAX
        self.assertAlmostEqual(float(lines[si][40:50]), 5.0)   # SPHINI mm
        # MAT_ELASTIC_FLUID: RO/E/K/CP scaled, PR/VC untouched
        mi = lines.index("Head") + 1
        self.assertAlmostEqual(float(lines[mi][10:20]), 2.6e-9)   # RO
        self.assertAlmostEqual(float(lines[mi][20:30]), 850.0)    # E MPa
        self.assertAlmostEqual(float(lines[mi][30:40]), 0.24)     # PR
        self.assertAlmostEqual(float(lines[mi][60:70]), 2200.0)   # K MPa
        self.assertAlmostEqual(float(lines[mi + 1][0:10]), 0.1)   # VC
        self.assertAlmostEqual(float(lines[mi + 1][10:20]), 1e14) # CP MPa
        # ELEMENT_SPH: positive = mass x1e-3; negative = volume x1e9;
        # VOLUME option = volume regardless of sign; i8,i8,e16 + pad trick
        ei = lines.index("*ELEMENT_SPH")
        self.assertEqual(lines[ei + 1][0:16], "  193725       8")
        self.assertAlmostEqual(float(lines[ei + 1][16:32]), 2.220812e-7)
        self.assertEqual(lines[ei + 1][16:32].rstrip(),
                         lines[ei + 1][16:30])
        self.assertAlmostEqual(float(lines[ei + 2][16:32]), -50.0)
        vi = lines.index("*ELEMENT_SPH_VOLUME")
        self.assertAlmostEqual(float(lines[vi + 1][16:32]), 40.0)
        self.assertTrue(ctx.self_check.startswith("OK"), ctx.self_check)
        self.assertTrue(ctx.roundtrip.startswith("OK"), ctx.roundtrip)

    def test_sph_times_to_ms(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, parse_system("kg-mm-ms"), out, self_check=False)
        lines = _lines(out)
        c1 = lines[lines.index("*CONTROL_SPH") + 1]
        self.assertAlmostEqual(float(c1[20:30]), 1.0e23)    # DT s -> ms
        self.assertAlmostEqual(float(c1[60:70]), 2.0)       # START
        self.assertAlmostEqual(float(c1[70:80]), 1.0e15)    # MAXV m/s == mm/ms
        sec = lines[lines.index("bird") + 1]
        self.assertAlmostEqual(float(sec[50:60]), 1.0e23)   # DEATH
        self.assertAlmostEqual(float(sec[60:70]), 1.0)      # START

    def test_mat_elastic_fluid_negative_e_refused(self):
        deck = self.DECK.replace("8.50000E8", "   -101.0")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "E < 0"):
            convert(p, SI, TON, p + ".o.k", self_check=False)


if __name__ == "__main__":
    unittest.main()
