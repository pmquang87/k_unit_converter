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

    def test_gui_imports(self):
        import kunit.gui  # noqa: F401  (no Tk instantiation)


if __name__ == "__main__":
    unittest.main()
