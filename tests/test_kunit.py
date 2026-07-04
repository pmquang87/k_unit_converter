import os
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import ConvertError, convert, detect, factor, parse_system
from kunit.parser import KFile, parse_number
from kunit.units import (DENSITY, LENGTH, MASS, PRESSURE, TIME, VELOCITY,
                         blast_unit5_factors)

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")

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


def _write(text, name="deck.k"):
    d = tempfile.mkdtemp(prefix="kunit_test_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


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
        convert(p, SI, TON, out)
        v = detect(out)
        self.assertEqual(v.system, TON)


class ConvertTests(unittest.TestCase):
    def _conv(self, text=DECK_SI):
        p = _write(text)
        out = p + ".out.k"
        ctx = convert(p, SI, TON, out)
        with open(out) as fh:
            return fh.read().split("\n"), ctx

    def _field(self, lines, marker, occurrence, widths, fi, following=0):
        idx = [i for i, ln in enumerate(lines) if ln.startswith(marker)]
        li = idx[occurrence] + 1 + following
        pos = sum(widths[:fi])
        return lines[li][pos:pos + widths[fi]]

    def test_node_and_mass_scaling(self):
        lines, _ = self._conv()
        kf_i = lines.index("*NODE")
        self.assertAlmostEqual(float(lines[kf_i + 1][8:24]), 1700.0)
        self.assertAlmostEqual(float(lines[kf_i + 1][24:40]), 1654.231)
        self.assertAlmostEqual(float(lines[kf_i + 2][40:56]), 78.5, places=4)
        em_i = lines.index("*ELEMENT_MASS")
        self.assertAlmostEqual(float(lines[em_i + 1][16:32]), 0.05)
        self.assertAlmostEqual(float(lines[em_i + 2][16:32]), 0.0055)
        # workaround positioning: value must sit inside chars 21-30
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
        self.assertAlmostEqual(float(lines[ci + 2][0:20]), 0.0)
        self.assertAlmostEqual(float(lines[ci + 3][0:20]), 1.0)  # time unchanged

    def test_termination_time_unchanged(self):
        lines, _ = self._conv()
        ti = lines.index("*CONTROL_TERMINATION")
        self.assertAlmostEqual(float(lines[ti + 1][0:10]), 0.006)

    def test_blast_builtin_unit7(self):
        lines, _ = self._conv()
        bi = lines.index("*LOAD_BLAST_ENHANCED")
        c1 = lines[bi + 1]
        self.assertAlmostEqual(float(c1[10:20]), 0.05)     # M
        self.assertAlmostEqual(float(c1[20:30]), 2500.0)   # XBO
        self.assertAlmostEqual(float(c1[40:50]), 100.0)    # ZBO
        self.assertEqual(int(c1[60:70]), 7)                # built-in ton-mm-s
        c2 = lines[bi + 2]
        self.assertAlmostEqual(float(c2[0:10]), 0.0)       # CFs zeroed

    def test_blast_forced_unit5(self):
        p = _write(DECK_SI)
        out = p + ".u5.k"
        convert(p, SI, TON, out, blast_unit=5)
        with open(out) as fh:
            lines = fh.read().split("\n")
        bi = lines.index("*LOAD_BLAST_ENHANCED")
        self.assertEqual(int(lines[bi + 1][60:70]), 5)
        c2 = lines[bi + 2]
        self.assertAlmostEqual(float(c2[0:10]), 2204.62262, places=3)
        self.assertAlmostEqual(float(c2[10:20]), 3.28084e-3, places=7)
        self.assertAlmostEqual(float(c2[20:30]), 1000.0)
        self.assertAlmostEqual(float(c2[30:40]), 145.0377, places=3)
        self.assertAlmostEqual(float(c2[50:60]), 1e20)     # DEATH untouched

    def test_round_trip(self):
        p = _write(DECK_SI)
        mid = p + ".ton.k"
        back = p + ".back.k"
        convert(p, SI, TON, mid)
        convert(mid, TON, SI, back)
        kf_a, kf_b = KFile(p), KFile(back)
        # each conversion adds 2 provenance comments -> 4 extra after A->B->A
        self.assertEqual(len(kf_a.lines), len(kf_b.lines) - 4)
        # numeric equality of every NODE coordinate
        na = kf_a.lines[kf_a.lines.index("*NODE") + 1]
        nb = kf_b.lines[kf_b.lines.index("*NODE") + 1]
        for sl in (slice(8, 24), slice(24, 40), slice(40, 56)):
            self.assertAlmostEqual(float(na[sl]), float(nb[sl]), places=6)

    def test_unknown_keyword_refused(self):
        deck = DECK_SI.replace("*END", "*AIRBAG_SIMPLE_PRESSURE_VOLUME\n"
                               "         1\n*END")
        p = _write(deck)
        with self.assertRaises(ConvertError):
            convert(p, SI, TON, p + ".o.k")
        ctx = convert(p, SI, TON, p + ".o.k", allow_unknown=True)
        self.assertIn("AIRBAG_SIMPLE_PRESSURE_VOLUME", ctx.unknown)

    def test_include_hard_stop(self):
        deck = DECK_SI.replace("*END", "*INCLUDE\nother.k\n*END")
        p = _write(deck)
        with self.assertRaises(ConvertError):
            convert(p, SI, TON, p + ".o.k", allow_unknown=True)

    def test_parameter_field_refused(self):
        deck = DECK_SI.replace(
            "         1    7850.02.10000E11",
            "         1    &dens 2.10000E11")
        p = _write(deck)
        with self.assertRaises(Exception):
            convert(p, SI, TON, p + ".o.k")

    def test_free_format_line(self):
        deck = DECK_SI.replace(
            "       1             1.7        1.654231       0.2799996       0       0",
            "1, 1.7, 1.654231, 0.2799996, 0, 0")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out)
        with open(out) as fh:
            lines = fh.read().split("\n")
        ni = lines.index("*NODE")
        toks = lines[ni + 1].split(",")
        self.assertAlmostEqual(float(toks[1]), 1700.0)
        self.assertAlmostEqual(float(toks[3]), 279.9996)


if __name__ == "__main__":
    unittest.main()
