"""Tier-1 coverage additions (schema.py): new material Specs, CONTACT_*_MORTAR
routing, DEFINE_TABLE_2D/3D aliasing, flag-only whitelists, MAT_100 failure-card
refusal and *AIRBAG_REFERENCE_GEOMETRY.  Each keyword is asserted to classify as
spec/custom/white (NOT unknown) and a representative dimensional field is checked
to scale (or to be correctly left unchanged).  kg-m-s -> ton-mm-s throughout
(pressure x1e-6, length x1e3, density x1e-12, force/time x1)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import convert, parse_system
from kunit.schema import resolve, h_contact, h_define_table

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")


def F(*vals, w=10):
    return "".join(str(v).rjust(w) for v in vals)


def _write(text, name="deck.k", d=None):
    d = d or tempfile.mkdtemp(prefix="kunit_t1_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _lines(path):
    with open(path, newline="") as fh:
        return fh.read().split("\n")


def _conv(text, src=SI, dst=TON, **kw):
    p = _write(text)
    out = p + ".o.k"
    kw.setdefault("self_check", False)
    ctx = convert(p, src, dst, out, **kw)
    return _lines(out), ctx


class ClassificationTests(unittest.TestCase):
    """resolve() verdicts for every added keyword."""

    def test_kinds(self):
        spec = {"MAT_002", "MAT_054", "MAT_055", "MAT_057", "MAT_063",
                "MAT_098", "MAT_100", "MAT_123", "MAT_181",
                "MAT_ORTHOTROPIC_ELASTIC", "MAT_SIMPLIFIED_JOHNSON_COOK",
                "MAT_SPOTWELD", "MAT_CRUSHABLE_FOAM", "MAT_LOW_DENSITY_FOAM",
                "MAT_ENHANCED_COMPOSITE_DAMAGE",
                "MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY",
                "MAT_SIMPLIFIED_RUBBER/FOAM", "AIRBAG_REFERENCE_GEOMETRY"}
        for name in spec:
            self.assertEqual(resolve(name)[0], "spec", name)
        custom = {"CONTACT_AUTOMATIC_SINGLE_SURFACE_MORTAR": h_contact,
                  "DEFINE_TABLE_2D": h_define_table,
                  "DEFINE_TABLE_3D": h_define_table}
        for name, fn in custom.items():
            self.assertEqual(resolve(name), ("custom", fn), name)
        for name in ("SECTION_TSHELL", "DEFINE_COORDINATE_VECTOR",
                     "CONSTRAINED_TIED_NODES_FAILURE"):
            self.assertEqual(resolve(name), ("white", None), name)


class ContactMortarTests(unittest.TestCase):
    def test_single_surface_mortar(self):
        deck = ("*KEYWORD\n*CONTACT_AUTOMATIC_SINGLE_SURFACE_MORTAR\n"
                + F(1, 0, 2, 0) + "\n"
                + F(0.2, 0.1, 0.0, "1.0E6", 0.0, 0, 0.0, "1.0E20") + "\n"
                + F(1.0, 1.0, 5.0, 5.0) + "\n*END\n")
        lines, ctx = _conv(deck)
        i = lines.index("*CONTACT_AUTOMATIC_SINGLE_SURFACE_MORTAR")
        c2, c3 = lines[i + 2], lines[i + 3]
        self.assertAlmostEqual(float(c2[30:40]), 1.0)      # VC 1e6 Pa -> MPa
        self.assertAlmostEqual(float(c3[20:30]), 5000.0)   # SST m -> mm
        self.assertAlmostEqual(float(c3[30:40]), 5000.0)   # MST m -> mm


class MaterialSpecTests(unittest.TestCase):
    def test_mat_002_orthotropic_elastic(self):
        deck = ("*KEYWORD\n*MAT_002\n"
                + F(1, 1450.0, "1.3E11", "4.5E10", "4.5E10", 0.15, 0.15, 0.25)
                + "\n" + F("8.0E9", "8.0E9", "8.0E9", 2.0, 0, 0) + "\n"
                + F(1.5, -2.5, 0.0, 1.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 45.0) + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*MAT_002")
        c1, c2, c3, c4 = lines[i + 1], lines[i + 2], lines[i + 3], lines[i + 4]
        self.assertAlmostEqual(float(c1[10:20]), 1.45e-9)   # RO
        self.assertAlmostEqual(float(c1[20:30]), 1.3e5)     # EA
        self.assertAlmostEqual(float(c1[50:60]), 0.15)      # PRBA unchanged
        self.assertAlmostEqual(float(c2[0:10]), 8000.0)     # GAB
        self.assertAlmostEqual(float(c2[30:40]), 2.0)       # AOPT unchanged
        self.assertAlmostEqual(float(c3[0:10]), 1500.0)     # XP length
        self.assertAlmostEqual(float(c4[60:70]), 45.0)      # BETA unchanged

    def test_mat_123_modified_piecewise(self):
        deck = ("*KEYWORD\n*MAT_123\n"
                + F(1, 7850.0, "2.1E11", 0.3, "3.5E8", "1.1E10", 0.0, 0.0)
                + "\n" + F(0.0, 0.0, 100, 0) + "\n"
                "*DEFINE_CURVE\n" + F(100, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", "3.5E8", w=20) + "\n"
                + F("0.1", "4.5E8", w=20) + "\n*END\n")
        lines, ctx = _conv(deck)
        i = lines.index("*MAT_123")
        c1 = lines[i + 1]
        self.assertAlmostEqual(float(c1[10:20]), 7.85e-9)   # RO
        self.assertAlmostEqual(float(c1[20:30]), 2.1e5)     # E
        self.assertAlmostEqual(float(c1[40:50]), 350.0)     # SIGY
        self.assertAlmostEqual(float(c1[50:60]), 1.1e4)     # ETAN
        ci = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[ci + 2][0:20]), 0.0)   # strain
        self.assertAlmostEqual(float(lines[ci + 2][20:40]), 350.0)  # stress
        self.assertFalse(any("unreferenced" in w for w in ctx.warnings),
                         ctx.warnings)

    def test_mat_098_simplified_johnson_cook(self):
        deck = ("*KEYWORD\n*MAT_098\n"
                + F(1, 7850.0, "2.0E11", 0.3, 0) + "\n"
                + F("3.0E8", "1.0E8", 0.26, 0.014, 0.8, "6.0E8", "5.0E8", 1.0)
                + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*MAT_098")
        c1, c2 = lines[i + 1], lines[i + 2]
        self.assertAlmostEqual(float(c1[10:20]), 7.85e-9)   # RO
        self.assertAlmostEqual(float(c1[20:30]), 2.0e5)     # E
        self.assertAlmostEqual(float(c2[0:10]), 300.0)      # A
        self.assertAlmostEqual(float(c2[10:20]), 100.0)     # B
        self.assertAlmostEqual(float(c2[30:40]), 0.014)     # C unchanged
        self.assertAlmostEqual(float(c2[50:60]), 600.0)     # SIGMAX
        self.assertAlmostEqual(float(c2[60:70]), 500.0)     # SIGSAT
        self.assertAlmostEqual(float(c2[70:80]), 1.0)       # EPS0 unchanged

    def test_mat_054_enhanced_composite_damage(self):
        deck = ("*KEYWORD\n*MAT_054\n"
                + F(1, 1450.0, "1.3E11", "4.5E10", "4.5E10", 0.15, 0.15, 0.25)
                + "\n" + F("8.0E9", "8.0E9", "8.0E9", "2.0E9", 2.0, 0) + "\n"
                + F(1.5, -2.5, 0.0, 1.0, 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01, 0.02) + "\n"
                + F(0.0, 0.2, 1.0, 0.5, 1.2, 0.01, 0.01, 0.0) + "\n"
                + F("1.5E9", "2.0E9", "1.0E8", "2.5E8", "1.0E8", 0.0, 0.0)
                + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*MAT_054")
        c1, c2 = lines[i + 1], lines[i + 2]
        c4, c5, c6 = lines[i + 4], lines[i + 5], lines[i + 6]
        self.assertAlmostEqual(float(c1[20:30]), 1.3e5)     # EA
        self.assertAlmostEqual(float(c2[0:10]), 8000.0)     # GAB
        self.assertAlmostEqual(float(c2[30:40]), 2000.0)    # KF bulk modulus
        self.assertAlmostEqual(float(c4[70:80]), 0.02)      # DFAILS strain
        self.assertAlmostEqual(float(c5[10:20]), 0.2)       # ALPH unchanged
        self.assertAlmostEqual(float(c6[0:10]), 1500.0)     # XC
        self.assertAlmostEqual(float(c6[10:20]), 2000.0)    # XT

    def test_mat_063_crushable_foam(self):
        deck = ("*KEYWORD\n*MAT_063\n"
                + F(1, 100.0, "1.0E7", 0.0, 50, "1.0E5", 0.1, 0.0) + "\n"
                "*DEFINE_CURVE\n" + F(50, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", "1.0E5", w=20) + "\n"
                + F("0.5", "2.0E5", w=20) + "\n*END\n")
        lines, ctx = _conv(deck)
        i = lines.index("*MAT_063")
        c1 = lines[i + 1]
        self.assertAlmostEqual(float(c1[20:30]), 10.0)      # E 1e7 -> MPa
        self.assertAlmostEqual(float(c1[50:60]), 0.1)       # TSC 1e5 -> MPa
        ci = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[ci + 3][0:20]), 0.5)   # strain
        self.assertAlmostEqual(float(lines[ci + 3][20:40]), 0.2)  # stress
        self.assertFalse(any("unreferenced" in w for w in ctx.warnings),
                         ctx.warnings)

    def test_mat_057_low_density_foam(self):
        deck = ("*KEYWORD\n*MAT_057\n"
                + F(1, 60.0, "5.0E6", 40, "1.0E5", 0.5, 0.0, 0.1) + "\n"
                "*DEFINE_CURVE\n" + F(40, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", "5.0E6", w=20) + "\n"
                + F("0.3", "8.0E6", w=20) + "\n*END\n")
        lines, ctx = _conv(deck)
        i = lines.index("*MAT_057")
        c1 = lines[i + 1]
        self.assertAlmostEqual(float(c1[20:30]), 5.0)       # E 5e6 -> MPa
        self.assertAlmostEqual(float(c1[40:50]), 0.1)       # TC 1e5 -> MPa
        ci = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[ci + 2][20:40]), 5.0)   # stress
        self.assertFalse(any("unreferenced" in w for w in ctx.warnings),
                         ctx.warnings)

    def test_mat_181_simplified_rubber(self):
        deck = ("*KEYWORD\n*MAT_181\n"
                + F(1, 1000.0, "2.0E9", 0.0, 0.0, 0.0, 0, 0) + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*MAT_181")
        c1 = lines[i + 1]
        self.assertAlmostEqual(float(c1[10:20]), 1.0e-9)    # RO
        self.assertAlmostEqual(float(c1[20:30]), 2000.0)    # KM 2e9 -> MPa

    def test_mat_100_spotweld_card1_scaled_card2_warned(self):
        deck = ("*KEYWORD\n*MAT_100\n"
                + F(1, 7850.0, "2.1E11", 0.3, "3.5E8", "1.0E9", 0.0, "1.0E20")
                + "\n"
                + F(0.1, "5.0E4", "5.0E4", "5.0E4", 10.0, 10.0, 10.0, 0)
                + "\n*END\n")
        lines, ctx = _conv(deck)
        i = lines.index("*MAT_100")
        c1, c2 = lines[i + 1], lines[i + 2]
        self.assertAlmostEqual(float(c1[20:30]), 2.1e5)     # E
        self.assertAlmostEqual(float(c1[40:50]), 350.0)     # SIGY
        self.assertAlmostEqual(float(c1[50:60]), 1000.0)    # ET
        self.assertAlmostEqual(float(c2[10:20]), 5.0e4)     # NRR left unscaled
        self.assertTrue(any("resultants" in w for w in ctx.warnings),
                        ctx.warnings)


class DefineTableTests(unittest.TestCase):
    def test_define_table_2d_pair_form(self):
        deck = ("*KEYWORD\n*MAT_PIECEWISE_LINEAR_PLASTICITY\n"
                + F(1, 7850.0, "2.1E11", 0.3, "3.5E8", 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 200, 0, 0) + "\n"
                "*DEFINE_TABLE_2D\n" + F(200, 1.0, 0.0) + "\n"
                + F("0.0", 201, w=20) + "\n"
                "*DEFINE_CURVE\n" + F(201, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("0.0", "3.5E8", w=20) + "\n*END\n")
        lines, ctx = _conv(deck)
        ci = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[ci + 2][20:40]), 350.0)  # stress
        self.assertFalse(any("unreferenced" in w for w in ctx.warnings),
                         ctx.warnings)

    def test_define_table_3d_unreferenced_unchanged(self):
        deck = ("*KEYWORD\n*DEFINE_TABLE_3D\n" + F(300, 1.0, 0.0) + "\n"
                + F("0.0", 301, w=20) + "\n*END\n")
        # unreferenced table: no abort, value column left unchanged
        lines, ctx = _conv(deck)
        i = lines.index("*DEFINE_TABLE_3D")
        self.assertAlmostEqual(float(lines[i + 2][0:20]), 0.0)
        self.assertTrue(any("unreferenced" in w for w in ctx.warnings),
                        ctx.warnings)


class WhitelistTests(unittest.TestCase):
    def test_constrained_tied_nodes_failure(self):
        deck = ("*KEYWORD\n*CONSTRAINED_TIED_NODES_FAILURE\n"
                + F(5, 0.1, 0) + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*CONSTRAINED_TIED_NODES_FAILURE")
        self.assertAlmostEqual(float(lines[i + 1][10:20]), 0.1)  # EPPF strain

    def test_section_tshell(self):
        deck = ("*KEYWORD\n*SECTION_TSHELL\n"
                + F(1, 2, 1.0, 5, 0, 0, 0, 0) + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*SECTION_TSHELL")
        self.assertAlmostEqual(float(lines[i + 1][20:30]), 1.0)  # SHRF unchanged

    def test_define_coordinate_vector(self):
        deck = ("*KEYWORD\n*DEFINE_COORDINATE_VECTOR\n"
                + F(3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0) + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*DEFINE_COORDINATE_VECTOR")
        self.assertAlmostEqual(float(lines[i + 1][10:20]), 1.0)  # XX unchanged


class AirbagTests(unittest.TestCase):
    def test_reference_geometry_node_coords(self):
        deck = ("*KEYWORD\n*AIRBAG_REFERENCE_GEOMETRY\n"
                "       1" + "1.5".rjust(16) + "2.5".rjust(16) + "0.0".rjust(16)
                + "\n       2" + "-1.6".rjust(16) + "0.0".rjust(16)
                + "3.0".rjust(16) + "\n*END\n")
        lines, _ = _conv(deck)
        i = lines.index("*AIRBAG_REFERENCE_GEOMETRY")
        self.assertEqual(lines[i + 1][0:8], "       1")     # NID untouched
        self.assertAlmostEqual(float(lines[i + 1][8:24]), 1500.0)   # X
        self.assertAlmostEqual(float(lines[i + 1][24:40]), 2500.0)  # Y
        self.assertAlmostEqual(float(lines[i + 2][8:24]), -1600.0)  # X node 2


if __name__ == "__main__":
    unittest.main()
