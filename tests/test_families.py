"""Keyword families added from the R16 manual research pass.

kg-m-s -> ton-mm-s factors used throughout: length x1e3, mass x1e-3,
density x1e-12, pressure x1e-6, force x1, moment/energy x1e3, volume x1e9,
area x1e6, specific heat x1e6, thermal conductivity x1, 1/stress x1e6,
power x1e3.  Time (and therefore frequency and rate) is UNCHANGED between
those two systems, so the time-sensitive assertions convert to kg-mm-ms
instead (time x1e3).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import ConvertError, convert, factor, parse_system

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")
KGMM = parse_system("kg-mm-ms")


def F(*vals, w=10):
    """Fixed-width card line from values."""
    return "".join(str(v).rjust(w) for v in vals)


def I8(*vals):
    """Integer card written in the 8-char columns *ELEMENT_* uses."""
    return "".join(str(v).rjust(8) for v in vals)


def _write(text, name="deck.k", d=None):
    d = d or tempfile.mkdtemp(prefix="kunit_fam_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _lines(path):
    with open(path, newline="") as fh:
        return fh.read().split("\n")


def _at(lines, kw, n):
    """Line n of the block introduced by `kw` (convert() inserts a header,
    so raw line numbers do not line up between input and output)."""
    return lines[lines.index(kw) + n]


class TshellStructTests(unittest.TestCase):
    """Thick shells, beam orientation, IGA patches (R16 Vol I p.19/29/41)."""

    DECK = ("*KEYWORD\n"
            "*ELEMENT_TSHELL\n"
            + I8(1, 1, 1, 2, 3, 4, 5, 6, 7, 8) + "\n"
            "*ELEMENT_TSHELL_BETA\n"
            + I8(2, 1, 1, 2, 3, 4, 5, 6, 7, 8) + "\n"
            + F("", "", "", "", 30.0, w=16) + "\n"
            "*ELEMENT_SHELL_BETA\n"
            + I8(11, 1, 1, 2, 3, 4) + "\n"
            + F(0.05, 0.04, 0.03, 0.02, 30.0, w=16) + "\n"
            "*SECTION_TSHELL\n"
            + F(1, 2, 0.0, 5, 1.0, 0, 0, 0) + "\n"
            "*INTEGRATION_SHELL\n"
            + F(1, 3, 0) + "\n"
            + F(-0.7745967, 0.5555556, 0) + "\n"
            + F(0.0, 0.8888889, 0) + "\n"
            + F(0.7745967, 0.5555556, 0) + "\n"
            "*ELEMENT_BEAM_ORIENTATION\n"
            + I8(101, 101, 101, 102, 0, 0, 0, 0, 0, 2) + "\n"
            + F(0.0, 1.0, 0.0) + "\n"
            "*ELEMENT_SOLID_NURBS_PATCH\n"
            + F(1, 1, 9, 1, 33, 1, 9, 1) + "\n"
            + F(0, 1, 1, 1, 0) + "\n"
            + F(0.0, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75) + "\n"
            + F(1, 2, 3, 4, 5, 6, 7, 8) + "\n"
            "*MAT_ELASTIC\n"
            + F(1, 7850.0, "2.10000E11", 0.3) + "\n"
            "*END\n")

    def test_topology_untouched_but_shell_beta_thickness_scaled(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        src, dst = _lines(p), _lines(out)
        for kw in ("*ELEMENT_TSHELL", "*ELEMENT_TSHELL_BETA",
                   "*SECTION_TSHELL", "*INTEGRATION_SHELL",
                   "*ELEMENT_BEAM_ORIENTATION", "*ELEMENT_SOLID_NURBS_PATCH"):
            i, j = src.index(kw), dst.index(kw)
            n = 1
            while i + n < len(src) and not src[i + n].startswith("*"):
                self.assertEqual(src[i + n], dst[j + n],
                                 f"{kw} data line {n} changed")
                n += 1
        # *ELEMENT_SHELL_BETA shares *ELEMENT_SHELL's Thickness Card, so its
        # four THIC fields are lengths (they were silently whitelisted before)
        b = dst.index("*ELEMENT_SHELL_BETA")
        self.assertEqual(dst[b + 1], src[src.index("*ELEMENT_SHELL_BETA") + 1])
        self.assertAlmostEqual(float(dst[b + 2][0:16]), 50.0)
        self.assertAlmostEqual(float(dst[b + 2][16:32]), 40.0)
        self.assertAlmostEqual(float(dst[b + 2][32:48]), 30.0)
        self.assertAlmostEqual(float(dst[b + 2][48:64]), 20.0)
        self.assertAlmostEqual(float(dst[b + 2][64:80]), 30.0)   # BETA, deg


class ConstrainedTests(unittest.TestCase):
    """*CONSTRAINED_* family (R16 Vol I p.10)."""

    DECK = ("*KEYWORD\n"
            "*CONSTRAINED_GENERALIZED_WELD_BUTT\n"
            + F(1, 0, 4, 0.002) + "\n"
            + F("1.00000E20", 0.3, 0.25, 0.9, 10.0, 2.0, 1.0) + "\n"
            "*CONSTRAINED_GLOBAL\n"
            + F(1, 0, 1, 1.6, 0.0, 0.0, 0.001) + "\n"
            + F(2, 0, 2, 0.0, 0.61, 0.0, 0.0) + "\n"
            "*CONSTRAINED_JOINT_SCREW_ID\n"
            + F(7) + "screw joint\n"
            + F(1, 2, 3, 4, 5, 6, 1.0, 0.0) + "\n"
            + F(25.0, 0, 0, 0.5, 30.0) + "\n"
            "*CONSTRAINED_NODAL_RIGID_BODY_SPC\n"
            + F(1, 0, 1, 0, 0, 0, 0) + "\n"
            + F(1.0, 7, 100000, 0, 0.1, 0.2, 0.3) + "\n"
            "*CONSTRAINED_NODE_SET\n"
            + F(11, 1, "1.00000E20") + "\n"
            "*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED\n"
            + F(1, 1, 2, 1, 1, 0, 0.0) + "\n"
            + F(0, 0, 0, 0, 0, 0) + "\n"
            + F(191.0, 80.0, 0.0, 0.0, 0.0, 0.0) + "\n"
            + F(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + "\n"
            "*CONSTRAINED_SHELL_IN_SOLID_ID\n"
            + F(3) + "shell in solid\n"
            + F(1, 2, 0, 0) + "\n"
            + F(0.001, 0.5, 0, 0, 0, 0.1) + "\n"
            "*CONSTRAINED_JOINT_REVOLUTE_ID\n"
            + F(9) + "revolute\n"
            + F(1, 9, 2, 10, 0, 0, 1.0, 1.0) + "\n"
            "*CONSTRAINED_SHELL_TO_SOLID\n"
            + F(326, 1) + "\n"
            "*CONSTRAINED_TIED_NODES_FAILURE\n"
            + F(101, 0.085, 0, 0) + "\n"
            "*END\n")

    def test_lengths_stresses_moments(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        src, dst = _lines(p), _lines(out)
        w = dst.index("*CONSTRAINED_GENERALIZED_WELD_BUTT")
        self.assertAlmostEqual(float(dst[w + 2][20:30]), 2.5e-7)   # SIGY
        self.assertAlmostEqual(float(dst[w + 2][30:40]), 0.9)      # BETA
        self.assertAlmostEqual(float(dst[w + 2][40:50]), 10000.0)  # L
        self.assertAlmostEqual(float(dst[w + 2][50:60]), 2000.0)   # D
        self.assertAlmostEqual(float(dst[w + 2][60:70]), 1000.0)   # legacy Lt
        g = dst.index("*CONSTRAINED_GLOBAL")
        self.assertAlmostEqual(float(dst[g + 1][30:40]), 1600.0)   # X
        self.assertAlmostEqual(float(dst[g + 1][60:70]), 1.0)      # TOL
        self.assertAlmostEqual(float(dst[g + 2][40:50]), 610.0)    # 2nd card Y
        s = dst.index("*CONSTRAINED_JOINT_SCREW_ID")
        self.assertEqual(dst[s + 2],
                         _at(src, "*CONSTRAINED_JOINT_SCREW_ID", 2))  # nodes
        self.assertAlmostEqual(float(dst[s + 3][0:10]), 25000.0)   # PARM
        self.assertAlmostEqual(float(dst[s + 3][30:40]), 500.0)    # R1
        self.assertAlmostEqual(float(dst[s + 3][40:50]), 30.0)     # H_ANGLE
        n = dst.index("*CONSTRAINED_NODAL_RIGID_BODY_SPC")
        self.assertEqual(dst[n + 1],
                         _at(src, "*CONSTRAINED_NODAL_RIGID_BODY_SPC", 1))
        self.assertAlmostEqual(float(dst[n + 2][0:10]), 1.0)       # CMO flag
        self.assertEqual(int(dst[n + 2][20:30]), 100000)           # CON2 bits
        self.assertAlmostEqual(float(dst[n + 2][40:50]), 100.0)    # XSPC
        self.assertAlmostEqual(float(dst[n + 2][60:70]), 300.0)    # ZSPC
        j = dst.index("*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED")
        self.assertAlmostEqual(float(dst[j + 3][0:10]), 191000.0)  # ESPH
        self.assertAlmostEqual(float(dst[j + 3][10:20]), 80000.0)  # FMPH
        self.assertEqual(dst[j + 4], _at(
            src, "*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED", 4))    # stop ang.
        for kw in ("*CONSTRAINED_JOINT_REVOLUTE_ID",
                   "*CONSTRAINED_SHELL_TO_SOLID",
                   "*CONSTRAINED_TIED_NODES_FAILURE"):
            self.assertEqual(dst[dst.index(kw) + 1], src[src.index(kw) + 1])

    def test_times_scale_to_ms(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        dst = _lines(out)
        w = dst.index("*CONSTRAINED_GENERALIZED_WELD_BUTT")
        self.assertAlmostEqual(float(dst[w + 1][30:40]), 2.0)      # WINDOW
        self.assertAlmostEqual(float(dst[w + 2][0:10]), 1e23)      # TFAIL
        c = dst.index("*CONSTRAINED_NODE_SET")
        self.assertAlmostEqual(float(dst[c + 1][20:30]), 1e23)     # TF
        s = dst.index("*CONSTRAINED_SHELL_IN_SOLID_ID")
        self.assertAlmostEqual(float(dst[s + 3][0:10]), 1.0)       # START
        self.assertAlmostEqual(float(dst[s + 3][10:20]), 500.0)    # END
        self.assertAlmostEqual(float(dst[s + 3][50:60]), 0.1)      # PSSF

    def test_shell_in_solid_9999_sentinel(self):
        deck = ("*KEYWORD\n*CONSTRAINED_SHELL_IN_SOLID\n"
                + F(1, 2, 0, 0) + "\n"
                + F(77, -9999, 0, 0, 0, 0.1) + "\n"
                "*DEFINE_CURVE\n"
                + F(77, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("0.0", "0.002", w=20) + "\n"
                + F("0.004", "0.006", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, KGMM, out, self_check=False)
        dst = _lines(out)
        i = dst.index("*CONSTRAINED_SHELL_IN_SOLID")
        self.assertEqual(int(dst[i + 2][0:10]), 77)       # START = curve id
        self.assertEqual(int(dst[i + 2][10:20]), -9999)   # sentinel intact
        c = dst.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(dst[c + 2][0:20]), 0.0)
        self.assertAlmostEqual(float(dst[c + 2][20:40]), 2.0)   # both axes
        self.assertAlmostEqual(float(dst[c + 3][0:20]), 4.0)    # are times
        self.assertTrue(any("-9999" in n for n in ctx.notes), ctx.notes)

    def test_joint_stiffness_curve_id_not_scaled(self):
        deck = ("*KEYWORD\n*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED\n"
                + F(1, 1, 2, 1, 1, 0, 0.0) + "\n"
                + F(0, 0, 0, 0, 0, 0) + "\n"
                + F(191.0, -3, 0.0, 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + "\n"
                "*DEFINE_CURVE\n"
                + F(3, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("0.0", "50.0", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        dst = _lines(out)
        i = dst.index("*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED")
        self.assertEqual(int(dst[i + 3][10:20]), -3)     # curve id untouched
        c = dst.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(dst[c + 2][20:40]), 50000.0)   # moment

    def test_joint_stiffness_table_form_refused(self):
        deck = ("*KEYWORD\n*CONSTRAINED_JOINT_STIFFNESS_GENERALIZED\n"
                + F(1, 1, 2, 1, 1, 3800007, 0.0) + "\n"
                + F(0, 0, 0, 0, 0, 0) + "\n"
                + F(191.0, -3, 0.0, 0.0, 0.0, 0.0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "table"):
            convert(p, SI, TON, p + ".o.k", self_check=False)


class MatEosNewTests(unittest.TestCase):
    """Materials and equations of state added from R16 Vol II."""

    DECK = ("*KEYWORD\n"
            "*MAT_ELASTIC_PLASTIC_HYDRO\n"
            + F(1, 8900.0, "4.70000E10", "1.20000E8", "1.00000E8",
                "-1.00000E9", 0.0, 0.002) + "\n"
            + F(0.0, 0.1, 0.2) + "\n"
            + F(0.0, 0.0, 0.0) + "\n"
            + F("1.20000E8", "1.40000E8", "1.60000E8") + "\n"
            + F(0.0, 0.0, 0.0) + "\n"
            "*EOS_TABULATED\n"
            + F(9, 0.4, "1.00000E6", 1.0, 0, 0) + "\n"
            + F(0.0, -0.1, -0.2, -0.3, -0.4, w=16) + "\n"
            + F(-0.5, -0.6, -0.7, -0.8, -0.9, w=16) + "\n"
            + F(0.0, "1.00000E8", "2.00000E8", "3.00000E8", "4.00000E8",
                w=16) + "\n"
            + F("5.00000E8", "6.00000E8", "7.00000E8", "8.00000E8",
                "9.00000E8", w=16) + "\n"
            + F(0.4, 0.4, 0.4, 0.4, 0.4, w=16) + "\n"
            + F(0.4, 0.4, 0.4, 0.4, 0.4, w=16) + "\n"
            "*EOS_MURNAGHAN\n"
            + F(19, 7.0, "1.50000E8", 0.0) + "\n"
            "*MAT_ACOUSTIC\n"
            + F(90, 1.21, 340.0, 0.5, 0, 101325.0, 9.80665) + "\n"
            + F(0.1, 0.2, 0.3, 0.0, 0.0, 1.0) + "\n"
            "*MAT_ELASTIC_PLASTIC_THERMAL\n"
            + F(4, 7850.0) + "\n"
            + F(-1000.0, 0.0, 1000.0) + "\n"
            + F("2.10000E11", "2.00000E11", "1.00000E11") + "\n"
            + F(0.3, 0.3, 0.3) + "\n"
            + F("1.20000E-5", "1.20000E-5", "1.20000E-5") + "\n"
            + F("3.00000E8", "2.50000E8", "1.00000E8") + "\n"
            + F("1.00000E9", "1.00000E9", "1.00000E9") + "\n"
            "*MAT_MOONEY-RIVLIN_RUBBER_TITLE\n"
            "rubber\n"
            + F(27, 1130.0, 0.495, "1.00000E6", "1.00000E5", 0.0) + "\n"
            + F(0.05, 0.01, 0.002, 5) + "\n"
            "*DEFINE_CURVE\n"
            + F(5, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("0.001", "20.0", w=20) + "\n"
            "*MAT_ANISOTROPIC_ELASTIC\n"
            + F(2, 2510.0, "5.94000E9", "1.00000E9", "5.94000E9",
                "1.00000E9", "1.00000E9", "5.94000E9") + "\n"
            + F(0, 0, 0, "1.18000E9", 0, 0, 0, 0) + "\n"
            + F("1.18000E9", 0, 0, 0, 0, 0, "1.18000E9", 0) + "\n"
            + F(0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0, 0) + "\n"
            + F(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 45.0, 0) + "\n"
            "*MAT_ADD_THERMAL_EXPANSION\n"
            + F(1, 0, "1.20000E-5", 0, 0.0, 0, 0.0, 293.0) + "\n"
            "*END\n")

    def test_material_fields(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        src, dst = _lines(p), _lines(out)
        h = dst.index("*MAT_ELASTIC_PLASTIC_HYDRO")
        self.assertAlmostEqual(float(dst[h + 1][10:20]), 8.9e-9)    # RO
        self.assertAlmostEqual(float(dst[h + 1][20:30]), 4.7e4)     # G
        self.assertAlmostEqual(float(dst[h + 1][50:60]), -1000.0)   # PC sign
        self.assertAlmostEqual(float(dst[h + 1][70:80]), 2.0)       # CHARL
        self.assertEqual(dst[h + 2],
                         _at(src, "*MAT_ELASTIC_PLASTIC_HYDRO", 2))  # EPS card
        self.assertAlmostEqual(float(dst[h + 4][0:10]), 120.0)      # ES1
        self.assertAlmostEqual(float(dst[h + 4][20:30]), 160.0)     # ES3
        e = dst.index("*EOS_TABULATED")
        self.assertAlmostEqual(float(dst[e + 1][20:30]), 1.0)       # E0
        self.assertEqual(dst[e + 2], _at(src, "*EOS_TABULATED", 2))  # EV card
        self.assertAlmostEqual(float(dst[e + 4][16:32]), 100.0)     # C2, 16ch
        self.assertAlmostEqual(float(dst[e + 5][64:80]), 900.0)     # C10
        self.assertEqual(dst[e + 6], _at(src, "*EOS_TABULATED", 6))  # T card
        m = dst.index("*EOS_MURNAGHAN")
        self.assertAlmostEqual(float(dst[m + 1][10:20]), 7.0)       # GAMMA
        self.assertAlmostEqual(float(dst[m + 1][20:30]), 150.0)     # K0
        a = dst.index("*MAT_ACOUSTIC")
        self.assertAlmostEqual(float(dst[a + 1][10:20]), 1.21e-12)  # RO
        self.assertAlmostEqual(float(dst[a + 1][20:30]), 340000.0)  # C
        self.assertAlmostEqual(float(dst[a + 1][30:40]), 0.5)       # BETA
        self.assertAlmostEqual(float(dst[a + 1][50:60]), 0.101325)  # ATMOS
        self.assertAlmostEqual(float(dst[a + 1][60:70]), 9806.65)   # GRAV
        self.assertAlmostEqual(float(dst[a + 2][0:10]), 100.0)      # XP
        self.assertAlmostEqual(float(dst[a + 2][50:60]), 1.0)       # ZN cosine
        t = dst.index("*MAT_ELASTIC_PLASTIC_THERMAL")
        self.assertEqual(dst[t + 2],
                         _at(src, "*MAT_ELASTIC_PLASTIC_THERMAL", 2))  # T1..T3
        self.assertAlmostEqual(float(dst[t + 3][0:10]), 210000.0)   # E1
        self.assertEqual(dst[t + 5],
                         _at(src, "*MAT_ELASTIC_PLASTIC_THERMAL", 5))  # ALPHA
        self.assertAlmostEqual(float(dst[t + 6][0:10]), 300.0)      # SIGY1
        r = dst.index("*MAT_MOONEY-RIVLIN_RUBBER_TITLE")
        self.assertAlmostEqual(float(dst[r + 2][10:20]), 1.13e-9)   # RO
        self.assertAlmostEqual(float(dst[r + 2][30:40]), 1.0)       # A
        self.assertAlmostEqual(float(dst[r + 2][40:50]), 0.1)       # B
        self.assertAlmostEqual(float(dst[r + 3][0:10]), 50.0)       # SGL
        c = dst.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(dst[c + 2][0:20]), 1.0)        # dL
        self.assertAlmostEqual(float(dst[c + 2][20:40]), 20.0)      # force
        n = dst.index("*MAT_ANISOTROPIC_ELASTIC")
        self.assertAlmostEqual(float(dst[n + 1][10:20]), 2.51e-9)   # RO
        self.assertAlmostEqual(float(dst[n + 1][20:30]), 5940.0)    # C11
        self.assertAlmostEqual(float(dst[n + 3][0:10]), 1180.0)     # C55
        self.assertAlmostEqual(float(dst[n + 4][0:10]), 100.0)      # XP
        self.assertAlmostEqual(float(dst[n + 4][30:40]), 1.0)       # A1
        self.assertEqual(dst[n + 5],
                         _at(src, "*MAT_ANISOTROPIC_ELASTIC", 5))   # V/D/BETA
        x = dst.index("*MAT_ADD_THERMAL_EXPANSION")
        self.assertEqual(dst[x + 1],
                         _at(src, "*MAT_ADD_THERMAL_EXPANSION", 1))  # nothing

    def test_thermal_isotropic_branches(self):
        def base(tgrlc, tgmult):
            return ("*KEYWORD\n*MAT_THERMAL_ISOTROPIC\n"
                    + F(1, 7830.0, tgrlc, tgmult, 0.0, 250000.0) + "\n"
                    + F(460.0, 46.0) + "\n*END\n")
        # TGRLC = 0: TGMULT IS the volumetric heat generation rate
        p = _write(base(0, "1.43000E7"), "a.k")
        convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        i = d.index("*MAT_THERMAL_ISOTROPIC")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 7.83e-9)     # TRO
        self.assertAlmostEqual(float(d[i + 1][30:40]), 14.3)        # PWR_VOL
        self.assertAlmostEqual(float(d[i + 1][50:60]), 2.5e11)      # HLAT
        self.assertAlmostEqual(float(d[i + 2][0:10]), 4.6e8)        # HC
        self.assertAlmostEqual(float(d[i + 2][10:20]), 46.0)        # TC x1
        # TGRLC != 0: TGMULT is a plain multiplier on the curve
        deck = base(210, 1.0).replace(
            "*END", "*DEFINE_CURVE\n" + F(210, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("0.0", "1.43000E7", w=20) + "\n*END")
        p = _write(deck, "b.k")
        convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        i = d.index("*MAT_THERMAL_ISOTROPIC")
        self.assertAlmostEqual(float(d[i + 1][30:40]), 1.0)         # untouched
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][20:40]), 14.3)        # the curve

    def test_spotweld_sign_encoded_resultants(self):
        deck = ("*KEYWORD\n*MAT_SPOTWELD_TITLE\nmild steel\n"
                + F(1, 7830.0, "2.10000E11", 0.3, "3.00000E8", "5.00000E10",
                    0.0, "1.00000E20") + "\n"
                + F(0.8, 5000.0, -7, 0.0, 250.0, 0.0, 0.0, 0.0) + "\n"
                "*DEFINE_CURVE\n"
                + F(7, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("100.0", "6000.0", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        d = _lines(out)
        i = d.index("*MAT_SPOTWELD_TITLE")
        self.assertAlmostEqual(float(d[i + 2][10:20]), 7.83e-9)     # RO
        self.assertAlmostEqual(float(d[i + 2][20:30]), 210000.0)    # E
        self.assertAlmostEqual(float(d[i + 2][40:50]), 300.0)       # SIGY
        self.assertAlmostEqual(float(d[i + 3][10:20]), 5000.0)      # NRR, F x1
        self.assertEqual(int(d[i + 3][20:30]), -7)                  # NRS curve
        self.assertAlmostEqual(float(d[i + 3][40:50]), 250000.0)    # MRR
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][20:40]), 6000.0)      # force

    def test_spotweld_negative_sigy_refused(self):
        deck = ("*KEYWORD\n*MAT_SPOTWELD\n"
                + F(1, 7830.0, "2.10000E11", 0.3, -12, "5.00000E10") + "\n"
                + F(0.8) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "SIGY"):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_inv_hyperbolic_sin_leaves_molar_pair_alone(self):
        deck = ("*KEYWORD\n*MAT_INV_HYPERBOLIC_SIN\n"
                + F(1, 2700.0, "6.90000E10", 0.3, 1260.0, "4.55000E-9",
                    0.0) + "\n"
                + F("1.72370E-7", 3.83, "4.50000E7", 1188000.0, 8.3144,
                    "1.00000E-6", 0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        d = _lines(out)
        i = d.index("*MAT_INV_HYPERBOLIC_SIN")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 2.7e-9)      # RO
        self.assertAlmostEqual(float(d[i + 1][20:30]), 69000.0)     # E
        self.assertAlmostEqual(float(d[i + 1][50:60]), 4.55e-3)     # HC x1e6
        self.assertAlmostEqual(float(d[i + 2][0:10]), 0.17237)      # ALPHA
        self.assertAlmostEqual(float(d[i + 2][10:20]), 3.83)        # N
        self.assertAlmostEqual(float(d[i + 2][20:30]), 4.5e7)       # A, 1/time
        self.assertAlmostEqual(float(d[i + 2][30:40]), 1188000.0)   # Q kept
        self.assertAlmostEqual(float(d[i + 2][40:50]), 8.3144)      # G kept
        self.assertTrue(any("Q/(G*T)" in n for n in ctx.notes), ctx.notes)

    def test_frazer_nash_coefficients_are_flags_under_fit(self):
        def body(c, lcid):
            return ("*KEYWORD\n*MAT_FRAZER_NASH_RUBBER_MODEL\n"
                    + F(1, 1254.0, 0.495, c, 0.0, 0.0, 0.0) + "\n"
                    + F(c, 0.0, c, c, 1.0, 0.9, -0.9, 0.0) + "\n"
                    + F(1.0, 1.0, 1.0, lcid) + "\n"
                    "*END\n")
        # direct branch: the coefficients are strain-energy densities
        p = _write(body("1.00000E6", 0), "direct.k")
        convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        i = d.index("*MAT_FRAZER_NASH_RUBBER_MODEL")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 1.254e-9)    # RO
        self.assertAlmostEqual(float(d[i + 1][30:40]), 1.0)         # C100
        self.assertAlmostEqual(float(d[i + 2][0:10]), 1.0)          # C110
        self.assertAlmostEqual(float(d[i + 3][0:10]), 1000.0)       # SGL
        # least-squares branch: the same fields are 0/1 inclusion flags
        deck = body(1.0, 2).replace(
            "*END", "*DEFINE_CURVE\n" + F(2, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("0.001", "20.0", w=20) + "\n*END")
        p = _write(deck, "fit.k")
        ctx = convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        i = d.index("*MAT_FRAZER_NASH_RUBBER_MODEL")
        self.assertAlmostEqual(float(d[i + 1][30:40]), 1.0)   # flag, unscaled
        self.assertAlmostEqual(float(d[i + 2][0:10]), 1.0)    # flag, unscaled
        self.assertAlmostEqual(float(d[i + 3][0:10]), 1000.0)  # SGL is a length
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][0:20]), 1.0)     # gauge change
        self.assertAlmostEqual(float(d[c + 2][20:40]), 20.0)   # force
        self.assertTrue(any("least-squares" in n for n in ctx.notes), ctx.notes)

    def test_low_density_foam_decay_constants_are_rates(self):
        deck = ("*KEYWORD\n*MAT_LOW_DENSITY_FOAM\n"
                + F(57, 240.0, "2.50000E5", 1, "4.00000E4", 0.5, 250.0,
                    0.1) + "\n"
                + F(0.0, 0.0, 0, "1.00000E5", 500.0, "1.00000E6", 0) + "\n"
                "*DEFINE_CURVE\n"
                + F(1, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("0.5", "1.00000E5", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        d = _lines(out)
        i = d.index("*MAT_LOW_DENSITY_FOAM")
        self.assertAlmostEqual(float(d[i + 1][60:70]), 0.25)   # BETA x1e-3
        self.assertAlmostEqual(float(d[i + 2][40:50]), 0.5)    # BETA1 x1e-3
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][0:20]), 0.5)     # strain
        self.assertAlmostEqual(float(d[c + 2][20:40]), 1.0e-4)  # stress -> GPa


class IcfdNewTests(unittest.TestCase):
    """*ICFD_BOUNDARY_PRESCRIBED_TEMP, *ICFD_INITIAL and *MESH_NODE."""

    def test_prescribed_temp_and_initial(self):
        deck = ("*KEYWORD\n"
                "*ICFD_BOUNDARY_PRESCRIBED_TEMP\n"
                + F(2, 4, 20.0, "1.00000E28", 0.0) + "\n"
                + F(3, 4, 60.0, "1.00000E28", 0.0) + "\n"
                "*ICFD_INITIAL\n"
                + F(0, 1.5, 0.0, 0.0, 20.0, 101325.0) + "\n"
                "*ICFD_DATABASE_TEMP\n"
                + F(4, 0.01) + "\n"
                "*MESH_NODE\n"
                "     330        1.281916       0.7851647             0.0"
                "       0       0\n"
                "*DEFINE_CURVE\n"
                + F(4, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("0.0", "1.0", w=20) + "\n"
                + F("10.0", "1.0", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        d = _lines(out)
        t = d.index("*ICFD_BOUNDARY_PRESCRIBED_TEMP")
        self.assertAlmostEqual(float(d[t + 1][20:30]), 20.0)   # SF kept
        self.assertAlmostEqual(float(d[t + 1][30:40]), 1e31)   # DEATH x1e3
        self.assertAlmostEqual(float(d[t + 2][20:30]), 60.0)   # 2nd card SF
        i = d.index("*ICFD_INITIAL")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 1.5)    # Vx m/s->mm/ms
        self.assertAlmostEqual(float(d[i + 1][40:50]), 20.0)   # T kept
        self.assertAlmostEqual(float(d[i + 1][50:60]), 1.01325e-4)  # P -> GPa
        b = d.index("*ICFD_DATABASE_TEMP")
        self.assertAlmostEqual(float(d[b + 1][10:20]), 10.0)   # DTOUT
        n = d.index("*MESH_NODE") + 1
        self.assertEqual(d[n][0:8], "     330")
        self.assertAlmostEqual(float(d[n][8:24]), 1281.916)
        self.assertEqual(d[n][56:72], "       0       0")      # TC/RC intact
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][0:20]), 0.0)
        self.assertAlmostEqual(float(d[c + 3][0:20]), 10000.0)  # time abscissa
        self.assertAlmostEqual(float(d[c + 3][20:40]), 1.0)     # TEMP ordinate

    def test_shared_temp_velocity_curve_refused(self):
        deck = ("*KEYWORD\n"
                "*ICFD_BOUNDARY_PRESCRIBED_VEL\n"
                + F(1, 1, 1, 2, -1.0, 0, "1.00000E28", 0.0) + "\n"
                "*ICFD_BOUNDARY_PRESCRIBED_TEMP\n"
                + F(2, 2, 20.0, "1.00000E28", 0.0) + "\n"
                "*DEFINE_CURVE\n"
                + F(2, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("0.0", "1.0", w=20) + "\n"
                + F("10.0", "1.0", w=20) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "conflicting dimension"):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_icfd_initial_dfunc_refused(self):
        deck = ("*KEYWORD\n*ICFD_INITIAL\n"
                + F(0, 11, 12, 13, 14, 15, 0, 1) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "DFUNC=1"):
            convert(p, SI, TON, p + ".o.k", self_check=False)


class FreqDomainNewTests(unittest.TestCase):
    """*FREQUENCY_DOMAIN_SSD / FRF / RESPONSE_SPECTRUM / ACOUSTIC_*."""

    SSD = ("*KEYWORD\n"
           "*FREQUENCY_DOMAIN_SSD\n"
           + F(1, 100, 0.0, 2000.0, 0, 0, 0, 0) + "\n"
           + F(0.01, 0, 0, 3.0, 0.002, 0) + "\n"
           + F("", 0, 0, 0, 0, 0, 0, 0) + "\n"
           + F(131, 0, 3, 3, 100, 200, 1.0, 0) + "\n"
           "*DEFINE_CURVE\n"
           + F(100, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
           + F("10.0", "9.81", w=20) + "\n"
           "*DEFINE_CURVE\n"
           + F(200, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
           + F("10.0", "90.0", w=20) + "\n"
           "*DATABASE_FREQUENCY_ASCII_NODOUT_SSD\n"
           + F(1.0, 500.0, 500, 0, 0) + "\n"
           "*DAMPING_FREQUENCY_RANGE\n"
           + F(0.01, 30.0, 300.0, 0) + "\n"
           "*END\n")

    def test_ssd_frequencies_damping_and_curves(self):
        p = _write(self.SSD)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        d = _lines(out)
        i = d.index("*FREQUENCY_DOMAIN_SSD")
        self.assertAlmostEqual(float(d[i + 1][30:40]), 2.0)     # FNMAX
        self.assertAlmostEqual(float(d[i + 2][0:10]), 0.01)     # DAMPF ratio
        self.assertAlmostEqual(float(d[i + 2][30:40]), 0.003)   # DMPMAS alpha
        self.assertAlmostEqual(float(d[i + 2][40:50]), 2.0)     # DMPSTF beta
        self.assertEqual(d[i + 3][0:10], "          ")          # Card 3 blank
        self.assertEqual(int(d[i + 4][40:50]), 100)             # LC1 id kept
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][0:20]), 0.01)     # freq x1e-3
        self.assertAlmostEqual(float(d[c + 2][20:40]), 9.81e-3)  # VAD=3 accel
        c2 = d.index("*DEFINE_CURVE", c + 1)
        self.assertAlmostEqual(float(d[c2 + 2][20:40]), 90.0)   # phase, deg
        a = d.index("*DATABASE_FREQUENCY_ASCII_NODOUT_SSD")
        self.assertAlmostEqual(float(d[a + 1][0:10]), 0.001)    # FMIN
        self.assertAlmostEqual(float(d[a + 1][10:20]), 0.5)     # FMAX
        self.assertEqual(int(d[a + 1][20:30]), 500)             # NFREQ
        g = d.index("*DAMPING_FREQUENCY_RANGE")
        self.assertAlmostEqual(float(d[g + 1][0:10]), 0.01)     # CDAMP
        self.assertAlmostEqual(float(d[g + 1][10:20]), 0.03)    # FLOW
        self.assertAlmostEqual(float(d[g + 1][20:30]), 0.3)     # FHIGH

    def test_ssd_shifted_card3_refused(self):
        bad = self.SSD.replace(F("", 0, 0, 0, 0, 0, 0, 0),
                               F(200.0, 1500.0, 14, 0, 0, 0, 0, 0))
        p = _write(bad)
        with self.assertRaisesRegex(ConvertError, "blank first column"):
            convert(p, SI, KGMM, p + ".o.k", self_check=False)

    def test_ssd_erp_reference_power(self):
        deck = ("*KEYWORD\n*FREQUENCY_DOMAIN_SSD_ERP\n"
                + F(1, 20, 0.0, 0.0, 0, 0, 0, 0) + "\n"
                + F(0.01, 0, 0, 0.0, 0.0, 0) + "\n"
                + F("", 0, 0, 1, 0, 0, 0, 0) + "\n"
                + F(1.21, 340.0, 1.0, "1.0E-12", 0) + "\n"
                + F(1, 2) + "\n"
                + F(0, 0, 1, 3, 0, 0, 1.0, 0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        d = _lines(out)
        i = d.index("*FREQUENCY_DOMAIN_SSD_ERP")
        self.assertAlmostEqual(float(d[i + 4][0:10]), 1.21e-12)   # RO
        self.assertAlmostEqual(float(d[i + 4][10:20]), 340000.0)  # C
        self.assertAlmostEqual(float(d[i + 4][20:30]), 1.0)       # ERPRLF
        self.assertAlmostEqual(float(d[i + 4][30:40]), 1e-9)      # ERPREF x1e3
        self.assertEqual(d[i + 5], F(1, 2))                       # PID PTYP

    def test_frf_and_response_spectrum(self):
        deck = ("*KEYWORD\n*FREQUENCY_DOMAIN_FRF\n"
                + F(25, 0, 0, 3, 0, 20000.0, 0, 0) + "\n"
                + F(0.01, 0, 0, 3.0, 0.002) + "\n"
                + F(32, 0, 2, 1, 0, 0) + "\n"
                + F(100.0, 10000.0, 100, 0, 0, 0, 0) + "\n"
                "*FREQUENCY_DOMAIN_RESPONSE_SPECTRUM\n"
                + F(1, 20, 0.0, 2000.0, 0, 1, 0, 0) + "\n"
                + F(0.001, 0, 0, 3.0, 0.002) + "\n"
                + F(1, 3, 100, 0.0, 0, 0, 1, 0) + "\n"
                "*DEFINE_CURVE\n"
                + F(100, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("0.2", "1.26", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        d = _lines(out)
        f = d.index("*FREQUENCY_DOMAIN_FRF")
        self.assertAlmostEqual(float(d[f + 1][50:60]), 20.0)    # FNMAX
        self.assertAlmostEqual(float(d[f + 2][30:40]), 0.003)   # DMPMAS
        self.assertAlmostEqual(float(d[f + 2][40:50]), 2.0)     # DMPSTF
        self.assertAlmostEqual(float(d[f + 4][0:10]), 0.1)      # FMIN
        self.assertAlmostEqual(float(d[f + 4][10:20]), 10.0)    # FMAX
        r = d.index("*FREQUENCY_DOMAIN_RESPONSE_SPECTRUM")
        self.assertAlmostEqual(float(d[r + 1][30:40]), 2.0)     # FNMAX
        self.assertAlmostEqual(float(d[r + 2][30:40]), 0.003)   # DMPMAS
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][0:20]), 2.0e-4)   # freq x1e-3
        self.assertAlmostEqual(float(d[c + 2][20:40]), 1.26e-3)  # LCTYP=1 acc

    def test_frf_lcfreq_refused(self):
        deck = ("*KEYWORD\n*FREQUENCY_DOMAIN_FRF\n"
                + F(25, 0, 0, 3, 0, 20000.0, 0, 0) + "\n"
                + F(0.01, 0, 0, 0.0, 0.0) + "\n"
                + F(32, 0, 2, 1, 0, 0) + "\n"
                + F(100.0, 10000.0, 100, 0, 55, 0, 0) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "LCFREQ=55"):
            convert(p, SI, KGMM, p + ".o.k", self_check=False)

    BEM = ("*KEYWORD\n"
           "*FREQUENCY_DOMAIN_ACOUSTIC_BEM_HALF_SPACE\n"
           + F(1.21, 343.0, 1.0, 300.0, 300, 0.005, 0.0, "2.00000E-5") + "\n"
           + F(1, 1, 0, 0, 0, 0, 0, 1) + "\n"
           + F(3, 2000, "1.00000E-6", 8, "1.00000E-6", "1.00000E-6", 200,
               2) + "\n"
           + F("", 1, 0, 0, 0, 0, 0, 0) + "\n"
           + F(1, 1, 1, 3, 2, 0) + "\n"
           + F(1, 1.0) + "\n"
           "*DEFINE_CURVE\n"
           + F(2, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
           + F("100.0", "0.05", w=20) + "\n"
           "*FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_SPHERE\n"
           + F(1, 7.0, 13, 0.0, 0.0, 0.0, 0, 0) + "\n"
           "*END\n")

    def test_acoustic_bem_iunits_remap(self):
        p = _write(self.BEM)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        d = _lines(out)
        i = d.index("*FREQUENCY_DOMAIN_ACOUSTIC_BEM_HALF_SPACE")
        self.assertAlmostEqual(float(d[i + 1][0:10]), 1.21e-12)   # RO
        self.assertAlmostEqual(float(d[i + 1][10:20]), 343000.0)  # C
        self.assertAlmostEqual(float(d[i + 1][20:30]), 1.0)       # FMIN
        self.assertAlmostEqual(float(d[i + 1][50:60]), 0.005)     # DTOUT
        self.assertAlmostEqual(float(d[i + 1][70:80]), 2.0e-11)   # PREF
        self.assertEqual(int(d[i + 2][70:80]), 4)                 # IUNITS 1->4
        self.assertEqual(d[i + 4][0:10], "          ")            # Card4 blank
        self.assertEqual(int(d[i + 4][10:20]), 1)                 # NBC
        self.assertEqual(d[i + 6], F(1, 1.0))                     # Card 6
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][20:40]), 50.0)   # BEMTYP=3 vel
        s = d.index("*FREQUENCY_DOMAIN_ACOUSTIC_FRINGE_PLOT_SPHERE")
        self.assertAlmostEqual(float(d[s + 1][10:20]), 7000.0)  # radius
        self.assertEqual(int(d[s + 1][20:30]), 13)              # mesh density

    def test_acoustic_bem_unsupported_target_refused(self):
        p = _write(self.BEM)
        with self.assertRaisesRegex(ConvertError, "IUNITS"):
            convert(p, SI, parse_system("g-mm-ms"), p + ".o.k",
                    self_check=False)

    def test_acoustic_fem_impedance_curve(self):
        deck = ("*KEYWORD\n*FREQUENCY_DOMAIN_ACOUSTIC_FEM\n"
                + F(1.21, 343.0, 10.0, 500.0, 99, 0.0, 0.0,
                    "2.00000E-5") + "\n"
                + F("", 0, 0, 0) + "\n"
                + F(1, 0) + "\n"
                + F(1, 2, 11, 3, 10, 30, 1.0, 0) + "\n"
                + F(2, 2, 42, 0, 20, 21, 1.0, 0) + "\n"
                + F(200, 1, 0, 0) + "\n"
                "*DEFINE_CURVE\n" + F(10, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("100.0", "0.02", w=20) + "\n"
                "*DEFINE_CURVE\n" + F(30, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("100.0", "45.0", w=20) + "\n"
                "*DEFINE_CURVE\n" + F(20, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("100.0", "415.0", w=20) + "\n"
                "*DEFINE_CURVE\n" + F(21, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
                + F("100.0", "-20.0", w=20) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        d = _lines(out)
        i = d.index("*FREQUENCY_DOMAIN_ACOUSTIC_FEM")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 343000.0)   # C
        self.assertEqual(d[i + 6], F(200, 1, 0, 0))                # Card 5
        fimp = float(factor((1, -2, -1), SI, TON))
        c10 = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c10 + 2][20:40]), 20.0)     # velocity
        c30 = d.index("*DEFINE_CURVE", c10 + 1)
        self.assertAlmostEqual(float(d[c30 + 2][20:40]), 45.0)     # phase, deg
        c20 = d.index("*DEFINE_CURVE", c30 + 1)
        self.assertAlmostEqual(float(d[c20 + 2][20:40]), 415.0 * fimp)
        c21 = d.index("*DEFINE_CURVE", c20 + 1)
        self.assertAlmostEqual(float(d[c21 + 2][20:40]), -20.0 * fimp)

    def test_incident_wave_plane_vs_spherical(self):
        plane = ("*KEYWORD\n*FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE\n"
                 + F(1, 100.0, 1.0, 0.0, 0.0) + "\n*END\n")
        p = _write(plane, "plane.k")
        convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        i = d.index("*FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 1.0e-4)   # MAG pressure
        self.assertAlmostEqual(float(d[i + 1][20:30]), 1.0)      # cosine kept
        # TYPE=2: Remark 2's p(r) = A*exp(-ikr)/r makes MAG a pressure TIMES
        # a length, so it picks up the length factor the plane wave does not,
        # and XC/YC/ZC are point-source coordinates rather than cosines
        sph = ("*KEYWORD\n*FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE\n"
               + F(2, 1.0, 4.0, 0.0, 0.0) + "\n*END\n")
        p = _write(sph, "sph.k")
        ctx = convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        i = d.index("*FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE")
        self.assertAlmostEqual(float(d[i + 1][10:20]), 1.0e-3)  # Pa*m -> MPa*mm
        self.assertAlmostEqual(float(d[i + 1][20:30]), 4000.0)  # XC is a length
        self.assertTrue(any("pressure*length" in w for w in ctx.warnings),
                        ctx.warnings)

    def test_incident_wave_magnitude_curve_matches_the_scalar(self):
        # MAG < 0 names a frequency-dependent magnitude curve; its ordinate
        # must take the same dimension the scalar branch would have, or the
        # two branches disagree about what MAG means
        deck = ("*KEYWORD\n*FREQUENCY_DOMAIN_ACOUSTIC_INCIDENT_WAVE\n"
                + F(2, -7, 4.0, 0.0, 0.0) + "\n"
                "*DEFINE_CURVE\n" + F(7, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
                + F("100.0", w=20) + F("1.0", w=20) + "\n*END\n")
        p = _write(deck, "sphcurve.k")
        convert(p, SI, TON, p + ".o.k", self_check=False)
        d = _lines(p + ".o.k")
        j = d.index(F(7, 0, 1.0, 1.0, 0.0, 0.0, 0, 0))
        self.assertAlmostEqual(float(d[j + 1][0:20]), 100.0)    # frequency
        self.assertAlmostEqual(float(d[j + 1][20:40]), 1.0e-3)  # Pa*m -> MPa*mm


class SaleTests(unittest.TestCase):
    """S-ALE keywords and the *DEFINE_VECTOR / *DEFINE_BOX context guards."""

    DECK = ("*KEYWORD\n"
            "*ALE_STRUCTURED_MESH\n"
            + F(1, 100, 1000000, 1000000, 0, 0, 0, "1.00000E16") + "\n"
            + F(1001, 1002, 1003, 0, 0) + "\n"
            "*ALE_STRUCTURED_MESH_CONTROL_POINTS\n"
            + F(1001, 0, 1, 1.0, 0, 0.01) + "\n"
            + F(1, 0.0, 0.005, w=20) + "\n"
            + F(21, 0.2, 0.005, w=20) + "\n"
            "*ALE_STRUCTURED_MESH_VOLUME_FILLING\n"
            + F(1, 0, "air", 0, 4, 0, 0, 7) + "\n"
            + F("CYLINDER", 0, 11, 12, 0.03, 0.05) + "\n"
            + F(1, 0, "water", 0, 4, 0, 0, 0) + "\n"
            + F("BOXCPT", 0, 9) + "\n"
            "*DEFINE_VECTOR\n"
            + F(7, 100.0, -20.0, 0.0, 0.0, 0.0, 0.0) + "\n"
            "*DEFINE_VECTOR\n"
            + F(8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) + "\n"
            "*DEFINE_BOX\n"
            + F(9, 8, 15, 8, 15, 8, 15) + "\n"
            "*DEFINE_BOX\n"
            + F(10, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5) + "\n"
            "*ALE_REFERENCE_SYSTEM_GROUP\n"
            + F(1, 0, 4, 0, 0, 7, 0, 0) + "\n"
            + F(0.1, 0.2, 0.3, 1.5, 0.0, "", 0.05, 1) + "\n"
            "*ALE_STRUCTURED_FSI\n"
            + F(1, 2, 0, 0, 0, 0, 0, 0) + "\n"
            + F(0.001, "1.00000E10", 0.1, 0.3, 0, 0, 0.0, 0.0) + "\n"
            "*BOUNDARY_SALE_MESH_FACE\n"
            + F("NOFLOW", 1, 1, 0, 0, 0, 0, 0) + "\n"
            + F("PRES", 1, 0, 44, 0, 0, 0, 0) + "\n"
            "*DEFINE_CURVE\n"
            + F(44, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("0.0", "101325.0", w=20) + "\n"
            "*INITIAL_VOLUME_FRACTION_GEOMETRY\n"
            + F(1, 1, 1, 8) + "\n"
            + F(3, 0, 2, 12.0, 0.0, -3.0) + "\n"
            + F(0.025, 0.02, 0.0, 0.0, -1.0, 0.0) + "\n"
            + F(6, 0, 3, 0.0, 0.0, 0.0) + "\n"
            + F(0.1, 0.1, 0.1, 0.05) + "\n"
            "*END\n")

    def test_sale_geometry_and_context_guards(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        d = _lines(out)
        m = d.index("*ALE_STRUCTURED_MESH")
        self.assertEqual(int(d[m + 1][20:30]), 1000000)       # NBID counter
        cp = d.index("*ALE_STRUCTURED_MESH_CONTROL_POINTS")
        self.assertAlmostEqual(float(d[cp + 1][30:40]), 1.0)   # SFO stays
        self.assertAlmostEqual(float(d[cp + 1][50:60]), 10.0)  # OFFO x1e3
        self.assertEqual(int(d[cp + 2][0:20]), 1)              # N index
        self.assertAlmostEqual(float(d[cp + 3][20:40]), 200.0)   # X x1e3
        self.assertAlmostEqual(float(d[cp + 3][40:60]), 5.0)     # ICASE=1 XL
        vf = d.index("*ALE_STRUCTURED_MESH_VOLUME_FILLING")
        self.assertEqual(d[vf + 1][20:30], "       air")       # AMMG name
        self.assertAlmostEqual(float(d[vf + 2][40:50]), 30.0)  # cylinder R1
        self.assertAlmostEqual(float(d[vf + 2][50:60]), 50.0)  # cylinder R2
        # the VOLUME_FILLING VID vector holds VELOCITIES, not coordinates
        v7 = d.index("*DEFINE_VECTOR")
        self.assertAlmostEqual(float(d[v7 + 1][10:20]), 100000.0)
        self.assertAlmostEqual(float(d[v7 + 1][20:30]), -20000.0)
        v8 = d.index("*DEFINE_VECTOR", v7 + 1)
        self.assertAlmostEqual(float(d[v8 + 1][10:20]), 100.0)   # plain length
        self.assertAlmostEqual(float(d[v8 + 1][40:50]), 400.0)
        # the GEOM=BOXCPT box holds control-point INDICES
        b9 = d.index("*DEFINE_BOX")
        self.assertEqual(int(d[b9 + 1][10:20]), 8)
        self.assertEqual(int(d[b9 + 1][20:30]), 15)
        b10 = d.index("*DEFINE_BOX", b9 + 1)
        self.assertAlmostEqual(float(d[b10 + 1][20:30]), 500.0)  # real box
        self.assertTrue(any("BOXCPT" in n for n in ctx.notes), ctx.notes)
        self.assertTrue(any("VELOCITIES" in n for n in ctx.notes), ctx.notes)
        rs = d.index("*ALE_REFERENCE_SYSTEM_GROUP")
        self.assertAlmostEqual(float(d[rs + 2][0:10]), 100.0)   # XC
        self.assertAlmostEqual(float(d[rs + 2][30:40]), 1.5)    # EXPLIM ratio
        self.assertAlmostEqual(float(d[rs + 2][60:70]), 0.05)   # FRCPAD
        fs = d.index("*ALE_STRUCTURED_FSI")
        self.assertAlmostEqual(float(d[fs + 2][20:30]), 0.1)    # PFAC ratio
        c = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c + 2][20:40]), 0.101325)   # PRES face
        iv = d.index("*INITIAL_VOLUME_FRACTION_GEOMETRY")
        self.assertAlmostEqual(float(d[iv + 2][30:40]), 12000.0)  # VX
        self.assertAlmostEqual(float(d[iv + 3][0:10]), 25.0)      # X0 plane
        self.assertAlmostEqual(float(d[iv + 3][40:50]), -1.0)     # cosine kept
        self.assertAlmostEqual(float(d[iv + 5][30:40]), 50.0)     # sphere R0

    def test_sale_times_scale_to_ms(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        d = _lines(out)
        m = d.index("*ALE_STRUCTURED_MESH")
        self.assertAlmostEqual(float(d[m + 1][70:80]), 1e19)    # TDEATH
        fs = d.index("*ALE_STRUCTURED_FSI")
        self.assertAlmostEqual(float(d[fs + 2][0:10]), 1.0)     # START
        self.assertAlmostEqual(float(d[fs + 2][10:20]), 1e13)   # END

    def test_ale_ref_group_undocumented_column_warns(self):
        deck = ("*KEYWORD\n*ALE_REFERENCE_SYSTEM_GROUP\n"
                + F(1, 0, 4, 0, 0, 7, 0, 0) + "\n"
                + F(0.1, 0.2, 0.3, 1.5, 0.0, 12.5, 0.05, 1) + "\n*END\n")
        p = _write(deck)
        ctx = convert(p, SI, TON, p + ".o.k", self_check=False)
        self.assertTrue(any("SMOOTHVMX" in w for w in ctx.warnings),
                        ctx.warnings)

    def test_volume_filling_unknown_geom_refused(self):
        deck = ("*KEYWORD\n*ALE_STRUCTURED_MESH_VOLUME_FILLING\n"
                + F(1, 0, "air", 0, 4, 0, 0, 0) + "\n"
                + F("SPHERE", 0, 11, 0.03) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "SPHERE"):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_volume_fraction_define_function_refused(self):
        deck = ("*KEYWORD\n*INITIAL_VOLUME_FRACTION_GEOMETRY\n"
                + F(1, 1, 1, 8) + "\n"
                + F(7, 0, 2, 0.0, 0.0, 0.0) + "\n"
                + F(1234) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "CNTTYP=7"):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_sale_mesh_face_ambient_packed_curve_refused(self):
        deck = ("*KEYWORD\n*BOUNDARY_SALE_MESH_FACE\n"
                + F("AMBIENT", 1, 20301, 0, 0, 0, 0, 0) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "AMBIENT"):
            convert(p, SI, TON, p + ".o.k", self_check=False)


class AirbagMiscTests(unittest.TestCase):
    """*AIRBAG_SIMPLE_*, *CASE_*, thermal loads, geometric rigid walls and
    the CONTACT MORTAR / MPP routing."""

    DECK = ("*KEYWORD\n"
            "*CASE_BEGIN_1\n"
            "*AIRBAG_SIMPLE_AIRBAG_MODEL_ID\n"
            + F(2000001) + "airbag\n"
            + F(2000003, 1, 0, 1.0, 1.0, 0.002, 5.0, 0.0) + "\n"
            + F(717.0, 1004.0, 300.0, 20, 0.7, 0.01, 101325.0, 1.204) + "\n"
            + F(0, 293.0, 0.0, 0.0, 0.0, 0.0) + "\n"
            "*AIRBAG_SIMPLE_PRESSURE_VOLUME_ID\n"
            + F(1) + "tyre\n"
            + F(3, 1, 0, 0.0, 0.0, 0.0, 150.0, 0.0) + "\n"
            + F(300000.0, 1.0, 0, 0) + "\n"
            "*CASE_END_1\n"
            "*DEFINE_PLANE\n"
            + F(1, 1.0, 0.0, -1.5, 0.0, 1.0, -1.5, 0) + "\n"
            + F(0.0, 0.0, -1.5) + "\n"
            "*DEFINE_COORDINATE_VECTOR\n"
            + F(1, 0.99144, -0.13053, 0.0, 0.0, 0.0, 1.0, 0) + "\n"
            "*INITIAL_TEMPERATURE_SET\n"
            + F(2, 20.0, 0) + "\n"
            "*INTERFACE_LINKING_SEGMENT\n"
            + F(1, 1, 0.02) + "\n"
            "*INTERFACE_COMPONENT_SEGMENT\n"
            + F(1, 0, 0) + "\n"
            "*LOAD_THERMAL_LOAD_CURVE\n"
            + F(2, 0) + "\n"
            "*LOAD_THERMAL_VARIABLE\n"
            + F(0, 0, 0) + "\n"
            + F(1.0, 0.0, 3, 0.0, 0.0, 0, 0, 0) + "\n"
            "*DEFINE_CURVE\n"
            + F(2, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("1.0", "300.0", w=20) + "\n"
            "*DEFINE_CURVE\n"
            + F(3, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("0.001", "25.0", w=20) + "\n"
            "*DEFINE_CURVE\n"
            + F(20, 0, 1.0, 1.0, 0.0, 0.0) + "\n"
            + F("0.0", "0.5", w=20) + "\n"
            "*RIGIDWALL_GEOMETRIC_FLAT_DISPLAY_ID\n"
            + F(1) + "rigidwall\n"
            + F(1, 0, 0, 0.0, 0.0) + "\n"
            + F(0.56, -0.05, -0.005, 0.56, -0.05, 0.0, 0.0) + "\n"
            + F(0.56, 0.001, -0.005, 0.07, 0.07) + "\n"
            + F(0, "1.00000E-9", "1.00000E-4", 0.3) + "\n"
            "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_MORTAR_ID\n"
            + F(1) + "mortar\n"
            + F(1, 2, 3, 3, 0, 0, 0, 0) + "\n"
            + F(0.3, 0.3, 0.05, "1.00000E6", 20.0, 0, 0.0,
                "1.00000E20") + "\n"
            + F(1.0, 1.0, 0.002, 0.003, 1.0, 1.0, 1.0, 1.0) + "\n"
            + F(0, 0.1, 0, 1.025, 4, 2, 0, 1) + "\n"
            + F(0.004, 0, 0, 0, 0, 0, 0.005, "1.00000E7") + "\n"
            "*END\n")

    def test_airbag_and_misc_fields(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        src, d = _lines(p), _lines(out)
        AB = "*AIRBAG_SIMPLE_AIRBAG_MODEL_ID"
        a = d.index(AB)
        self.assertEqual(d[a + 1], _at(src, AB, 1))               # id/title
        self.assertAlmostEqual(float(d[a + 2][50:60]), 2.0e6)     # VINI x1e9
        self.assertAlmostEqual(float(d[a + 2][60:70]), 5.0)       # MWD (1/t)
        self.assertAlmostEqual(float(d[a + 3][0:10]), 7.17e8)     # CV x1e6
        self.assertAlmostEqual(float(d[a + 3][10:20]), 1.004e9)   # CP
        self.assertAlmostEqual(float(d[a + 3][20:30]), 300.0)     # T kept
        self.assertAlmostEqual(float(d[a + 3][50:60]), 10000.0)   # AREA x1e6
        self.assertAlmostEqual(float(d[a + 3][60:70]), 0.101325)  # PE
        self.assertAlmostEqual(float(d[a + 3][70:80]), 1.204e-12)  # RO
        self.assertEqual(d[a + 4], _at(src, AB, 4))   # CV != 0 -> Card 4b
        v = d.index("*AIRBAG_SIMPLE_PRESSURE_VOLUME_ID")
        self.assertAlmostEqual(float(d[v + 2][60:70]), 150.0)     # MWD
        self.assertAlmostEqual(float(d[v + 3][0:10]), 0.3)        # CN pressure
        self.assertAlmostEqual(float(d[v + 3][10:20]), 1.0)       # BETA
        for kw in ("*CASE_BEGIN_1", "*CASE_END_1",
                   "*DEFINE_COORDINATE_VECTOR",
                   "*INTERFACE_COMPONENT_SEGMENT"):
            self.assertIn(kw, d)
        cv = "*DEFINE_COORDINATE_VECTOR"
        self.assertEqual(d[d.index(cv) + 1], src[src.index(cv) + 1])
        pl = d.index("*DEFINE_PLANE")
        self.assertAlmostEqual(float(d[pl + 1][30:40]), -1500.0)  # Z1
        self.assertAlmostEqual(float(d[pl + 2][20:30]), -1500.0)  # Z3
        it = "*INITIAL_TEMPERATURE_SET"
        self.assertEqual(d[d.index(it) + 1], src[src.index(it) + 1])
        rw = d.index("*RIGIDWALL_GEOMETRIC_FLAT_DISPLAY_ID")
        self.assertAlmostEqual(float(d[rw + 3][0:10]), 560.0)     # XT
        self.assertAlmostEqual(float(d[rw + 3][60:70]), 0.0)      # FRIC
        self.assertAlmostEqual(float(d[rw + 4][30:40]), 70.0)     # LENL
        self.assertAlmostEqual(float(d[rw + 5][10:20]), 1.0e-18)  # display RO
        self.assertAlmostEqual(float(d[rw + 5][20:30]), 1.0e-10)  # display E
        ct = d.index("*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_MORTAR_ID")
        self.assertAlmostEqual(float(d[ct + 3][20:30]), 5.0e-5)   # DC (1/vel)
        self.assertAlmostEqual(float(d[ct + 3][30:40]), 1.0)      # VC stress
        self.assertAlmostEqual(float(d[ct + 4][20:30]), 2.0)      # SAST
        cts = src.index("*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_MORTAR_ID")
        self.assertEqual(d[ct + 5], src[cts + 5])                 # Card A
        # Mortar Optional Card B: PENMAX is a penetration DISTANCE
        self.assertAlmostEqual(float(d[ct + 6][0:10]), 4.0)       # PENMAX
        self.assertAlmostEqual(float(d[ct + 6][60:70]), 5.0)      # SLDTHK
        self.assertAlmostEqual(float(d[ct + 6][70:80]), 10.0)     # SLDSTF
        self.assertFalse(ctx.unknown, ctx.unknown)

    def test_thermal_curve_abscissae_scale_to_ms(self):
        p = _write(self.DECK)
        out = p + ".o.k"
        convert(p, SI, KGMM, out, self_check=False)
        d = _lines(out)
        i = d.index("*INTERFACE_LINKING_SEGMENT")
        self.assertAlmostEqual(float(d[i + 1][20:30]), 20.0)     # DEATH
        c2 = d.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(d[c2 + 2][0:20]), 1000.0)   # time
        self.assertAlmostEqual(float(d[c2 + 2][20:40]), 300.0)   # temperature
        c3 = d.index("*DEFINE_CURVE", c2 + 1)
        self.assertAlmostEqual(float(d[c3 + 2][0:20]), 1.0)      # time
        self.assertAlmostEqual(float(d[c3 + 2][20:40]), 25.0)    # temperature

    def test_airbag_foreign_unit_system_refused(self):
        deck = ("*KEYWORD\n*AIRBAG_SIMPLE_PRESSURE_VOLUME\n"
                + F(3, 1, 0, 2.0, 1.0, 0.0, 150.0, 0.0) + "\n"
                + F(300000.0, 1.0, 0, 0) + "\n*END\n")
        p = _write(deck)
        with self.assertRaisesRegex(ConvertError, "VSCA"):
            convert(p, SI, TON, p + ".o.k", self_check=False)

    def test_contact_mpp_cards_shift_the_plan(self):
        deck = ("*KEYWORD\n*CONTACT_AUTOMATIC_NODES_TO_SURFACE_MPP_ID\n"
                + F(1) + "mpp contact\n"
                + F(0, 200, 0, 3, 2, 1.0005, 0, 0) + "\n"
                + F(1, 2, 3, 3, 0, 0, 0, 0) + "\n"
                + F(0.3, 0.3, 0.05, "1.00000E6", 20.0, 0, 0.0,
                    "1.00000E20") + "\n"
                + F(1.0, 1.0, 0.002, 0.003, 1.0, 1.0, 1.0, 1.0) + "\n*END\n")
        p = _write(deck)
        out = p + ".o.k"
        ctx = convert(p, SI, TON, out, self_check=False)
        src, d = _lines(p), _lines(out)
        MPP = "*CONTACT_AUTOMATIC_NODES_TO_SURFACE_MPP_ID"
        i = d.index(MPP)
        self.assertEqual(d[i + 2], _at(src, MPP, 2))          # MPP card
        self.assertEqual(d[i + 3], _at(src, MPP, 3))          # Card 1 ids
        self.assertAlmostEqual(float(d[i + 4][20:30]), 5.0e-5)   # Card 2 DC
        self.assertAlmostEqual(float(d[i + 4][30:40]), 1.0)      # Card 2 VC
        self.assertAlmostEqual(float(d[i + 5][20:30]), 2.0)      # Card 3 SAST
        self.assertFalse(ctx.unknown, ctx.unknown)


if __name__ == "__main__":
    unittest.main()
