"""T6 tests: the k_examples_db corpus batch of 2026-08-23 (airbags,
seatbelts, viscoelastic / Ogden / cable / discrete-beam materials, flux,
porosity leakage, permeability, SPG/EFG sections, whitelist additions)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import ConvertError, convert, parse_system

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")
KMM = parse_system("kg-mm-ms")


def F(*vals, w=10):
    return "".join(str(v).rjust(w) for v in vals)


def _write(text, name="deck.k"):
    d = tempfile.mkdtemp(prefix="kunit_t6_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _conv(body, src=SI):
    p = _write("*KEYWORD\n" + body + "*END\n")
    ctx = convert(p, src, TON, p + ".o.k", self_check=False)
    with open(p + ".o.k", newline="") as fh:
        return fh.read().split("\n"), ctx


def _after(lines, kw, n):
    return lines[lines.index(kw) + n]


def fld(line, i, w=10):
    return float(line[i * w:(i + 1) * w])


CURVE = "*DEFINE_CURVE\n" + F(7) + "\n" + F(0.0, 0.0, w=20) + "\n" + F(1.0, 2.0, w=20) + "\n"


class MaterialTests(unittest.TestCase):
    def test_mat_viscoelastic(self):
        lines, _ = _conv("*MAT_VISCOELASTIC\n" + F(1, 7850.0, "2.0E11", "8.0E10", "7.0E10", 100.0) + "\n")
        c = _after(lines, "*MAT_VISCOELASTIC", 1)
        self.assertAlmostEqual(fld(c, 1), 7.85e-9, delta=1e-14)
        self.assertAlmostEqual(fld(c, 2), 2.0e5)
        self.assertAlmostEqual(fld(c, 4), 7.0e4)
        self.assertAlmostEqual(fld(c, 5), 100.0)
        with self.assertRaisesRegex(ConvertError, "temperature-curve"):
            _conv("*MAT_VISCOELASTIC\n" + F(1, 7850.0, -3, "8.0E10", "7.0E10", 100.0) + "\n")

    def test_mat_cable_discrete_beam(self):
        lines, _ = _conv("*MAT_CABLE_DISCRETE_BEAM\n" + F(1, 7850.0, "2.0E11", 0, 500.0, 0.5, 0.1, 1) + "\n"
                         + F(0, 0.2, 1.0, 0.1, 2000.0) + "\n")
        c = _after(lines, "*MAT_CABLE_DISCRETE_BEAM", 1)
        self.assertAlmostEqual(fld(c, 2), 2.0e5)          # modulus
        self.assertAlmostEqual(fld(c, 4), 500.0)          # force
        self.assertAlmostEqual(fld(_after(lines, "*MAT_CABLE_DISCRETE_BEAM", 2), 4), 2000.0)
        lines, _ = _conv("*MAT_CABLE_DISCRETE_BEAM\n" + F(1, 7850.0, -5000.0, 0, 0, 0, 0, 0) + "\n")
        self.assertAlmostEqual(fld(_after(lines, "*MAT_CABLE_DISCRETE_BEAM", 1), 2), -5.0)  # stiffness N/m -> N/mm

    def test_mat_general_viscoelastic(self):
        body = ("*MAT_GENERAL_VISCOELASTIC\n" + F(1, 1100.0, "2.0E9", 0, 0, 293.0, 17.4, 51.6) + "\n"
                + F(7, 0, 0.0, 0.0, 0, 0, 0, 0) + "\n" + F("1.0E6", 10.0, "1.0E7", 0.1) + "\n" + CURVE)
        lines, ctx = _conv(body)
        c1 = _after(lines, "*MAT_GENERAL_VISCOELASTIC", 1)
        self.assertAlmostEqual(fld(c1, 2), 2.0e3)
        self.assertAlmostEqual(fld(c1, 7), 51.6)          # temperature untouched
        c3 = _after(lines, "*MAT_GENERAL_VISCOELASTIC", 3)
        self.assertAlmostEqual(fld(c3, 0), 1.0)           # Pa -> MPa
        self.assertAlmostEqual(fld(c3, 1), 10.0)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[20:40]), 2.0e-6)  # relaxation stress ordinate

    def test_mat_ogden_rubber(self):
        body = ("*MAT_OGDEN_RUBBER\n" + F(1, 1100.0, -0.49, 0, 0, "1.0E6", 1000.0, 0) + "\n"
                + F(0, 0, 0, 0, 0, "2.0E-8", 0, 0) + "\n"
                + F("3.0E5", "2.0E4", 0, 0, 0, 0, 0, 0) + "\n" + F(1.3, 5.0) + "\n"
                + F("1.0E5", 100.0, 0) + "\n")
        lines, _ = _conv(body)
        c1 = _after(lines, "*MAT_OGDEN_RUBBER", 1)
        self.assertAlmostEqual(fld(c1, 5), 1.0)
        self.assertAlmostEqual(fld(c1, 6), 1.0e-3)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 2), 5), 2.0e-2)   # D1 1/Pa -> 1/MPa
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 3), 0), 0.3)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 4), 0), 1.3)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 5), 0), 0.1)
        body = ("*MAT_OGDEN_RUBBER\n" + F(1, 1100.0, 0.49, 2, 0, 0, 0, 0) + "\n"
                + F(0.05, 0.01, 0.002, 7, 1, 0, 0.0, 0.0) + "\n" + CURVE)
        lines, _ = _conv(body)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 2), 0), 50.0)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1000.0)   # force vs dL: abscissa m -> mm

    def test_mat_seatbelt(self):
        body = ("*MAT_SEATBELT\n" + F(1, 0.05, 7, 7, 0.002, 0, 0.1, "2.0E9") + "\n"
                + F("1.0E-5", "1.0E-10", "2.0E-10", "1.0E-5", 5000.0, 10.0, 0.05) + "\n" + CURVE)
        lines, _ = _conv(body)
        c1 = _after(lines, "*MAT_SEATBELT", 1)
        self.assertAlmostEqual(fld(c1, 1), 5.0e-8)        # kg/m -> ton/mm
        self.assertAlmostEqual(fld(c1, 4), 2.0)
        self.assertAlmostEqual(fld(c1, 7), 2.0e3)
        c2 = _after(lines, "*MAT_SEATBELT", 2)
        self.assertAlmostEqual(fld(c2, 0), 10.0)          # m2 -> mm2
        self.assertAlmostEqual(fld(c2, 1), 100.0)         # m4 -> mm4
        self.assertAlmostEqual(fld(c2, 4), 5000.0)        # N
        self.assertAlmostEqual(fld(c2, 5), 1.0e4)         # N m -> N mm
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[20:40]), 2.0)     # force ordinate unchanged

    def test_mat_spring_general_nonlinear(self):
        body = ("*SECTION_DISCRETE\n" + F(1, 0) + "\n" + F(0, 0) + "\n"
                "*PART\nspring\n" + F(1, 1, 1) + "\n"
                "*MAT_SPRING_GENERAL_NONLINEAR\n" + F(1, 7, 7, 0.0, 100.0, -100.0) + "\n" + CURVE)
        lines, _ = _conv(body)
        c = _after(lines, "*MAT_SPRING_GENERAL_NONLINEAR", 1)
        self.assertAlmostEqual(fld(c, 4), 100.0)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1000.0)    # displacement m -> mm

    def test_mat_general_nonlinear_1dof(self):
        body = ("*MAT_GENERAL_NONLINEAR_1DOF_DISCRETE_BEAM\n" + F(1, 7850.0, 5000.0, 2, 0.0, 1.0) + "\n"
                + F(7, 0, 0, 0) + "\n" + F(0.01, 0.02, 0.0) + "\n" + CURVE)
        lines, _ = _conv(body)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_GENERAL_NONLINEAR_1DOF_DISCRETE_BEAM", 1), 2), 5.0)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_GENERAL_NONLINEAR_1DOF_DISCRETE_BEAM", 3), 0), 10.0)
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[:20]), 1000.0)

    def test_mat_general_nonlinear_6dof(self):
        cards = [F(1, 7850.0, 5000.0, 200.0, 2, 0.0, 1.0, 0), F(7, 0, 0, 0, 0, 0),
                 F(0, 0, 0, 0, 0, 0), F(0, 0, 0, 0, 0, 0), F(0, 0, 0, 0, 0, 0),
                 F(0.01, 0, 0, 0.5, 0, 0, 0), F(0, 0, 0, 0, 0, 0), F(0.001, 0, 0, 0, 0, 0),
                 F(4000.0, 3000.0, 150.0, 120.0)]
        body = "*MAT_GENERAL_NONLINEAR_6DOF_DISCRETE_BEAM\n" + "\n".join(cards) + "\n" + CURVE
        lines, _ = _conv(body)
        kw = "*MAT_GENERAL_NONLINEAR_6DOF_DISCRETE_BEAM"
        self.assertAlmostEqual(fld(_after(lines, kw, 1), 2), 5.0)         # N/m -> N/mm
        self.assertAlmostEqual(fld(_after(lines, kw, 1), 3), 2.0e5)       # N m -> N mm
        self.assertAlmostEqual(fld(_after(lines, kw, 6), 0), 10.0)        # UTFAILR
        self.assertAlmostEqual(fld(_after(lines, kw, 6), 3), 0.5)         # rotation untouched
        self.assertAlmostEqual(fld(_after(lines, kw, 8), 0), 1.0)         # IUR
        self.assertAlmostEqual(fld(_after(lines, kw, 9), 2), 1.5e5)       # KRS
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[:20]), 1000.0)
        cards[0] = F(1, 7850.0, 5000.0, 200.0, 2, 0.0, 1.0, 2)
        with self.assertRaisesRegex(ConvertError, "IFLAG=2"):
            _conv("*MAT_GENERAL_NONLINEAR_6DOF_DISCRETE_BEAM\n" + "\n".join(cards) + "\n")

    def test_mat_add_permeability(self):
        lines, _ = _conv("*MAT_ADD_PERMEABILITY\n" + F(1, "1.0E-6", 0, 0, 0.0, 0, 0) + "\n")
        self.assertAlmostEqual(fld(_after(lines, "*MAT_ADD_PERMEABILITY", 1), 1), 1.0e-3)


class AirbagSeatbeltTests(unittest.TestCase):
    def test_airbag_wang_nefske(self):
        body = ("*AIRBAG_WANG_NEFSKE_ID\n" + F(1) + "bag\n"
                + F(1, 1, 0, 1.0, 1.0, 0.001, 0.0, 0.0) + "\n"
                + F(718.0, 1005.0, 600.0, 0, 7, 0.0, 0, 0.0) + "\n"
                + F(0.6, 0, 0.002, 0, 0.0, 0, 0.0, 0) + "\n"
                + F("1.0E5", 1.2, 1.0, 0, 0.0, 0.0, 0, 0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0) + "\n" + CURVE)
        lines, ctx = _conv(body)
        kw = "*AIRBAG_WANG_NEFSKE_ID"
        c1 = _after(lines, kw, 2)
        self.assertAlmostEqual(fld(c1, 5), 1.0e6)          # VINI m3 -> mm3
        c3 = _after(lines, kw, 3)
        self.assertAlmostEqual(fld(c3, 0), 7.18e8)         # J/kg/K -> mm2/s2
        self.assertAlmostEqual(fld(c3, 2), 600.0)          # temperature
        c4 = _after(lines, kw, 4)
        self.assertAlmostEqual(fld(c4, 2), 2000.0)         # A23 m2 -> mm2
        c5 = _after(lines, kw, 5)
        self.assertAlmostEqual(fld(c5, 0), 0.1)            # PE
        self.assertAlmostEqual(fld(c5, 1), 1.2e-9)         # RO
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[20:40]), 2.0e-3)   # mass flow kg/s -> ton/s

    def test_airbag_wang_nefske_gc_refused(self):
        body = ("*AIRBAG_WANG_NEFSKE\n" + F(1, 1, 0, 1.0, 1.0, 0.001, 0.0, 0.0) + "\n"
                + F(718.0, 1005.0, 600.0, 0, 0, 0.0, 0, 0.0) + "\n" + F(0.6) + "\n"
                + F("1.0E5", 1.2, 9.81, 0) + "\n" + F(0.0) + "\n")
        with self.assertRaisesRegex(ConvertError, "GC="):
            _conv(body)

    def test_airbag_linear_fluid(self):
        body = ("*AIRBAG_LINEAR_FLUID\n" + F(1, 1, 0, 1.0, 1.0, 0.001, 0.0, 0.0) + "\n"
                + F("2.2E9", 1000.0, 7, 0, 0, 0, 0, 0) + "\n" + F("5.0E5", 0, 0) + "\n" + CURVE)
        lines, _ = _conv(body)
        self.assertAlmostEqual(fld(_after(lines, "*AIRBAG_LINEAR_FLUID", 2), 0), 2.2e3)
        self.assertAlmostEqual(fld(_after(lines, "*AIRBAG_LINEAR_FLUID", 2), 1), 1.0e-9)
        self.assertAlmostEqual(fld(_after(lines, "*AIRBAG_LINEAR_FLUID", 3), 0), 0.5)
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[20:40]), 2.0e-3)

    def test_porosity_leakage(self):
        lines, _ = _conv("*MAT_ADD_AIRBAG_POROSITY_LEAKAGE\n" + F(1, 0.7, 3.0, 0.0, 7, 0.0, 0.0) + "\n"
                         + F(2, 0.7, 3.0, 0.0, 1, 0.0, 0.0) + "\n" + F(3, 0.1, 0.2, 0.0, 1, 0.5, 0.0) + "\n")
        b = lines.index("*MAT_ADD_AIRBAG_POROSITY_LEAKAGE")
        self.assertAlmostEqual(fld(lines[b + 1], 2), 3000.0)   # velocity m/s -> mm/s
        self.assertAlmostEqual(fld(lines[b + 1], 1), 0.7)
        self.assertAlmostEqual(fld(lines[b + 2], 2), 3.0)      # unit-less for FVOPT 1
        self.assertAlmostEqual(fld(lines[b + 3], 2), 0.2)      # X3
        # FVOPT = 0 without any airbag: FAC unused, left alone
        lines, ctx = _conv("*MAT_ADD_AIRBAG_POROSITY_LEAKAGE\n" + F(1, 0.7, 3.0, 0.0, 0, 0.0, 0.0) + "\n")
        self.assertAlmostEqual(fld(_after(lines, "*MAT_ADD_AIRBAG_POROSITY_LEAKAGE", 1), 2), 3.0)
        self.assertTrue(any("FAC is unused" in n for n in ctx.notes))
        # FVOPT = 0 with one airbag OPT = 7: FAC is a velocity
        bag = ("*AIRBAG_WANG_NEFSKE\n" + F(1, 1, 0, 1.0, 1.0, 0.001, 0.0, 0.0) + "\n"
               + F(718.0, 1005.0, 600.0, 0, 0, 0.0, 0, 0.0) + "\n" + F(0.6) + "\n"
               + F("1.0E5", 1.2, 1.0, 0, 0.0, 0.0, 7, 0) + "\n" + F(0.0) + "\n")
        lines, ctx = _conv(bag + "*MAT_ADD_AIRBAG_POROSITY_LEAKAGE\n" + F(1, 0.7, 3.0, 0.0, 0, 0.0, 0.0) + "\n")
        self.assertAlmostEqual(fld(_after(lines, "*MAT_ADD_AIRBAG_POROSITY_LEAKAGE", 1), 2), 3000.0)
        self.assertTrue(any("OPT=7" in n for n in ctx.notes))

    def test_seatbelt_elements_and_section(self):
        # SLEN is an E16 spanning two 8-char columns (R16 Vol I p.19-70)
        body = ("*ELEMENT_SEATBELT\n" + F(1, 2, 3, 4, 0, w=8) + F(0.05, w=16)
                + F(0, 0, w=8) + "\n"
                "*ELEMENT_SEATBELT_ACCELEROMETER\n" + F(1, 10, 11, 12, 0, 0, 0.5) + "\n"
                "*SECTION_SEATBELT\n" + F(1, "1.0E-4", 0.002) + "\n")
        lines, _ = _conv(body)
        self.assertAlmostEqual(float(_after(lines, "*ELEMENT_SEATBELT", 1)[40:56]), 50.0)
        self.assertAlmostEqual(fld(_after(lines, "*ELEMENT_SEATBELT_ACCELEROMETER", 1), 6), 5.0e-4)
        self.assertAlmostEqual(fld(_after(lines, "*SECTION_SEATBELT", 1), 1), 100.0)
        self.assertAlmostEqual(fld(_after(lines, "*SECTION_SEATBELT", 1), 2), 2.0)

    def test_seatbelt_accelerometer_igrav_curve(self):
        # IGRAV > 1 names a gravitation-flag-vs-TIME curve (p.19-73)
        body = ("*ELEMENT_SEATBELT_ACCELEROMETER\n" + F(1, 10, 11, 12, 7, 0, 0.5) + "\n"
                + CURVE)
        lines, _ = _conv(body, src=KMM)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1.0e-3)   # ms -> s


class MiscTests(unittest.TestCase):
    def test_boundary_flux(self):
        lines, _ = _conv("*BOUNDARY_FLUX_SET\n" + F(5, 0) + "\n" + F(0, 2000.0, 2000.0, 2000.0, 2000.0, 0, 0) + "\n")
        self.assertAlmostEqual(fld(_after(lines, "*BOUNDARY_FLUX_SET", 2), 1), 2.0)   # W/m2 -> mW/mm2 = N/(mm s)
        lines, _ = _conv("*BOUNDARY_FLUX_SEGMENT\n" + F(1, 2, 3, 4) + "\n" + F(7, 1.0, 1.0, 1.0, 1.0, 0, 0) + "\n" + CURVE)
        self.assertAlmostEqual(fld(_after(lines, "*BOUNDARY_FLUX_SEGMENT", 2), 1), 1.0)
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[20:40]), 2.0e-3)
        with self.assertRaisesRegex(ConvertError, "NHISV"):
            _conv("*BOUNDARY_FLUX_SET\n" + F(5, 0) + "\n"
                  + F(0, 1.0, 1.0, 1.0, 1.0, 0, 2) + "\n")
        with self.assertRaisesRegex(ConvertError, "pair"):
            _conv("*BOUNDARY_FLUX_SET\n" + F(5, 0) + "\n" + F(0, 1.0, 1.0, 1.0, 1.0, 0, 0) + "\n" + F(6, 0) + "\n")

    def test_boundary_flux_stacked_pairs(self):
        # every (segment, flux) pair is scaled, not only the first
        body = ("*BOUNDARY_FLUX_SEGMENT\n"
                + F(1, 2, 3, 4) + "\n" + F(0, -3000.0, 0, 0, 0, 0, 0) + "\n"
                + F(5, 6, 7, 8) + "\n" + F(0, -3000.0, 0, 0, 0, 0, 0) + "\n")
        lines, _ = _conv(body)
        b = lines.index("*BOUNDARY_FLUX_SEGMENT")
        self.assertAlmostEqual(fld(lines[b + 2], 1), -3.0)
        self.assertAlmostEqual(fld(lines[b + 4], 1), -3.0)

    def test_damping_part_stiffness(self):
        lines, _ = _conv("*DAMPING_PART_STIFFNESS\n" + F(1, -0.5) + "\n" + F(2, 0.1) + "\n", src=KMM)
        b = lines.index("*DAMPING_PART_STIFFNESS")
        self.assertAlmostEqual(fld(lines[b + 1], 1), -5.0e-4)   # ms -> s
        self.assertAlmostEqual(fld(lines[b + 2], 1), 0.1)
        lines, _ = _conv("*DAMPING_PART_MASS_SET\n" + F(1, 7, 1.0, 0) + "\n" + CURVE, src=KMM)
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[20:40]), 2000.0)  # 1/ms -> 1/s

    def test_section_solid_spg_efg(self):
        body = ("*SECTION_SOLID_SPG\n" + F(1, 47, 0) + "\n" + F(1.6, 1.6, 1.6, 0, 1, 15, 0) + "\n"
                + F(2, "5.0E8", 1.2, 1, 0, 0, 0, -0.001) + "\n"
                "*SECTION_SOLID_EFG\n" + F(2, 41, 0) + "\n" + F(1.2, 1.2, 1.2, 0, 0, 3, 2, 0.01) + "\n"
                + F(0, 0.5, 0, 0, 0, 1, 1.01, 0.002) + "\n")
        lines, _ = _conv(body)
        self.assertAlmostEqual(fld(_after(lines, "*SECTION_SOLID_SPG", 3), 1), 500.0)
        self.assertAlmostEqual(fld(_after(lines, "*SECTION_SOLID_SPG", 2), 0), 1.6)
        self.assertAlmostEqual(fld(_after(lines, "*SECTION_SOLID_EFG", 3), 7), 2.0)

    def test_whitelisted_keywords_do_not_block(self):
        body = ("*CONSTRAINED_ADAPTIVITY\n" + F(10, 11, 12, w=10) + "\n"
                "*DATABASE_CROSS_SECTION_SET_ID\n" + F(1) + "section A\n" + F(5, 0, 0, 0, 0, 0, 0, 0) + "\n"
                "*COMMENTUNITS: kg m s\nfree text\n"
                "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n")
        lines, ctx = _conv(body)
        self.assertFalse(ctx.unknown)
        self.assertIn("free text", lines)


class Batch2Tests(unittest.TestCase):
    def test_part_move_and_intfor_file(self):
        # PART_MOVE has the *NODE column split: PID I8, XMOV/YMOV/ZMOV E16
        # (R16 Vol I p.37-38); the INTFOR FILE option only prepends the
        # file-name card - the DT card still follows (p.16-33) and scales
        body = ("*PART_MOVE\n" + F(1, w=8) + F(0.1, 0.0, -0.25, w=16)
                + F(0, 0, w=8) + "\n"
                "*DATABASE_BINARY_INTFOR_FILE\nintfor_run1\n"
                + F(0.6, 0, 0, 0) + "\n"
                "*MAT_ELASTIC\n" + F(1, "7.85E-6", 210.0, 0.3) + "\n")
        lines, ctx = _conv(body, src=KMM)
        c = _after(lines, "*PART_MOVE", 1)
        self.assertAlmostEqual(float(c[8:24]), 0.1)       # mm -> mm
        self.assertAlmostEqual(float(c[40:56]), -0.25)
        self.assertFalse(ctx.unknown)
        self.assertIn("intfor_run1", lines)
        self.assertAlmostEqual(fld(_after(lines, "*DATABASE_BINARY_INTFOR_FILE", 2), 0), 6.0e-4)
        # and with a length change the displacements scale
        lines, _ = _conv("*PART_MOVE\n" + F(1, w=8) + F(0.1, 0.0, -0.25, w=16)
                         + F(0, 0, w=8) + "\n"
                         "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n")
        c = _after(lines, "*PART_MOVE", 1)
        self.assertAlmostEqual(float(c[8:24]), 100.0)     # m -> mm
        self.assertAlmostEqual(float(c[40:56]), -250.0)

    def test_damping_part_mass_set_flag_card(self):
        # the FLAG=1 scale-factor card is not another PID/LCID card: its
        # STY must not be taken as a curve id (R16 Vol I p.15-10)
        body = ("*DAMPING_PART_MASS_SET\n" + F(901021, 0, 18.85, 1) + "\n"
                + F(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) + "\n"
                "*MAT_ELASTIC\n" + F(1, "7.85E-6", 210.0, 0.3) + "\n" + CURVE)
        lines, ctx = _conv(body, src=KMM)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1.0)       # curve 7 untouched
        self.assertAlmostEqual(float(pt[20:40]), 2.0)
        self.assertTrue(any("SF=18.85" in w for w in ctx.warnings))
        # a real Ds(t) curve is (TIME, 1/time)
        body = ("*DAMPING_PART_MASS\n" + F(1, 7, 1.0, 0) + "\n"
                "*MAT_ELASTIC\n" + F(1, "7.85E-6", 210.0, 0.3) + "\n" + CURVE)
        lines, _ = _conv(body, src=KMM)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1.0e-3)    # ms -> s
        self.assertAlmostEqual(float(pt[20:40]), 2000.0)  # 1/ms -> 1/s

    def test_mat_119_iflag1(self):
        # IFLAG=1: F = -K*dl/l0 makes the translational K a FORCE (R16 Vol
        # II p.2-808 Remark 2) - kg-m-s -> ton-mm-s leaves N unchanged
        # (STIFF would have divided by 1000); rotational data is refused
        cards = [F(1, 7850.0, 5000.0, 0, 2, 0.0, 1.0, 1), F(7, 0, 0, 0, 0, 0),
                 F(0, 0, 0, 0, 0, 0), F(0, 0, 0, 0, 0, 0), F(0, 0, 0, 0, 0, 0),
                 F(0.01, 0, 0, 0, 0, 0, 0), F(0, 0, 0, 0, 0, 0), F(0, 0, 0, 0, 0, 0),
                 F(4000.0, 3000.0, 0, 0)]
        body = "*MAT_GENERAL_NONLINEAR_6DOF_DISCRETE_BEAM\n" + "\n".join(cards) + "\n" + CURVE
        lines, _ = _conv(body)
        kw = "*MAT_GENERAL_NONLINEAR_6DOF_DISCRETE_BEAM"
        self.assertAlmostEqual(fld(_after(lines, kw, 1), 2), 5000.0)
        self.assertAlmostEqual(fld(_after(lines, kw, 9), 0), 4000.0)   # KTS
        self.assertAlmostEqual(fld(_after(lines, kw, 6), 0), 0.01)     # strain now
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[:20]), 1.0)  # strain abscissa
        cards[0] = F(1, 7850.0, 5000.0, 200.0, 2, 0.0, 1.0, 1)         # KR nonzero
        with self.assertRaisesRegex(ConvertError, "ROTATIONAL"):
            _conv("*MAT_GENERAL_NONLINEAR_6DOF_DISCRETE_BEAM\n" + "\n".join(cards) + "\n")

    def test_mat_ogden_vflag1_and_tbhys(self):
        # VFLAG=1: the Prony Gi are normalized moduli - left unchanged
        body = ("*MAT_OGDEN_RUBBER\n" + F(1, 1100.0, 0.49, 0, 0, 0, 0, 0) + "\n"
                + F("3.0E5", 0, 0, 0, 0, 0, 0, 0) + "\n" + F(1.3, 5.0) + "\n"
                + F(0.2, 100.0, 1) + "\n" + F(0.1, 1000.0) + "\n")
        lines, ctx = _conv(body)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 4), 0), 0.2)
        self.assertAlmostEqual(fld(_after(lines, "*MAT_OGDEN_RUBBER", 5), 0), 0.1)
        self.assertTrue(any("VFLAG=1" in n for n in ctx.notes))
        # TBHYS: damage table indexed by strain-energy DENSITY (p.2-557)
        body = ("*MAT_OGDEN_RUBBER\n" + F(1, 1100.0, -0.49, 0, 0, 0, 0, 0) + "\n"
                + F(9, 0, 0, 0, 0, 0, 0, 0) + "\n"
                + F("3.0E5", 0, 0, 0, 0, 0, 0, 0) + "\n" + F(1.3, 5.0) + "\n"
                "*DEFINE_TABLE\n" + F(9) + "\n" + F(100.0, w=20) + "\n"
                "*DEFINE_CURVE\n" + F(7) + "\n" + F(0.0, 1.0, w=20) + "\n"
                + F(100.0, 0.5, w=20) + "\n")
        lines, _ = _conv(body)
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_TABLE", 2)[:20]), 1.0e-4)
        self.assertAlmostEqual(float(_after(lines, "*DEFINE_CURVE", 3)[:20], ), 1.0e-4)

    def test_airbag_lcjrv_is_dimensionless(self):
        # LCJRV: normalized jet-velocity factor vs angle (p.3-31, Fig 3-3)
        body = ("*AIRBAG_WANG_NEFSKE_MULTIPLE_JETTING\n"
                + F(1, 1, 0, 1.0, 1.0, 0.001, 0.0, 0.0) + "\n"
                + F("7.18E8", "1.005E9", 600.0, 0, 0, 0.0, 0, 0.0) + "\n"
                + F(0.6, 0, 2000.0, 0, 0.0, 0, 0.0, 0) + "\n"
                + F(0.1, "1.2E-9", 1.0, 0, 0.0, 0.0, 0, 0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0) + "\n"
                + F(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7, 1.0) + "\n"
                + F(0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0) + "\n" + CURVE)
        lines, _ = _conv(body, src=KMM)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1.0)
        self.assertAlmostEqual(float(pt[20:40]), 2.0)

    def test_porosity_leakage_x0_1_ratio_curves(self):
        # X0 = 1: the FLC/FAC curves run over RATIOS (p.2-40/2-41)
        body = ("*MAT_ADD_AIRBAG_POROSITY_LEAKAGE\n" + F(1, -7, 3.0, 0.0, 1, 1.0, 0.0) + "\n"
                "*MAT_ELASTIC\n" + F(1, "7.85E-6", 210.0, 0.3) + "\n" + CURVE)
        lines, _ = _conv(body, src=KMM)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1.0)       # ratio abscissa untouched
        with self.assertRaisesRegex(ConvertError, "HYBRID"):
            _conv("*AIRBAG_HYBRID\n" + F(1, 1, 0) + "\n"
                  "*MAT_ADD_AIRBAG_POROSITY_LEAKAGE\n" + F(1, 0.7, 3.0, 0.0, 0, 0.0, 0.0) + "\n"
                  "*MAT_ELASTIC\n" + F(1, "7.85E-6", 210.0, 0.3) + "\n", src=KMM)

    def test_mat_076_moisture_mst_curve(self):
        body = ("*MAT_GENERAL_VISCOELASTIC_MOISTURE\n"
                + F(1, "1.1E-6", 2.0, 0, 0, 293.0, 17.4, 51.6) + "\n"
                + F(0, 0, 0.0, 0.0, 0, 0, 0, 0) + "\n"
                + F(0.1, 0.0, 0.0, 0.0, 7) + "\n" + F(1.0, 10.0, "1.0E-2", 0.1) + "\n"
                + CURVE)
        lines, _ = _conv(body, src=KMM)
        pt = _after(lines, "*DEFINE_CURVE", 3)
        self.assertAlmostEqual(float(pt[:20]), 1.0e-3)    # MST(t): ms -> s

    def test_g_mm_s_shares_pa_with_kg_m_s(self):
        # deliberate: a modulus-only deck (Pa evidence only) ties between
        # kg-m-s and g-mm-s and must come out AMBIGUOUS
        from kunit import detect
        p = _write("*KEYWORD\n*MAT_ELASTIC\n" + F(1, "", "2.1E11", 0.3) + "\n*END\n")
        v = detect(p, follow_includes=False, use_headers=False)
        self.assertTrue(v.ambiguous)
        top2 = {sys.key for _s, sys in v.ranked[:2]}
        self.assertEqual(top2, {"kg-m-s", "g-mm-s"})


if __name__ == "__main__":
    unittest.main()
