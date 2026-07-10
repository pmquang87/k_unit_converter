"""T2 engine coverage: deterministic property-style tests for previously
untested behaviour in kunit.units, kunit.parser and kunit.detect.

No third-party deps (no hypothesis): "property" tests loop over fixed/seeded
value sets so they are fully deterministic and CI-safe.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import detect, factor, parse_system
from kunit.cli import main as cli_main
from kunit.parser import KFile, format_fixed, parse_number
from kunit.units import (DENSITY, FORCE, LENGTH, PRESSURE, VELOCITY, MASS,
                         MOMENT, PRESETS, apply_factor, parse_dim_name,
                         parse_system as u_parse_system)

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")
SLINCH = parse_system("slinch-in-s")
LBFT = parse_system("lb-ft-s")

_TMPDIRS = []


def _mkdeck(text, name="deck.k", binary=False):
    d = tempfile.mkdtemp(prefix="kunit_t2_")
    _TMPDIRS.append(d)
    p = os.path.join(d, name)
    mode = "wb" if binary else "w"
    with open(p, mode, newline="" if not binary else None) as fh:
        fh.write(text)
    return p


def tearDownModule():
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


# ── units.py ─────────────────────────────────────────────────────────────────
class UnitsFactorTests(unittest.TestCase):
    IMPERIAL_DIMS = (DENSITY, PRESSURE, FORCE, LENGTH, VELOCITY)

    def test_imperial_round_trips_exact(self):
        # SI -> imperial -> SI recovers the exact Fraction; factors invert.
        for d in self.IMPERIAL_DIMS:
            for imp in (SLINCH, LBFT):
                f_there = factor(d, SI, imp)
                f_back = factor(d, imp, SI)
                self.assertEqual(f_there * f_back, Fraction(1),
                                 f"{d} {imp.key}")
                # apply_factor round-trip on a representative value
                v = Decimal("1234.5")
                there = apply_factor(v, f_there)
                back = apply_factor(there, f_back)
                self.assertEqual(back, v, f"apply {d} {imp.key}")

    def test_factor_inverse_over_all_preset_pairs(self):
        dims = (MASS, LENGTH, DENSITY, PRESSURE, FORCE, VELOCITY, MOMENT)
        systems = list(PRESETS.values())
        for d in dims:
            for a in systems:
                for b in systems:
                    self.assertEqual(
                        factor(d, a, b) * factor(d, b, a), Fraction(1),
                        f"{d} {a.key}<->{b.key}")

    def test_factor_composition_over_all_preset_pairs(self):
        dims = (MASS, LENGTH, DENSITY, PRESSURE, FORCE, VELOCITY, MOMENT)
        systems = list(PRESETS.values())
        for d in dims:
            for a in systems:
                for b in systems:
                    for c in systems:
                        self.assertEqual(
                            factor(d, a, b) * factor(d, b, c),
                            factor(d, a, c),
                            f"{d} {a.key}->{b.key}->{c.key}")


class UnitsDescribeTests(unittest.TestCase):
    def _d(self, key):
        return PRESETS[key].describe()

    def test_metric_labels(self):
        self.assertIn("pressure = Pa", self._d("kg-m-s"))
        self.assertIn("force = N", self._d("kg-m-s"))
        self.assertIn("pressure = MPa", self._d("ton-mm-s"))
        self.assertIn("force = N", self._d("ton-mm-s"))
        self.assertIn("pressure = Mbar", self._d("g-cm-us"))

    def test_imperial_labels(self):
        # slinch-in-s is the fully-imperial consistent system: psi + lbf.
        s = self._d("slinch-in-s")
        self.assertIn("pressure = psi", s)
        self.assertIn("force = lbf", s)
        # slug-ft-s yields lbf force but psf pressure (NOT psi) - assert actual.
        slug = self._d("slug-ft-s")
        self.assertIn("force = lbf", slug)
        self.assertNotIn("psi", slug)
        # lb-ft-s is a valid poundal-style system: neither psi nor lbf.
        # (This is correct behaviour, not a bug - the derived pressure/force
        # units simply have no common name.)
        lb = self._d("lb-ft-s")
        self.assertNotIn("psi", lb)
        self.assertNotIn("lbf", lb)

    def test_describe_uses_parsed_system(self):
        self.assertEqual(parse_system("ton-mm-s").describe(),
                         PRESETS["ton-mm-s"].describe())


class UnitsParseTests(unittest.TestCase):
    def test_parse_system_wrong_token_count(self):
        for bad in ("kg-m", "kg-m-s-extra", "kg"):
            with self.assertRaises(ValueError):
                parse_system(bad)

    def test_parse_system_unknown_unit(self):
        for bad in ("kg-foo-s", "zog-m-s", "kg-m-century"):
            with self.assertRaises(ValueError):
                parse_system(bad)

    def test_parse_system_aliases(self):
        self.assertEqual(parse_system("Mg,mm,s"), TON)
        self.assertEqual(parse_system("tonne mm sec"), TON)
        self.assertEqual(u_parse_system("kilogram meter second"), SI)

    def test_parse_dim_name(self):
        self.assertEqual(parse_dim_name("stress"), PRESSURE)
        self.assertEqual(parse_dim_name("modulus"), PRESSURE)
        self.assertEqual(parse_dim_name("velocity"), VELOCITY)
        self.assertEqual(parse_dim_name("density"), DENSITY)
        self.assertEqual(parse_dim_name(" FORCE "), FORCE)
        for bad in ("nonsense", "kg", ""):
            with self.assertRaises(ValueError):
                parse_dim_name(bad)


# ── parser.py ────────────────────────────────────────────────────────────────
def _bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


class ParserEolTests(unittest.TestCase):
    def test_crlf_preserved(self):
        deck = ("*KEYWORD\r\n*NODE\r\n"
                "       1             1.7             0.0             0.0\r\n"
                "*END\r\n")
        p = _mkdeck(deck.encode("latin-1"), "crlf.k", binary=True)
        kf = KFile(p)
        self.assertEqual(kf.eol, "\r\n")
        b = next(x for x in kf.blocks if x.name == "NODE")
        widths = [8, 16, 16, 16]
        self.assertTrue(kf.scale_field(b.data[0], widths, b.long, 1,
                                       Fraction(1000)))
        out = os.path.join(os.path.dirname(p), "crlf_out.k")
        kf.write(out)
        raw = _bytes(out)
        self.assertIn(b"\r\n", raw)
        # no bare LF that isn't part of a CRLF
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
        # value actually scaled 1.7 -> 1700
        kf2 = KFile(out)
        b2 = next(x for x in kf2.blocks if x.name == "NODE")
        self.assertAlmostEqual(
            float(kf2.get_number(b2.data[0], widths, b2.long, 1)), 1700.0)

    def test_lf_only_stays_lf(self):
        deck = ("*KEYWORD\n*NODE\n"
                "       1             1.7             0.0             0.0\n"
                "*END\n")
        p = _mkdeck(deck.encode("latin-1"), "lf.k", binary=True)
        kf = KFile(p)
        self.assertEqual(kf.eol, "\n")
        out = os.path.join(os.path.dirname(p), "lf_out.k")
        kf.write(out)
        self.assertNotIn(b"\r\n", _bytes(out))


class ParserLongFormatTests(unittest.TestCase):
    def test_long_format_20wide_scaling(self):
        # LONG=Y forces every fixed field to 20 chars (widths=[20]*n path).
        deck = ("*KEYWORD LONG=Y\n*NODE\n"
                + "7".rjust(20) + "1.7".rjust(20) + "2.5".rjust(20) + "\n"
                + "*END\n")
        p = _mkdeck(deck.encode("latin-1"), "long.k", binary=True)
        kf = KFile(p)
        b = next(x for x in kf.blocks if x.name == "NODE")
        self.assertTrue(b.long)
        widths = [10, 10, 10]  # nominal; long path overrides to [20]*n
        # scale x (field 1) by 1000, y (field 2) by 1000
        self.assertTrue(kf.scale_field(b.data[0], widths, b.long, 1,
                                       Fraction(1000)))
        self.assertTrue(kf.scale_field(b.data[0], widths, b.long, 2,
                                       Fraction(1000)))
        fl = kf.fields(b.data[0], widths, b.long)
        # fields land in 20-wide columns
        self.assertEqual(fl[1][1], 20)
        self.assertEqual(fl[1][2], 40)
        self.assertEqual(fl[2][1], 40)
        self.assertEqual(fl[2][2], 60)
        self.assertAlmostEqual(float(parse_number(fl[1][0])), 1700.0)
        self.assertAlmostEqual(float(parse_number(fl[2][0])), 2500.0)
        # node id untouched
        self.assertEqual(int(parse_number(fl[0][0])), 7)

    def test_long_format_per_block_plus_toggle(self):
        deck = ("*KEYWORD\n*NODE +\n"
                + "3".rjust(20) + "4.0".rjust(20) + "\n*END\n")
        p = _mkdeck(deck.encode("latin-1"), "plus.k", binary=True)
        kf = KFile(p)
        b = next(x for x in kf.blocks if x.name == "NODE")
        self.assertTrue(b.long)


class FormatFixedTests(unittest.TestCase):
    def test_format_fixed_invariants(self):
        # seeded magnitude/width sweep - deterministic "property" test
        mantissas = ("1.0", "1.23456789", "9.87654321", "3.14159265",
                     "7.85", "2.718281828")
        exponents = range(-12, 13)
        widths = (8, 10, 12, 16, 20)
        signs = (1, -1)
        for man in mantissas:
            for e in exponents:
                for w in widths:
                    for sgn in signs:
                        v = Decimal(man).scaleb(e) * sgn
                        try:
                            s, rel = format_fixed(v, w)
                        except ValueError:
                            # cannot fit even 1 sig-fig - acceptable, skip
                            continue
                        self.assertLessEqual(len(s), w,
                                             f"{v!r} w={w}: {s!r}")
                        back = parse_number(s)
                        self.assertIsNotNone(back)
                        fv = float(v)
                        if fv != 0.0:
                            got = abs(float(back) / fv - 1.0)
                            # reproduced within the reported rel_err (+eps)
                            self.assertLessEqual(got, rel + 1e-12,
                                                 f"{v!r} w={w}: {s!r} rel={rel}")

    def test_format_fixed_zero(self):
        s, rel = format_fixed(Decimal("0"), 10)
        self.assertEqual(rel, 0.0)
        self.assertEqual(s.strip(), "0.0")
        self.assertEqual(len(s), 10)
        s2, _ = format_fixed(Decimal("0"), 2)  # inner < 3 -> "0"
        self.assertEqual(s2.strip(), "0")

    def test_format_fixed_pad_right(self):
        s, _ = format_fixed(Decimal("1.5"), 10, pad_right=2)
        self.assertEqual(len(s), 10)
        self.assertTrue(s.endswith("  "))


class SetFieldOverflowTests(unittest.TestCase):
    def test_set_field_overflow_raises(self):
        deck = ("*KEYWORD\n*NODE\n"
                "       1             1.7             0.0             0.0\n"
                "*END\n")
        p = _mkdeck(deck.encode("latin-1"), "of.k", binary=True)
        kf = KFile(p)
        b = next(x for x in kf.blocks if x.name == "NODE")
        widths = [8, 16, 16, 16]
        # a 9-char pre-formatted string cannot fit an 8-char field
        with self.assertRaises(ValueError):
            kf.set_field(b.data[0], widths, b.long, 0, "123456789")
        # exactly 8 chars is fine
        kf.set_field(b.data[0], widths, b.long, 0, "12345678")


# ── detect.py ────────────────────────────────────────────────────────────────
class DetectAmbiguityTests(unittest.TestCase):
    def test_ambiguous_top_two_within_20pct(self):
        # density 0.00785 reads as steel (7850 kg/m^3) for several systems
        # (g-mm-ms, kg-cm-ms, kg-cm-s) since density is time-independent, so
        # the top scores tie -> ambiguous.
        deck = ("*KEYWORD\n*MAT_ELASTIC\n"
                "         1  0.00785       0.3\n*END\n")
        p = _mkdeck(deck)
        v = detect(p)
        self.assertTrue(v.ambiguous)
        self.assertIsNotNone(v.system)
        best = v.ranked[0][0]
        second = v.ranked[1][0]
        self.assertGreater(best, 0)
        self.assertGreater(second / best, 0.8)  # within 20%

    def test_no_evidence_deck(self):
        deck = ("*KEYWORD\n*ELEMENT_SOLID\n"
                "       1       1       1       2       3       4"
                "       5       6       7       8\n*END\n")
        p = _mkdeck(deck)
        v = detect(p)
        self.assertIsNone(v.system)
        self.assertTrue(v.ambiguous)
        # CLI detect returns 2 when no usable evidence is found
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cli_main(["detect", p])
        self.assertEqual(rc, 2)


class DetectDetonationTests(unittest.TestCase):
    def test_detonation_velocity_evidence(self):
        # *MAT_HIGH_EXPLOSIVE_BURN card: MID, RO, D(=detonation velocity), PCJ..
        deck = ("*KEYWORD\n*MAT_HIGH_EXPLOSIVE_BURN\n"
                "         1    1630.0    8000.02.10000E10"
                "       0.0       0.0       0.0\n*END\n")
        p = _mkdeck(deck)
        v = detect(p)
        # detonation velocity ~8000 m/s reads correctly only in kg-m-s
        self.assertEqual(v.system, SI)
        self.assertTrue(any("detonation velocity" in e for e in v.evidence),
                        v.evidence)

    def test_detonation_shifts_ranking_toward_kg_m_s(self):
        # A high-explosive with only a detonation velocity (no density anchor)
        # still contributes toward kg-m-s via the 3500-10000 m/s band.
        deck = ("*KEYWORD\n*MAT_HIGH_EXPLOSIVE_BURN\n"
                "         1       0.0    8000.0       0.0"
                "       0.0       0.0       0.0\n*END\n")
        p = _mkdeck(deck)
        v = detect(p)
        si_score = next(s for s, sys in v.ranked if sys == SI)
        self.assertGreater(si_score, 0.0)


if __name__ == "__main__":
    unittest.main()
