import os
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit.parser import (KFile, FieldWidthError, format_fixed, STD8)


class TestLatin1RoundTrip(unittest.TestCase):
    def test_non_ascii_bytes_preserved(self):
        # A comment containing 'µ' (0xB5 in latin-1) plus a '°' (0xB0).
        raw = ("$ units in \xb5s and \xb0C\n"
               "*KEYWORD\n"
               "*TITLE\n"
               "d\xe9j\xe0 test\n").encode("latin-1")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.k")
            dst = os.path.join(d, "out.k")
            with open(src, "wb") as fh:
                fh.write(raw)
            kf = KFile(src)
            kf.write(dst)
            with open(dst, "rb") as fh:
                out = fh.read()
        self.assertEqual(out, raw)


class TestFieldWidthError(unittest.TestCase):
    def test_overflow_raises_field_width_error(self):
        # A value that cannot fit even at 1 sig-fig into a narrow field.
        # 6-char field, scale x1 not allowed (f==1 short-circuits), so scale
        # a modest value up by a big factor to overflow.
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.k")
            with open(src, "w", encoding="latin-1") as fh:
                # single card line, first field holds '1.0' in a 4-char field
                fh.write("*KEYWORD\n*FOO\n 1.0\n")
            kf = KFile(src)
            widths = [4, 4, 4, 4, 4, 4, 4, 4]
            # scale by 1e9 -> 1e9 cannot fit a 4-char field even at 1 sig-fig
            with self.assertRaises(FieldWidthError) as ctx:
                kf.scale_field(2, widths, False, 0, Fraction(10) ** 9)
            self.assertEqual(ctx.exception.line_idx, 2)
            self.assertEqual(ctx.exception.width, 4)


class TestFreeFormatMaxFmtErr(unittest.TestCase):
    def test_free_format_updates_max_fmt_err(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.k")
            with open(src, "w", encoding="latin-1") as fh:
                # comma free-format card; value needing >9 sig-figs to round-trip
                fh.write("*KEYWORD\n*FOO\n1.23456789012345,2.0\n")
            kf = KFile(src)
            self.assertEqual(kf.max_fmt_err, 0.0)
            changed = kf.scale_field(2, STD8, False, 0, Fraction(3))
            self.assertTrue(changed)
            self.assertTrue(kf.is_free(2))
            # 9G formatting of 3.70370367... loses precision => nonzero err
            self.assertGreater(kf.max_fmt_err, 0.0)


class TestFormatFixed(unittest.TestCase):
    def test_fits_and_rel_err_accurate(self):
        val = Decimal("1.23456789012345")
        s, rel = format_fixed(val, 10)
        self.assertLessEqual(len(s), 10)
        expected_rel = abs(float(Decimal(s.strip())) / float(val) - 1.0)
        self.assertAlmostEqual(rel, expected_rel, places=15)
        # rel error must be small but nonzero for a truncated value
        self.assertGreater(rel, 0.0)
        self.assertLess(rel, 1e-6)

    def test_overflow_raises_valueerror(self):
        with self.assertRaises(ValueError):
            format_fixed(Decimal("123456"), 4)


if __name__ == "__main__":
    unittest.main()
