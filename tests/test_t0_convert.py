"""T0 tests for kunit/convert.py: FieldWidthError wrapping, atomic multi-file
writes, and the DIM_NAMES fallback guard in report()."""
import os
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import ConvertError, convert, parse_system, report, scan
from kunit.convert import FieldWidthError
from kunit.parser import KFile

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")

DECK_SI = """*KEYWORD
*NODE
       1             1.7        1.654231       0.2799996       0       0
       2            -1.6             0.0      7.85000-2       0       0
*END
"""


def _write(text, name="deck.k", d=None):
    d = d or tempfile.mkdtemp(prefix="kunit_t0_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _lines(path):
    with open(path, newline="") as fh:
        return fh.read().split("\n")


def _make_field_width_error(line_idx, msg):
    """Build a FieldWidthError regardless of the parser owner's constructor
    signature (it currently takes (line_idx, value, width); guard defensively
    in case that evolves)."""
    for args in ((line_idx, msg, 10), (line_idx, msg), (msg,)):
        try:
            e = FieldWidthError(*args)
            break
        except TypeError:
            continue
    else:
        e = FieldWidthError()
    if not hasattr(e, "line_idx"):
        e.line_idx = line_idx
    return e


class FieldWidthErrorWrapTests(unittest.TestCase):
    def test_field_width_overflow_surfaces_as_convert_error(self):
        # simulate scale_field overflowing a narrow field during the edit
        # pass; convert() must wrap it as ConvertError, not leak a traceback
        p = _write(DECK_SI)
        out = p + ".o.k"
        orig = KFile.scale_field

        def boom(self, line_idx, *a, **kw):
            raise _make_field_width_error(line_idx, "scaled value overflows "
                                          "its 10-char field")

        KFile.scale_field = boom
        try:
            with self.assertRaises(ConvertError) as cm:
                convert(p, SI, TON, out, self_check=False)
        finally:
            KFile.scale_field = orig
        # ConvertError, not a raw FieldWidthError, and it names the file
        self.assertIn("overflows", str(cm.exception))
        self.assertIn(os.path.basename(p), str(cm.exception))
        # nothing half-written, no temp files left behind
        self.assertFalse(os.path.exists(out))
        self.assertFalse(any(f.endswith(".tmp_kunit")
                             for f in os.listdir(os.path.dirname(p))))


class AtomicWriteTests(unittest.TestCase):
    def test_normal_convert_output_correct_and_no_temp_left(self):
        p = _write(DECK_SI)
        out = p + ".o.k"
        convert(p, SI, TON, out, self_check=False)
        li = _lines(out)
        ni = li.index("*NODE")
        self.assertAlmostEqual(float(li[ni + 1][8:24]), 1700.0)   # m -> mm
        self.assertAlmostEqual(float(li[ni + 1][24:40]), 1654.231)
        # no .tmp_kunit sidecar files anywhere in the output dir
        leftovers = [f for f in os.listdir(os.path.dirname(out))
                     if f.endswith(".tmp_kunit")]
        self.assertEqual(leftovers, [])

    def test_multifile_include_tree_atomic_no_temp_left(self):
        d = tempfile.mkdtemp(prefix="kunit_t0_inc_")
        _write("*KEYWORD\n*NODE\n"
               "       1             1.0             0.0             0.0\n"
               "*END\n", "child.k", d)
        p = _write("*KEYWORD\n*INCLUDE\nchild.k\n*END\n", "parent.k", d)
        out = os.path.join(d, "parent_ton.k")
        ctx = convert(p, SI, TON, out, follow_includes=True, self_check=False)
        self.assertEqual(len(ctx.written), 2)
        child_out = os.path.join(d, "child__ton-mm-s.k")
        ci = _lines(child_out)
        self.assertAlmostEqual(float(ci[ci.index("*NODE") + 1][8:24]), 1000.0)
        self.assertEqual([f for f in os.listdir(d)
                          if f.endswith(".tmp_kunit")], [])


class ReportDimGuardTests(unittest.TestCase):
    def test_report_survives_bogus_dim_missing_from_dim_names(self):
        kf = KFile(_write(DECK_SI))
        ctx = scan(kf, SI)
        bogus = (9, 9, 9)          # a Dim intentionally absent from DIM_NAMES
        ctx.factors_used[bogus] = Fraction(3, 2)
        # must not raise TypeError on format(tuple, '<28')
        text = report(ctx, SI, TON)
        self.assertIn(str(bogus), text)


if __name__ == "__main__":
    unittest.main()
