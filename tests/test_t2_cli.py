import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kunit.cli as cli
from kunit.cli import parse_curve_overrides
from kunit.units import ACCEL, TIME


def F(*vals, w=10):
    """Fixed-width card line from values (mirrors test_kunit.py)."""
    return "".join(str(v).rjust(w) for v in vals)


# steel deck with clear SI evidence (density 7850, modulus 2.1e11)
STEEL_SI = ("*KEYWORD\n*MAT_PIECEWISE_LINEAR_PLASTICITY\n"
            + F(1, 7850.0, "2.1E11", 0.3, "3.5E8", 0.0, 0.0, 0.0) + "\n"
            + F(0.0, 0.0, 0, 0, 0) + "\n"
            "*NODE\n"
            "       1             1.7        1.654231       0.2799996       0       0\n"
            "*END\n")

# a *DEFINE_CURVE with no referencing keyword: its dims are unresolved and
# left unscaled unless a --curve override supplies them
UNREF_CURVE = ("*KEYWORD\n*DEFINE_CURVE\n"
               + F(5, 0, 1.0, 1.0, 0.0, 0.0, 0, 0) + "\n"
               + F("0.0", w=20) + F("9.81", w=20) + "\n*END\n")

# an unknown keyword forces a ConvertError unless --allow-unknown
UNKNOWN_KW = STEEL_SI.replace(
    "*END", "*AIRBAG_SIMPLE_PRESSURE_VOLUME\n         1\n*END")

# no usable unit evidence -> auto-detect returns None
NO_EVIDENCE = "*KEYWORD\n*TITLE\nnothing here\n*END\n"


def _write(text, name="deck.k", d=None):
    d = d or tempfile.mkdtemp(prefix="kunit_cli_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _run(argv):
    """Drive cli.main in-process; return (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


class CliConvertTests(unittest.TestCase):
    def test_default_output_path(self):
        p = _write(STEEL_SI)
        rc, out, _ = _run(["convert", p, "--to", "ton-mm-s",
                           "--from", "kg-m-s"])
        self.assertEqual(rc, 0)
        expected = os.path.join(os.path.dirname(p), "deck__ton-mm-s.k")
        self.assertTrue(os.path.exists(expected), out)

    def test_output_flag(self):
        p = _write(STEEL_SI)
        target = p + ".chosen.k"
        rc, _, _ = _run(["convert", p, "--to", "ton-mm-s",
                         "--from", "kg-m-s", "-o", target])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(target))

    def _read(self, path):
        with open(path, newline="") as fh:
            return fh.read()

    def test_in_place_keeps_backup(self):
        p = _write(STEEL_SI)
        before = self._read(p)
        rc, _, _ = _run(["convert", p, "--to", "ton-mm-s",
                         "--from", "kg-m-s", "--in-place"])
        self.assertEqual(rc, 0)
        self.assertNotEqual(self._read(p), before)           # input overwritten
        self.assertTrue(os.path.exists(p + ".orig_kg-m-s"))  # backup kept
        self.assertEqual(self._read(p + ".orig_kg-m-s"), before)

    def test_in_place_no_backup(self):
        p = _write(STEEL_SI)
        rc, _, _ = _run(["convert", p, "--to", "ton-mm-s",
                         "--from", "kg-m-s", "--in-place", "--no-backup"])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(p + ".orig_kg-m-s"))

    def test_in_place_and_output_mutually_exclusive(self):
        p = _write(STEEL_SI)
        rc, _, err = _run(["convert", p, "--to", "ton-mm-s", "--from",
                           "kg-m-s", "--in-place", "-o", p + ".x.k"])
        self.assertEqual(rc, 2)
        self.assertIn("mutually exclusive", err)
        self.assertFalse(os.path.exists(p + ".x.k"))

    def test_identical_systems_nothing_to_do(self):
        p = _write(STEEL_SI)
        rc, out, _ = _run(["convert", p, "--to", "kg-m-s", "--from", "kg-m-s"])
        self.assertEqual(rc, 0)
        self.assertIn("nothing to do", out)
        self.assertFalse(os.path.exists(
            os.path.join(os.path.dirname(p), "deck__kg-m-s.k")))

    def test_auto_detect_success(self):
        p = _write(STEEL_SI)
        target = p + ".auto.k"
        rc, out, _ = _run(["convert", p, "--to", "ton-mm-s", "-o", target])
        self.assertEqual(rc, 0)
        self.assertIn("auto-detected", out)
        self.assertTrue(os.path.exists(target))

    def test_auto_detect_not_confident(self):
        p = _write(NO_EVIDENCE)
        rc, _, err = _run(["convert", p, "--to", "ton-mm-s"])
        self.assertEqual(rc, 2)
        self.assertIn("not confident enough", err)

    def test_dry_run_writes_nothing(self):
        p = _write(STEEL_SI)
        target = p + ".dry.k"
        rc, out, _ = _run(["convert", p, "--to", "ton-mm-s",
                           "--from", "kg-m-s", "-o", target, "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out)
        self.assertFalse(os.path.exists(target))
        self.assertFalse(os.path.exists(target + ".kunit.log"))

    def test_log_written_by_default(self):
        p = _write(STEEL_SI)
        target = p + ".log.k"
        rc, _, _ = _run(["convert", p, "--to", "ton-mm-s",
                         "--from", "kg-m-s", "-o", target])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(target + ".kunit.log"))

    def test_no_log_suppresses_log(self):
        p = _write(STEEL_SI)
        target = p + ".nolog.k"
        rc, _, _ = _run(["convert", p, "--to", "ton-mm-s", "--from",
                         "kg-m-s", "-o", target, "--no-log"])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(target))
        self.assertFalse(os.path.exists(target + ".kunit.log"))

    def test_unknown_keyword_refused_then_allowed(self):
        p = _write(UNKNOWN_KW)
        rc, _, err = _run(["convert", p, "--to", "ton-mm-s", "--from",
                           "kg-m-s", "-o", p + ".u.k"])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR:", err)
        rc2, _, _ = _run(["convert", p, "--to", "ton-mm-s", "--from",
                          "kg-m-s", "-o", p + ".u2.k", "--allow-unknown"])
        self.assertEqual(rc2, 0)
        self.assertTrue(os.path.exists(p + ".u2.k"))

    def test_self_check_failed_returns_3(self):
        # force the post-write self-check to report a mismatching system by
        # replacing detect (which convert imports from .detect at call time)
        import importlib
        kdetect = importlib.import_module("kunit.detect")

        class _FakeVerdict:
            system = cli.parse_system("kg-m-s")   # != target ton-mm-s
            ambiguous = False

        orig = kdetect.detect
        kdetect.detect = lambda *a, **kw: _FakeVerdict()
        try:
            p = _write(STEEL_SI)
            rc, _, _ = _run(["convert", p, "--to", "ton-mm-s",
                             "--from", "kg-m-s", "-o", p + ".sc.k",
                             "--no-log"])
        finally:
            kdetect.detect = orig
        self.assertEqual(rc, 3)

    def test_curve_override_plumbed_through(self):
        p = _write(UNREF_CURVE)
        # without an override the ordinate stays 9.81 (unresolved dims)
        base = p + ".base.k"
        rc0, _, _ = _run(["convert", p, "--to", "ton-mm-s", "--from",
                          "kg-m-s", "-o", base, "--no-log"])
        self.assertEqual(rc0, 0)
        # with --curve the ordinate is scaled as an acceleration (x1000)
        over = p + ".over.k"
        rc, _, _ = _run(["convert", p, "--to", "ton-mm-s", "--from", "kg-m-s",
                         "-o", over, "--curve", "5=time:accel", "--no-log"])
        self.assertEqual(rc, 0)
        with open(over, newline="") as fh:
            lines = fh.read().split("\n")
        ci = lines.index("*DEFINE_CURVE")
        self.assertAlmostEqual(float(lines[ci + 2][20:40]), 9810.0)


class CliMiscTests(unittest.TestCase):
    def test_systems_lists_presets(self):
        rc, out, _ = _run(["systems"])
        self.assertEqual(rc, 0)
        self.assertIn("ton-mm-s", out)
        self.assertIn("kg-m-s", out)

    def test_parse_curve_overrides_valid(self):
        self.assertEqual(parse_curve_overrides(["17=time:accel"]),
                         {17: (TIME, ACCEL)})

    def test_parse_curve_overrides_malformed(self):
        with self.assertRaises(SystemExit):
            parse_curve_overrides(["nope"])

    def test_parse_curve_overrides_bad_dims(self):
        with self.assertRaises(SystemExit):
            parse_curve_overrides(["5=bad:dims"])


if __name__ == "__main__":
    unittest.main()
