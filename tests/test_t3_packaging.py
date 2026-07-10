"""Tier-3 packaging / DX checks: version metadata, the --version flag, and
that the shipped example decks are detectable."""
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

import kunit
from kunit.cli import main as cli_main

# NB: kunit/__init__.py binds the ``detect`` *function* as ``kunit.detect``,
# shadowing the submodule attribute, so fetch the module explicitly.
detect_mod = importlib.import_module("kunit.detect")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "examples", "*.k")))


class VersionTests(unittest.TestCase):
    def test_version_is_nonempty_str(self):
        self.assertIsInstance(kunit.__version__, str)
        self.assertTrue(kunit.__version__.strip())

    def test_version_flag_exits_zero_and_prints(self):
        # argparse action="version" prints to stdout and raises SystemExit(0)
        from io import StringIO
        buf = StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            with self.assertRaises(SystemExit) as cm:
                cli_main(["--version"])
        finally:
            sys.stdout = old
        code = cm.exception.code
        self.assertIn(code, (0, None))
        self.assertIn(kunit.__version__, buf.getvalue())


class ExampleDeckTests(unittest.TestCase):
    def test_examples_exist(self):
        self.assertTrue(EXAMPLES, "no example decks found under examples/")

    def test_examples_are_detectable(self):
        for path in EXAMPLES:
            with self.subTest(deck=os.path.basename(path)):
                verdict = detect_mod.detect(path)
                # a Verdict is always returned (never raises); it carries the
                # ranked evidence even when a single system can't be pinned
                self.assertTrue(hasattr(verdict, "system"))
                self.assertTrue(hasattr(verdict, "ranked"))
                self.assertIsNotNone(verdict.system,
                                     f"{path} did not detect a unit system")


if __name__ == "__main__":
    unittest.main()
