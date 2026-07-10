"""T0 regression tests for kunit.detect header-unit parsing.

Covers the header token-order bug: the generic 'Unit system :' comment parser
used to pick the length/mass/time unit by *first* appearance in token order
rather than by specificity, so a header whose prose word "in" (the English
preposition) preceded the real unit "mm" resolved length to inches.
"""
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import detect, parse_system
from kunit.detect import _pick_unit, _TOKEN_RE


def _write(text, name="deck.k"):
    d = tempfile.mkdtemp(prefix="kunit_t0_detect_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


# A deck whose ONLY unit evidence is a generic header comment: no material
# cards, no curves, no gravity - so nothing physical can dominate the score.
_HEADER_ONLY = """{header}
*KEYWORD
*NODE
       1             1.7        1.654231       0.2799996       0       0
       2            -1.6             0.0      7.85000-2       0       0
*END
"""


class HeaderTokenOrderTests(unittest.TestCase):
    def test_pick_unit_prefers_specific_over_english_word(self):
        # This is the literal problem header from the bug report. Its token
        # list places the English preposition "in" before the real unit "mm".
        toks = [t.lower() for t in _TOKEN_RE.findall("$ units in mm, kg, ms")]
        self.assertEqual(toks, ["units", "in", "mm", "kg", "ms"])
        # length must resolve to mm, NOT to the "in" that merely precedes it.
        self.assertEqual(
            _pick_unit(toks, ("mm", "cm", "m", "in", "inch", "ft", "foot")),
            "mm")
        self.assertEqual(
            _pick_unit(toks, ("kg", "g", "ton", "tonne", "mg",
                              "lbm", "lb", "slug", "slinch")), "kg")
        self.assertEqual(_pick_unit(toks, ("s", "sec", "ms", "us", "µs")),
                         "ms")

    def test_pick_unit_falls_back_when_only_ambiguous(self):
        # When a dimension's ONLY candidate is an ambiguous token, keep it.
        toks = ["unit", "system", "m", "kg", "sec"]
        self.assertEqual(
            _pick_unit(toks, ("mm", "cm", "m", "in", "inch", "ft", "foot")),
            "m")
        self.assertEqual(_pick_unit(toks, ("s", "sec", "ms", "us", "µs")),
                         "sec")

    def test_generic_header_mm_kg_ms_via_detect(self):
        # Exercise the full generic-header path (not the kunit stamp). This
        # header triggers _HEADER_RE and has the same ambiguous token order
        # ('in' before 'mm'); the winning system must be kg-mm-ms with length
        # mm, never the slinch-in-s inch system.
        p = _write(_HEADER_ONLY.format(header="$ units : in mm, kg, ms"))
        v = detect(p)
        self.assertIsNotNone(v.system)
        self.assertEqual(v.system, parse_system("kg-mm-ms"))
        self.assertEqual(v.system.length_m, parse_system("kg-mm-ms").length_m)
        # length is mm (1e-3 m), definitely not inch (~0.0254 m)
        self.assertNotEqual(v.system, parse_system("slinch-in-s"))

    def test_generic_header_m_kg_s_regression(self):
        # The pre-existing tested header format must still resolve to kg-m-s.
        p = _write(_HEADER_ONLY.format(header="$ Unit system : m, kg, sec"))
        v = detect(p)
        self.assertIsNotNone(v.system)
        self.assertEqual(v.system, parse_system("kg-m-s"))


if __name__ == "__main__":
    unittest.main()
