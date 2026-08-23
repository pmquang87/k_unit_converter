"""T5 tests: parameter-aware conversion and detection.

*PARAMETER values are scaled by the dimension of the fields that reference
them (``&name`` / ``-&name``); the references themselves stay untouched.
Detection resolves parametrised densities/moduli through the same table.
Without ``parameters=True`` the old hard stop is unchanged.
"""
import os
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit import ConvertError, convert, detect, parse_system
from kunit.cli import main as cli_main

SI = parse_system("kg-m-s")
TON = parse_system("ton-mm-s")


def F(*vals, w=10):
    return "".join(str(v).rjust(w) for v in vals)


def _write(text, name="deck.k", d=None):
    d = d or tempfile.mkdtemp(prefix="kunit_t5_")
    p = os.path.join(d, name)
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return p


def _read(path):
    with open(path, newline="") as fh:
        return fh.read()


def _deck(param_lines, body):
    return "*KEYWORD\n*PARAMETER\n" + "\n".join(param_lines) + "\n" + body + "*END\n"


BODY = (
    "*MAT_ELASTIC\n" + F(1, "&RHO", "&E", "&NU") + "\n"
    "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK", "&THK", "&THK", "&THK") + "\n"
    "*INITIAL_VELOCITY\n" + F(0) + "\n" + F("-&V0", 0.0, 0.0) + "\n"
    "*NODE\n" + F(1, 0.0, 0.0, 0.0, w=8) + "\n"
)


class ParameterScalingTests(unittest.TestCase):
    def test_values_scaled_by_the_fields_they_feed(self):
        p = _write(_deck([F("R RHO", 7850.0, "R E", "2.1E11"),
                          F("R NU", 0.3, "R THK", 0.002),
                          F("R V0", 15.0)], BODY))
        out = p + ".ton.k"
        ctx = convert(p, SI, TON, out, parameters=True, self_check=False)
        txt = _read(out)
        lines = txt.split("\n")
        # the *PARAMETER card carries the converted values ...
        card1 = lines[[i for i, l in enumerate(lines)
                       if l.startswith("*PARAMETER")][0] + 1]
        self.assertAlmostEqual(float(card1[10:20]), 7.85e-9, delta=1e-13)
        self.assertAlmostEqual(float(card1[30:40]), 210000.0, delta=1e-6)
        card2 = lines[[i for i, l in enumerate(lines)
                       if l.startswith("*PARAMETER")][0] + 2]
        self.assertEqual(float(card2[10:20]), 0.3)        # dimensionless use
        self.assertAlmostEqual(float(card2[30:40]), 2.0)  # m -> mm
        card3 = lines[[i for i, l in enumerate(lines)
                       if l.startswith("*PARAMETER")][0] + 3]
        self.assertAlmostEqual(float(card3[10:20]), 15000.0)  # m/s -> mm/s
        # ... and the references are untouched
        self.assertIn("&RHO", txt)
        self.assertIn("-&V0", txt)
        self.assertIn("&THK", txt)
        self.assertFalse(ctx.errors)
        self.assertTrue(any("RHO" in n for n in ctx.notes))

    def test_conflicting_uses_are_refused(self):
        body = ("*MAT_ELASTIC\n" + F(1, "&X", "2.1E11", 0.3) + "\n"
                "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&X") + "\n")
        p = _write(_deck([F("R X", 1.0)], body))
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("X", str(cm.exception))
        self.assertIn("conflict", str(cm.exception).lower())
        self.assertFalse(os.path.exists(p + ".o.k"))

    def test_expression_parameter_is_wrapped(self):
        body = "*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n"
        text = ("*KEYWORD\n*PARAMETER\n" + F("R A", 7850.0) + "\n"
                "*PARAMETER_EXPRESSION\n" + "R RHO".ljust(10) + "A*1.0\n"
                + body + "*END\n")
        p = _write(text)
        ctx = convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        lines = _read(p + ".o.k").split("\n")
        ln = lines[lines.index("*PARAMETER_EXPRESSION") + 1]
        self.assertEqual(ln[:10], "R RHO".ljust(10))
        self.assertEqual(ln[10:], "(A*1.0)*0.000000000001")
        self.assertTrue(any("wrapped" in n for n in ctx.notes))
        # the rewritten expression must evaluate to source_value * factor
        # with the (unchanged) input parameter table
        val = eval(ln[10:].replace("A", "7850.0"))
        self.assertAlmostEqual(val, 7850.0 * 1e-12, delta=1e-21)

    def test_expression_parameter_feeding_another_expression_is_refused(self):
        body = ("*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n"
                "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK") + "\n")
        text = ("*KEYWORD\n*PARAMETER\n" + F("R A", 7850.0) + "\n"
                "*PARAMETER_EXPRESSION\n" + "R RHO".ljust(10) + "A*1.0\n"
                + "R THK".ljust(10) + "RHO*1.0e-6\n" + body + "*END\n")
        p = _write(text)
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("double-scaling", str(cm.exception))

    def test_expression_free_format_and_continuation(self):
        body = ("*INITIAL_VELOCITY\n" + F(0) + "\n" + F("&V0", 0.0, 0.0) + "\n"
                "*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n")
        text = ("*KEYWORD\n*PARAMETER\nrvel,15.0,ra,7850.0\n"
                "*PARAMETER_EXPRESSION\nrv0,vel*1.0\n"
                + "R RHO".ljust(10) + "a*1.0+0.0*(1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+\n"
                + " " * 10 + "1.0)\n" + body + "*END\n")
        p = _write(text)
        convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        out = _read(p + ".o.k")
        self.assertIn("rv0,(vel*1.0)*1000.0", out)
        lines = out.split("\n")
        i = lines.index("*PARAMETER_EXPRESSION")
        joined = lines[i + 2][10:] + lines[i + 3][10:]
        self.assertTrue(joined.startswith("(a*1.0+0.0*(1.0+"), joined)
        self.assertTrue(joined.endswith("1.0))*0.000000000001"), joined)

    def test_integer_expression_parameter_is_refused(self):
        body = "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK") + "\n"
        text = ("*KEYWORD\n*PARAMETER\n" + F("I N", 2) + "\n"
                "*PARAMETER_EXPRESSION\n" + "I THK".ljust(10) + "N*2\n"
                + body + "*END\n")
        p = _write(text)
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("only real (R) expressions", str(cm.exception))

    def test_expression_factor_is_a_plain_decimal(self):
        # slinch-in-s -> ton-mm-s: length factor 25.4, pressure 0.00689475...
        slin = parse_system("slinch-in-s")
        body = ("*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK") + "\n"
                "*MAT_ELASTIC\n" + F(1, "7.3E-4", "&E", 0.3) + "\n")
        text = ("*KEYWORD\n*PARAMETER\n" + F("R T0", 0.1, "R E0", "1.0E7") + "\n"
                "*PARAMETER_EXPRESSION\n" + "R THK".ljust(10) + "T0*1.0\n"
                + "R E".ljust(10) + "E0*1.0\n" + body + "*END\n")
        p = _write(text)
        convert(p, slin, TON, p + ".o.k", parameters=True, self_check=False)
        lines = _read(p + ".o.k").split("\n")
        i = lines.index("*PARAMETER_EXPRESSION")
        self.assertEqual(lines[i + 1][10:], "(T0*1.0)*25.4")
        fac = lines[i + 2][10:].split("*")[-1]
        self.assertNotIn("E", fac.upper())
        self.assertAlmostEqual(float(fac), 6894.757293168 * 1e-6 * 1e3 / 1e3, places=12)

    def test_roundtrip_with_parameters(self):
        p = _write(_deck([F("R RHO", 7850.0, "R E", "2.1E11"),
                          F("R NU", 0.3, "R THK", 0.002), F("R V0", 15.0)], BODY))
        ctx = convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False,
                      verify_roundtrip=True)
        self.assertTrue(ctx.roundtrip.startswith("OK"), ctx.roundtrip)

    def test_expression_input_also_scaled_is_refused(self):
        # forward guard: THK feeds a dimensional field (rescaled in place)
        # AND is an input of THK2 - wrapping THK2 would double-scale it
        body = ("*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK", "&THK2") + "\n"
                "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n")
        text = ("*KEYWORD\n*PARAMETER\n" + F("R THK", 0.002) + "\n"
                "*PARAMETER_EXPRESSION\n" + "R THK2".ljust(10) + "2.0*thk\n"
                + body + "*END\n")
        p = _write(text)
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("scale it twice", str(cm.exception))

    def test_self_referencing_expression_is_refused(self):
        body = "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK") + "\n"
        text = ("*KEYWORD\n*PARAMETER\n" + F("R T0", 0.002) + "\n"
                "*PARAMETER_EXPRESSION_MUTABLE\n" + "R THK".ljust(10) + "T0*1.0\n"
                + "R THK".ljust(10) + "THK*2.0\n" + body + "*END\n")
        p = _write(text)
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("redefined in terms of itself", str(cm.exception))

    def test_mixed_plain_and_expression_definition_is_refused(self):
        body = "*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n"
        text = ("*KEYWORD\n*PARAMETER_MUTABLE\n" + F("R RHO", 7850.0, "R A", 7850.0) + "\n"
                "*PARAMETER_EXPRESSION_MUTABLE\n" + "R RHO".ljust(10) + "A*1.0\n"
                + body + "*END\n")
        p = _write(text)
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("both *PARAMETER and", str(cm.exception))

    def test_fixed_format_expression_with_two_argument_function(self):
        # a comma inside max()/min() must not flip the card to free format
        body = "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&THK") + "\n"
        text = ("*KEYWORD\n*PARAMETER\n" + F("R A", 0.002) + "\n"
                "*PARAMETER_EXPRESSION\n" + "R THK".ljust(10) + "max(a,0.001)\n"
                + body + "*END\n")
        p = _write(text)
        ctx = convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        lines = _read(p + ".o.k").split("\n")
        ln = lines[lines.index("*PARAMETER_EXPRESSION") + 1]
        self.assertEqual(ln[10:], "(max(a,0.001))*1000.0")
        self.assertFalse(ctx.errors)

    def test_inline_bracket_expression_is_refused(self):
        text = ("*KEYWORD\n*PARAMETER\nrthk,0.002\n"
                "*SECTION_SHELL\n1,2\n<thk*1.0>,0.0\n"
                "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n*END\n")
        p = _write(text)
        with self.assertRaisesRegex(ConvertError, "inline <expression>"):
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        # default mode: same refusal (deck without a *PARAMETER block,
        # which would otherwise hard-stop first)
        p2 = _write("*KEYWORD\n*SECTION_SHELL\n1,2\n<0.002*1.0>,0.0\n"
                    "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n*END\n")
        with self.assertRaisesRegex(ConvertError, "inline <expression>"):
            convert(p2, SI, TON, p2 + ".o.k", self_check=False)

    def test_local_definitions_in_two_files_are_refused(self):
        d = tempfile.mkdtemp(prefix="kunit_t5_")
        _write("*KEYWORD\n*PARAMETER_LOCAL\n" + F("R T", 5.0) + "\n"
               "*SECTION_SHELL\n" + F(2, 2) + "\n" + F(0.001) + "\n*END\n",
               name="other.k", d=d)
        main = _write("*KEYWORD\n*PARAMETER_LOCAL\n" + F("R T", 0.002) + "\n"
                      "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&T") + "\n"
                      "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n"
                      "*INCLUDE\nother.k\n*END\n", name="main.k", d=d)
        with self.assertRaises(ConvertError) as cm:
            convert(main, SI, TON, os.path.join(d, "out.k"), parameters=True,
                    follow_includes=True, self_check=False)
        self.assertIn("LOCAL", str(cm.exception))

    def test_reference_in_unscaled_field_is_refused(self):
        # &X feeds E (scaled) but also PR (dimensionless - never scaled):
        # the PR use would silently receive the rescaled value
        body = "*MAT_ELASTIC\n" + F(1, 7850.0, "&X", "&X") + "\n"
        p = _write(_deck([F("R X", 0.3)], body))
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("does not scale", str(cm.exception))

    def test_chunked_expression_splits_at_operator_boundaries(self):
        long_expr = "a*1.0+0.0*(1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0)"
        body = "*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n"
        text = ("*KEYWORD\n*PARAMETER\n" + F("R A", 7850.0) + "\n"
                "*PARAMETER_EXPRESSION\n" + "R RHO".ljust(10) + long_expr[:60] + "\n"
                + " " * 10 + long_expr[60:] + "\n" + body + "*END\n")
        p = _write(text)
        convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        lines = _read(p + ".o.k").split("\n")
        i = lines.index("*PARAMETER_EXPRESSION")
        first, second = lines[i + 1][10:], lines[i + 2][10:]
        self.assertEqual(first + second, "(" + long_expr + ")*0.000000000001")
        # no numeric token may straddle the line break
        self.assertIn(first[-1], "+-*/(),")

    def test_expression_parameter_in_dimensionless_field_is_fine(self):
        body = "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", "&NU") + "\n"
        text = ("*KEYWORD\n*PARAMETER\n" + F("R A", 0.3) + "\n"
                "*PARAMETER_EXPRESSION\n" + "R NU".ljust(10) + "A*1.0\n"
                + body + "*END\n")
        p = _write(text)
        convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("A*1.0", _read(p + ".o.k"))

    def test_undefined_parameter_is_refused(self):
        body = "*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n"
        p = _write("*KEYWORD\n" + body + "*END\n")
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("RHO", str(cm.exception))
        self.assertIn("not defined", str(cm.exception))

    def test_integer_parameter_cannot_be_scaled(self):
        body = "*SECTION_SHELL\n" + F(1, 2) + "\n" + F("&N") + "\n"
        p = _write(_deck([F("I N", 3)], body))
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("integer", str(cm.exception).lower())

    def test_free_format_and_local_in_include(self):
        d = tempfile.mkdtemp(prefix="kunit_t5_")
        _write("*KEYWORD\n*PARAMETER_LOCAL\nrthk,0.002,rnu,0.3\n"
               "*SECTION_SHELL\n" + F(1, 2) + "\n"
               + F("&thk", "&thk", "&thk", "&thk") + "\n*END\n",
               name="sec.k", d=d)
        main = _write("*KEYWORD\n*PARAMETER\n" + F("R rho", 7850.0, "R E", "2.1E11")
                      + "\n*MAT_ELASTIC\n" + F(1, "&RHO", "&e", "&NU") + "\n"
                      "*INCLUDE\nsec.k\n*END\n", name="main.k", d=d)
        out = os.path.join(d, "out.k")
        convert(main, SI, TON, out, parameters=True, follow_includes=True,
                self_check=False)
        self.assertIn("rthk,2", _read(os.path.join(d, "sec__ton-mm-s.k")))
        self.assertIn("&thk", _read(os.path.join(d, "sec__ton-mm-s.k")))
        lines = _read(out).split("\n")
        card = lines[lines.index("*PARAMETER") + 1]
        self.assertAlmostEqual(float(card[10:20]), 7.85e-9, delta=1e-13)

    def test_unused_parameters_are_left_alone(self):
        body = "*MAT_ELASTIC\n" + F(1, 7850.0, "2.1E11", 0.3) + "\n"
        p = _write(_deck([F("R DUMMY", 42.0, "C NAME", "abc")], body))
        convert(p, SI, TON, p + ".o.k", parameters=True, self_check=False)
        self.assertIn("42.0", _read(p + ".o.k"))
        self.assertIn("abc", _read(p + ".o.k"))

    def test_default_mode_keeps_the_hard_stop(self):
        p = _write(_deck([F("R RHO", 7850.0)],
                         "*MAT_ELASTIC\n" + F(1, "&RHO", "2.1E11", 0.3) + "\n"))
        with self.assertRaises(ConvertError) as cm:
            convert(p, SI, TON, p + ".o.k", self_check=False)
        self.assertIn("PARAMETER", str(cm.exception))

    def test_cli_flag(self):
        p = _write(_deck([F("R RHO", 7850.0, "R E", "2.1E11")],
                         "*MAT_ELASTIC\n" + F(1, "&RHO", "&E", 0.3) + "\n"))
        out = p + ".cli.k"
        rc = cli_main(["convert", p, "--from", "kg-m-s", "--to", "ton-mm-s",
                       "-o", out, "--parameters", "--no-self-check", "--no-log"])
        self.assertEqual(rc, 0)
        self.assertIn("&RHO", _read(out))


class ParameterDetectionTests(unittest.TestCase):
    def test_detect_resolves_parametrised_density_and_modulus(self):
        p = _write(_deck([F("R RHO", 7850.0, "R E", "2.1E11")],
                         "*MAT_ELASTIC\n" + F(1, "&RHO", "&E", 0.3) + "\n"))
        v = detect(p, follow_includes=False, use_headers=False)
        self.assertIsNotNone(v.system)
        self.assertEqual(v.system.key, "kg-m-s")
        self.assertFalse(v.ambiguous)

    def test_negated_reference_is_resolved(self):
        from kunit.convert import load_tree
        p = _write(_deck([F("R RHO", 7850.0)],
                         "*MAT_ELASTIC\n" + F(1, "-&RHO", "2.1E11", 0.3) + "\n"))
        files, _ = load_tree(p, False)
        kf = files[0]
        mat = [b for b in kf.blocks if b.name == "MAT_ELASTIC"][0]
        self.assertEqual(kf.get_number(mat.data[0], [10] * 8, False, 1),
                         Decimal("-7850.0"))

    def test_self_check_passes_on_parametrised_deck(self):
        p = _write(_deck([F("R RHO", 7850.0, "R E", "2.1E11")],
                         "*MAT_ELASTIC\n" + F(1, "&RHO", "&E", 0.3) + "\n"))
        ctx = convert(p, SI, TON, p + ".o.k", parameters=True)
        self.assertTrue(ctx.self_check.startswith("OK"), ctx.self_check)


if __name__ == "__main__":
    unittest.main()
