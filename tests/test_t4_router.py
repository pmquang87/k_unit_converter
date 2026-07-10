"""Tier-4 router regression: pins the CURRENT resolve() classification for a
broad table of keyword names, so a refactor of the resolve() prefix-chain can be
proven behaviour-preserving.

Each case asserts the (kind, payload) verdict:
  * kind is the classification string {spec, custom, white, soft, hard, unknown};
  * for custom the payload must be the exact handler object;
  * for spec the payload must be the exact Spec object from SPECS;
  * for hard the payload must be the exact HARD_FLAGS message.

The table deliberately exercises the tricky precedence branches: HARD before
everything, dict lookups (with _TITLE/_ID stripping and _MAT_ALIASES numeric
aliases) before the prefix rules, and exact-WHITELIST / WHITELIST_PREFIXES
winning over the later CONTACT_/LOAD_BODY_/BOUNDARY_/DATABASE_/CONTROL_ rules.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kunit.schema import (
    resolve, SPECS, CUSTOM, HARD_FLAGS,
    h_contact, h_load_body, h_prescribed_motion, h_rigidwall_planar,
    h_database_dt, h_database_frequency, h_mat_cscm, h_define_table,
    h_define_curve,
)


class RouterVerdictTests(unittest.TestCase):
    def _kind(self, name):
        return resolve(name)[0]

    # ── HARD_FLAGS win over everything (checked first) ──────────────────────
    def test_hard_flags(self):
        for n in ("INCLUDE", "INCLUDE_TRANSFORM", "INCLUDE_PATH",
                  "PARAMETER", "PARAMETER_EXPRESSION",
                  "DEFINE_TRANSFORMATION", "DEFINE_FUNCTION",
                  "DEFINE_CURVE_FUNCTION"):
            kind, payload = resolve(n)
            self.assertEqual(kind, "hard", n)
            self.assertEqual(payload, HARD_FLAGS[n], n)

    def test_hard_beats_custom_prefix(self):
        # DEFINE_CURVE_FUNCTION is a HARD flag even though DEFINE_CURVE is a
        # CUSTOM handler - the exact HARD_FLAGS lookup happens first.
        self.assertEqual(resolve("DEFINE_CURVE_FUNCTION")[0], "hard")
        self.assertIs(resolve("DEFINE_CURVE")[1], h_define_curve)

    # ── dict lookups: alias + _TITLE/_ID stripping happen before prefixes ───
    def test_mat_numeric_alias_to_spec(self):
        # MAT_024 -> MAT_PIECEWISE_LINEAR_PLASTICITY (a Spec)
        kind, payload = resolve("MAT_024")
        self.assertEqual(kind, "spec")
        self.assertIs(payload, SPECS["MAT_PIECEWISE_LINEAR_PLASTICITY"])
        # canonical name resolves to the same Spec object
        self.assertIs(resolve("MAT_PIECEWISE_LINEAR_PLASTICITY")[1], payload)

    def test_title_suffix_stripped(self):
        # _TITLE stripped, then aliased, then Spec-looked-up
        self.assertIs(resolve("MAT_024_TITLE")[1],
                      SPECS["MAT_PIECEWISE_LINEAR_PLASTICITY"])

    def test_id_suffix_stripped_to_custom(self):
        self.assertIs(resolve("DEFINE_CURVE_TITLE")[1], h_define_curve)
        self.assertIs(resolve("DEFINE_CURVE_ID")[1], h_define_curve)

    def test_mat_alias_to_custom(self):
        # MAT_159_CONCRETE -> MAT_CSCM_CONCRETE (a CUSTOM handler)
        self.assertIs(resolve("MAT_159_CONCRETE")[1], h_mat_cscm)

    def test_define_table_alias(self):
        self.assertIs(resolve("DEFINE_TABLE_2D")[1], h_define_table)

    def test_spec_payload_identity(self):
        for n in ("NODE", "CONTROL_TIMESTEP", "CONTROL_TERMINATION"):
            kind, payload = resolve(n)
            self.assertEqual(kind, "spec", n)
            self.assertIs(payload, SPECS[n], n)

    # ── CONTACT_ family ─────────────────────────────────────────────────────
    def test_contact_custom(self):
        for n in ("CONTACT_AUTOMATIC_SINGLE_SURFACE",
                  "CONTACT_AUTOMATIC_SINGLE_SURFACE_MORTAR",
                  # a NON-leading TIEBREAK still reaches h_contact (current code)
                  "CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_TIEBREAK"):
            kind, payload = resolve(n)
            self.assertEqual(kind, "custom", n)
            self.assertIs(payload, h_contact, n)

    def test_contact_unknown(self):
        for n in ("CONTACT_TIEBREAK_NODES_ONLY",
                  "CONTACT_AUTOMATIC_SINGLE_SURFACE_DRAWBEAD",
                  "CONTACT_AUTOMATIC_SINGLE_SURFACE_MPP",
                  "CONTACT_AUTOMATIC_SINGLE_SURFACE_DAMPING",
                  "CONTACT_2D_AUTOMATIC_SURFACE_TO_SURFACE",
                  "CONTACT_ENTITY", "CONTACT_GEBOD_LOWER",
                  "CONTACT_INTERIOR", "CONTACT_GUIDED_CABLE",
                  "CONTACT_COUPLING", "CONTACT_AUTO_MOVE"):
            self.assertEqual(resolve(n), ("unknown", None), n)

    # ── LOAD_BODY_ family ───────────────────────────────────────────────────
    def test_load_body_custom(self):
        for n in ("LOAD_BODY_X", "LOAD_BODY_Y", "LOAD_BODY_Z",
                  "LOAD_BODY_RX", "LOAD_BODY_RY", "LOAD_BODY_RZ"):
            self.assertIs(resolve(n)[1], h_load_body, n)
            self.assertEqual(resolve(n)[0], "custom", n)

    def test_load_body_unknown(self):
        for n in ("LOAD_BODY_GENERALIZED", "LOAD_BODY_POROUS",
                  "LOAD_BODY_GENERALIZED_SET_NODE"):
            self.assertEqual(resolve(n), ("unknown", None), n)

    def test_load_body_parts_whitelist_wins(self):
        # LOAD_BODY_PARTS is in the exact WHITELIST -> white, even though the
        # LOAD_BODY_ prefix rule (tail PARTS) would otherwise say unknown.
        self.assertEqual(resolve("LOAD_BODY_PARTS"), ("white", None))

    # ── BOUNDARY_PRESCRIBED_MOTION family ───────────────────────────────────
    def test_prescribed_motion_custom(self):
        for n in ("BOUNDARY_PRESCRIBED_MOTION",
                  "BOUNDARY_PRESCRIBED_MOTION_SET",
                  "BOUNDARY_PRESCRIBED_MOTION_RIGID",
                  "BOUNDARY_PRESCRIBED_MOTION_NODE"):
            self.assertIs(resolve(n)[1], h_prescribed_motion, n)
            self.assertEqual(resolve(n)[0], "custom", n)

    def test_prescribed_motion_unknown(self):
        for n in ("BOUNDARY_PRESCRIBED_MOTION_SET_BOX",
                  "BOUNDARY_PRESCRIBED_MOTION_SET_EDGE_UVW",
                  "BOUNDARY_PRESCRIBED_MOTION_SET_FACE_XYZ",
                  "BOUNDARY_PRESCRIBED_MOTION_SET_LINE",
                  "BOUNDARY_PRESCRIBED_MOTION_SET_POINT_UVW"):
            self.assertEqual(resolve(n), ("unknown", None), n)

    # ── RIGIDWALL_PLANAR ────────────────────────────────────────────────────
    def test_rigidwall(self):
        self.assertIs(resolve("RIGIDWALL_PLANAR")[1], h_rigidwall_planar)
        self.assertIs(resolve("RIGIDWALL_PLANAR_ORTHO")[1], h_rigidwall_planar)
        self.assertEqual(resolve("RIGIDWALL_GEOMETRIC"), ("unknown", None))

    # ── DATABASE_ families ──────────────────────────────────────────────────
    def test_database_binary(self):
        self.assertIs(resolve("DATABASE_BINARY_D3PLOT")[1], h_database_dt)
        for tail in ("D3PART", "D3THDT", "INTFOR", "FSIFOR", "BLSTFOR",
                     "D3MEAN"):
            self.assertIs(resolve("DATABASE_BINARY_" + tail)[1], h_database_dt)
        for tail in ("D3DUMP", "RUNRSF", "D3DRLF", "D3PROP"):
            self.assertEqual(resolve("DATABASE_BINARY_" + tail),
                             ("white", None), tail)
        self.assertEqual(resolve("DATABASE_BINARY_FOO"), ("unknown", None))

    def test_database_ascii_custom(self):
        for tail in ("GLSTAT", "MATSUM", "RCFORC", "ELOUT", "NODOUT"):
            kind, payload = resolve("DATABASE_" + tail)
            self.assertEqual(kind, "custom", tail)
            self.assertIs(payload, h_database_dt, tail)

    def test_database_nodal_force_group_whitelist_wins(self):
        # In the exact WHITELIST -> white, before the DATABASE_ rule (unknown).
        self.assertEqual(resolve("DATABASE_NODAL_FORCE_GROUP"),
                         ("white", None))

    def test_database_frequency(self):
        self.assertIs(resolve("DATABASE_FREQUENCY_BINARY_D3SSD")[1],
                      h_database_frequency)
        self.assertEqual(resolve("DATABASE_FREQUENCY_ELOUT"),
                         ("unknown", None))

    def test_database_other_unknown(self):
        self.assertEqual(resolve("DATABASE_TRACER"), ("unknown", None))

    # ── WHITELIST_PREFIXES win over later prefix rules ──────────────────────
    def test_whitelist_prefixes(self):
        for n in ("SET_NODE", "SET_PART_LIST", "BOUNDARY_SPC_NODE",
                  "DATABASE_HISTORY_NODE", "CONTROL_MPP_DECOMPOSITION",
                  "DEFORMABLE_TO_RIGID", "INTERFACE_SPRINGBACK",
                  "DATABASE_EXTENT_SSSTAT"):
            self.assertEqual(resolve(n), ("white", None), n)

    def test_control_mpp_prefix_beats_control_soft(self):
        # CONTROL_MPP_ is a WHITELIST_PREFIX (white) checked before the generic
        # CONTROL_ rule (soft).
        self.assertEqual(resolve("CONTROL_MPP_DECOMPOSITION"),
                         ("white", None))

    def test_database_history_prefix_beats_database_unknown(self):
        self.assertEqual(resolve("DATABASE_HISTORY_SHELL"), ("white", None))

    # ── CONTROL_ soft fallback + exact whitelist ────────────────────────────
    def test_control_soft(self):
        for n in ("CONTROL_ADAPTIVE", "CONTROL_STAGED_CONSTRUCTION",
                  "CONTROL_DAMPING_GLOBAL"):
            kind, payload = resolve(n)
            self.assertEqual(kind, "soft", n)
            self.assertIn("not in the dimension table", payload)

    def test_control_exact_whitelist(self):
        for n in ("CONTROL_ENERGY", "CONTROL_OUTPUT", "CONTROL_CONTACT"):
            self.assertEqual(resolve(n), ("white", None), n)

    # ── plain whitelist + truly unknown ─────────────────────────────────────
    def test_plain_whitelist(self):
        for n in ("KEYWORD", "TITLE", "END", "ELEMENT_SHELL"):
            self.assertEqual(resolve(n), ("white", None), n)

    def test_unknown(self):
        for n in ("TOTALLY_UNKNOWN_KW", "FOO_BAR", "WIDGET_9000"):
            self.assertEqual(resolve(n), ("unknown", None), n)


if __name__ == "__main__":
    unittest.main()
