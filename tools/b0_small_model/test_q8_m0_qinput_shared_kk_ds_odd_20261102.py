#!/usr/bin/env python3
"""Static checks for the 20261102 DS vs coherent J_odd compare. No long run."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import build_geometry

ROOT = Path(__file__).resolve().parents[2]


class DsOddStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart(sh.K_FRAC)
        cls.ok, cls.gate = ds.input_gate(cls.geo, cls.k)

    def test_jobs_are_coherent_and_ds_theta0_both_neel(self):
        self.assertEqual(
            ds.JOBS,
            (("coh_p", "coherent", 1), ("coh_m", "coherent", -1),
             ("ds_p", "ds", 1), ("ds_m", "ds", -1)))
        self.assertEqual(ds.THETA, 0.0)
        self.assertEqual(ds.R_SQ, 0.3)
        self.assertEqual(ds.ALPHA, 0.5 + 0j)

    def test_squeeze_axis_is_theta_over_two(self):
        scan = {row["theta"]: row["axis_deg"] for row in self.gate["photon"]["theta_scan"]}
        self.assertAlmostEqual(scan[0.0], 0.0, places=12)
        self.assertAlmostEqual(scan[0.5 * np.pi], 45.0, places=12)
        self.assertAlmostEqual(scan[np.pi], 90.0, places=12)

    def test_large_space_omission_matches_external_theta0(self):
        row0 = self.gate["photon"]["theta_scan"][0]
        self.assertEqual(row0["theta"], 0.0)
        self.assertAlmostEqual(row0["large80_P_n_gt_8"], 2.63e-6, delta=2e-8)
        self.assertLess(row0["large80_P_n_gt_8"], ds.OMIT_GATE)
        row_pi2 = self.gate["photon"]["theta_scan"][1]
        self.assertGreater(row_pi2["large80_P_n_gt_8"], ds.OMIT_GATE)

    def test_code_trunc_omission_is_not_p_n_gt_8(self):
        row0 = self.gate["photon"]["theta_scan"][0]
        self.assertGreater(row0["code_trunc8_omitted_unnormalized"],
                           row0["large80_P_n_gt_8"])
        self.assertLess(row0["rel_diff_to_external"], 0.01)

    def test_electron_product_maps(self):
        el = self.gate["electron"]
        self.assertGreater(el["abs_Swap_PoxP_product"], 1.0 - 1e-12)
        self.assertGreater(el["abs_P_ek_plus_vs_emk_minus"], 1.0 - 1e-12)
        self.assertGreater(el["abs_P_emk_plus_vs_ek_minus"], 1.0 - 1e-12)
        self.assertLess(el["h8_P_k_plus_vs_mk_minus"], 1e-15)

    def test_gate_passes(self):
        self.assertTrue(self.ok)

    def test_even_odd_dummy(self):
        jp = np.array([[1.0, 0.2], [1.2, 0.4]], complex)
        jm = np.array([[0.4, -0.2], [0.6, 0.0]], complex)
        even, odd = ds.even_odd(jp, jm)
        np.testing.assert_allclose(even, 0.5 * (jp + jm))
        np.testing.assert_allclose(odd, 0.5 * (jp - jm))

    def test_envelope_is_one_cycle_rise(self):
        env = ds.dyn.Envelope(ds.pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        self.assertAlmostEqual(env.s(0.0), 0.0)
        self.assertAlmostEqual(env.s(env.t_ramp), 1.0)
        self.assertEqual(env.flat_cycles, 0.0)

    def test_old_archives_exist_and_are_not_this_report(self):
        for p in ds.OLD_ARCHIVES:
            self.assertTrue(p.is_file(), msg=str(p))
        dest = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json"
        self.assertNotIn(dest, ds.OLD_ARCHIVES)


if __name__ == "__main__":
    unittest.main()
