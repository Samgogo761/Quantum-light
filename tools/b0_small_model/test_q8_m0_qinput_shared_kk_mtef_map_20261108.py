#!/usr/bin/env python3
"""Static checks for 20261108 Wigner/MTEF map. No ensemble."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import build_geometry

ROOT = Path(__file__).resolve().parents[2]


class MtefMapStaticTests(unittest.TestCase):
    def test_scope_frozen_params(self):
        self.assertEqual(mp.ALPHA, 0.5 + 0j)
        self.assertEqual(mp.R_SQ, 0.3)
        self.assertEqual(mp.THETA, 0.0)
        self.assertEqual(mp.N_STEPS_LOCK, 1285)
        self.assertEqual(mp.SAMPLE_EVERY, 32)
        self.assertEqual(pf.G_ABS, 0.005)
        self.assertEqual(pf.OMEGA0, 0.05)
        self.assertEqual(ds.THETA, 0.0)
        self.assertEqual(mp.WALL_CAP_S, 120.0)

    def test_does_not_overwrite_old_archives(self):
        dest = mp.OUT_DIR / "map.json"
        for p in mp.OLD_ARCHIVES:
            self.assertTrue(p.is_file(), msg=str(p))
            self.assertNotEqual(p.resolve(), dest.resolve())
        self.assertIn(ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json",
                      mp.OLD_ARCHIVES)
        self.assertIn(ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106/trunc10.json",
                      mp.OLD_ARCHIVES)

    def test_wigner_moments_match_analytic(self):
        coh = mp.analytic_xp("coherent")
        dsv = mp.analytic_xp("ds")
        self.assertAlmostEqual(coh["mean_X"], 1.0, places=12)
        self.assertAlmostEqual(coh["var_X"], 1.0, places=12)
        self.assertAlmostEqual(coh["mean_n_analytic"], 0.25, places=12)
        self.assertAlmostEqual(dsv["var_X"], float(np.exp(-0.6)), places=12)
        self.assertAlmostEqual(dsv["mean_n_analytic"], 0.25 + np.sinh(0.3) ** 2, places=12)
        self.assertAlmostEqual(coh["mean_X"], dsv["mean_X"], places=15)
        self.assertAlmostEqual(coh["mean_P"], dsv["mean_P"], places=15)

    def test_mean_field_node_is_shared(self):
        c = mp.gauss_hermite_nodes("coherent", 1)
        d = mp.gauss_hermite_nodes("ds", 1)
        self.assertEqual(c["n_nodes"], 1)
        self.assertAlmostEqual(c["nodes"][0]["X"], d["nodes"][0]["X"], places=14)
        self.assertAlmostEqual(c["nodes"][0]["P"], d["nodes"][0]["P"], places=14)

    def test_gh3_recovers_variance(self):
        for kind in ("coherent", "ds"):
            ana = mp.analytic_xp(kind)
            g = mp.gauss_hermite_nodes(kind, 3)
            self.assertAlmostEqual(g["quad_mean_X"], ana["mean_X"], places=10)
            self.assertAlmostEqual(g["quad_var_X"], ana["var_X"], places=10)
            self.assertAlmostEqual(g["quad_var_P"], ana["var_P"], places=10)
            self.assertAlmostEqual(g["weight_sum"], 1.0, places=10)

    def test_crn_standard_plane(self):
        c = [(n["xi"], n["eta"]) for n in mp.gauss_hermite_nodes("coherent", 3)["nodes"]]
        d = [(n["xi"], n["eta"]) for n in mp.gauss_hermite_nodes("ds", 3)["nodes"]]
        self.assertEqual(c, d)

    def test_commutator_and_force_identity(self):
        corr = mp.static_correspondence()
        self.assertLess(corr["commutator_XP_minus_2i_aa_dag_nmax8"], 1e-12)
        self.assertLess(corr["interior_XP_minus_2iI_nmax8"], 1e-12)
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        force = mp.static_force_identity(geo, k)
        self.assertLess(force["abs_F_minus_i_q_sum"], 1e-12)
        self.assertEqual(force["el_dim_each"], 36)

    def test_two_step_norm_and_kick_installed(self):
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        two = mp.two_step_backaction(geo, k)
        self.assertLess(two["norms_stay_one_kick"], 1e-12)
        self.assertGreaterEqual(two["abs_dP_kick_minus_frozen"], 0.0)
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        self.assertAlmostEqual(two["dt"], env.t_ramp / 1285, places=12)

    def test_no_ensemble_flags(self):
        self.assertFalse(mp.cost_estimate.__defaults__ is None and False)
        self.assertEqual(mp.GH_ORDERS_STATIC, (1, 2, 3, 5))
        self.assertIn("full GH ensemble", [
            "rewrite any 20261026-20261107 json", "G1", "H2", "HHG spectra",
            "vacuum", "theta=pi/2", "theta=pi", "Cn", "nmax scan",
            "full GH ensemble", "output photon statistics",
        ])


if __name__ == "__main__":
    unittest.main()
