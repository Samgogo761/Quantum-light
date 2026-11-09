#!/usr/bin/env python3
"""Static gates for 20261110 GH5 quadrature check. No ensemble."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_compare_20261109 as cmp09
import q8_m0_qinput_shared_kk_mtef_gh5_20261110 as gh5
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import build_geometry

ROOT = Path(__file__).resolve().parents[2]


class MtefGh5StaticTests(unittest.TestCase):
    def test_frozen_params(self):
        self.assertEqual(gh5.WALL_CAP_S, 240.0)
        self.assertEqual(gh5.ORDER, 5)
        self.assertEqual(gh5.N_NODES, 25)
        self.assertEqual(gh5.N_TRAJ, 100)
        self.assertEqual(cmp09.N_STEPS, 1285)
        self.assertEqual(pf.G_ABS, 0.005)
        self.assertEqual(pf.OMEGA0, 0.05)
        self.assertEqual(mp.ALPHA, 0.5 + 0j)
        self.assertEqual(mp.R_SQ, 0.3)
        self.assertEqual(mp.THETA, 0.0)
        self.assertEqual(gh5.JOBS, ds.JOBS)

    def test_does_not_overwrite_old_archives(self):
        dest = gh5.OUT_DIR / "gh5.json"
        for p in gh5.OLD_ARCHIVES:
            self.assertTrue(p.is_file(), msg=str(p))
            self.assertNotEqual(p.resolve(), dest.resolve())
        self.assertTrue(gh5.ARCHIVE_GH3.is_file())
        self.assertIn(gh5.ARCHIVE_GH3, gh5.OLD_ARCHIVES)
        self.assertNotEqual(
            (ROOT / "results/q_line/qinput_a_shared_kk_mtef_compare_20261109/compare.json").resolve(),
            dest.resolve())

    def test_gh5_nodes_shared_crn_and_moments(self):
        c = mp.gauss_hermite_nodes("coherent", 5)
        d = mp.gauss_hermite_nodes("ds", 5)
        self.assertEqual(c["n_nodes"], 25)
        self.assertEqual(d["n_nodes"], 25)
        c_xy = [(n["xi"], n["eta"]) for n in c["nodes"]]
        d_xy = [(n["xi"], n["eta"]) for n in d["nodes"]]
        self.assertEqual(c_xy, d_xy)
        ana_c = mp.analytic_xp("coherent")
        ana_d = mp.analytic_xp("ds")
        self.assertAlmostEqual(c["quad_mean_X"], ana_c["mean_X"], places=10)
        self.assertAlmostEqual(c["quad_var_X"], ana_c["var_X"], places=10)
        self.assertAlmostEqual(d["quad_var_X"], ana_d["var_X"], places=10)
        self.assertAlmostEqual(d["quad_var_P"], ana_d["var_P"], places=10)
        self.assertAlmostEqual(c["weight_sum"], 1.0, places=10)
        self.assertAlmostEqual(c["quad_mean_X"], d["quad_mean_X"], places=12)

    def test_times_lock_to_qed_and_gh3(self):
        qed = cmp09.load_qed()
        g3 = gh5.load_gh3()
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        js = cmp09.sample_js()
        t = cmp09.times_of(cmp09.N_STEPS, js, env.t_ramp)
        np.testing.assert_allclose(t, qed["t"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(t, g3["t"], rtol=0, atol=1e-12)
        self.assertEqual(g3["order"], 3)
        np.testing.assert_allclose(g3["d_odd"], g3["odd_d"] - g3["odd_c"])

    def test_even_odd_and_qed_delta(self):
        jp = np.array([[1.0, 0.2], [1.2, 0.4]], complex)
        jm = np.array([[0.4, -0.2], [0.6, 0.0]], complex)
        even, odd = ds.even_odd(jp, jm)
        np.testing.assert_allclose(even, 0.5 * (jp + jm))
        np.testing.assert_allclose(odd, 0.5 * (jp - jm))
        qed = cmp09.load_qed()
        np.testing.assert_allclose(qed["dJ_odd"], qed["J_odd_ds"] - qed["J_odd_coh"])

    def test_cancellation_identity(self):
        rng = np.random.default_rng(10)
        oc = rng.normal(size=(8, 2)) + 1j * 0
        od = rng.normal(size=(8, 2)) + 1j * 0
        qc = rng.normal(size=(8, 2)) + 1j * 0
        qd = rng.normal(size=(8, 2)) + 1j * 0
        s = gh5.cancellation_split(oc, od, qc, qd)
        delta_err = (od - oc) - (qd - qc)
        self.assertAlmostEqual(s["differential_ds_minus_coh"]["x"],
                               float(np.max(np.abs(delta_err[:, 0]))), delta=1e-15)

    def test_force_kick_and_k(self):
        self.assertEqual(tuple(sh.K_FRAC), (0.173, 0.271))
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        self.assertEqual(k.shape, (2,))
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        self.assertAlmostEqual(env.s(0.0), 0.0)
        self.assertAlmostEqual(env.s(env.t_ramp), 1.0)

    def test_per_job_paths_are_new(self):
        old = ROOT / "results/q_line/qinput_a_shared_kk_mtef_compare_20261109/compare.json"
        for tag, _, _ in gh5.JOBS:
            self.assertNotEqual(gh5.job_path(tag).resolve(), old.resolve())
            self.assertNotEqual(gh5.ckpt_path(tag).resolve(), old.resolve())
        self.assertTrue(str(gh5.OUT_DIR).endswith("qinput_a_shared_kk_mtef_gh5_20261110"))


if __name__ == "__main__":
    unittest.main()
