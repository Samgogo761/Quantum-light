#!/usr/bin/env python3
"""Static gates for 20261109 QED vs MTEF compare. No ensemble."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_mtef_compare_20261109 as cmp
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import build_geometry

ROOT = Path(__file__).resolve().parents[2]


class MtefCompareStaticTests(unittest.TestCase):
    def test_frozen_params(self):
        self.assertEqual(cmp.N_STEPS, 1285)
        self.assertEqual(cmp.SAMPLE_EVERY, 32)
        self.assertEqual(cmp.N_STEPS_HALF, 2570)
        self.assertEqual(cmp.WALL_CAP_S, 180.0)
        self.assertEqual(pf.G_ABS, 0.005)
        self.assertEqual(pf.OMEGA0, 0.05)
        self.assertEqual(mp.ALPHA, 0.5 + 0j)
        self.assertEqual(mp.R_SQ, 0.3)
        self.assertEqual(mp.THETA, 0.0)
        self.assertEqual(cmp.JOBS, ds.JOBS)

    def test_does_not_overwrite_old_archives(self):
        dest = cmp.OUT_DIR / "compare.json"
        for p in cmp.OLD_ARCHIVES:
            self.assertTrue(p.is_file(), msg=str(p))
            self.assertNotEqual(p.resolve(), dest.resolve())
        self.assertTrue(cmp.ARCHIVE_QED.is_file())
        self.assertIn(ROOT / "results/q_line/qinput_a_shared_kk_mtef_map_20261108/map.json",
                      cmp.OLD_ARCHIVES)

    def test_sample_times_lock_to_20261102(self):
        qed = cmp.load_qed()
        self.assertEqual(qed["t"].size, 42)
        self.assertEqual(qed["n_steps"], 1285)
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        js = cmp.sample_js()
        t = cmp.times_of(cmp.N_STEPS, js, env.t_ramp)
        np.testing.assert_allclose(t, qed["t"], rtol=0, atol=1e-12)
        self.assertEqual(js[0], 0)
        self.assertEqual(js[-1], 1284)
        self.assertEqual(len(js), 42)

    def test_half_dt_indices_lock_same_times(self):
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        js = cmp.sample_js()
        js2 = cmp.sample_js_half()
        t = cmp.times_of(cmp.N_STEPS, js, env.t_ramp)
        t2 = cmp.times_of(cmp.N_STEPS_HALF, js2, env.t_ramp)
        np.testing.assert_allclose(t, t2, rtol=0, atol=1e-12)
        self.assertEqual(js2[0], 1)
        self.assertEqual(js2[-1], 2 * 1284 + 1)

    def test_gh1_nodes_shared_and_gh3_crn(self):
        c1 = mp.gauss_hermite_nodes("coherent", 1)
        d1 = mp.gauss_hermite_nodes("ds", 1)
        self.assertEqual(c1["n_nodes"], 1)
        self.assertAlmostEqual(c1["nodes"][0]["X"], d1["nodes"][0]["X"], places=14)
        self.assertAlmostEqual(c1["nodes"][0]["P"], d1["nodes"][0]["P"], places=14)
        c3 = [(n["xi"], n["eta"]) for n in mp.gauss_hermite_nodes("coherent", 3)["nodes"]]
        d3 = [(n["xi"], n["eta"]) for n in mp.gauss_hermite_nodes("ds", 3)["nodes"]]
        self.assertEqual(c3, d3)
        self.assertEqual(len(c3), 9)

    def test_even_odd_definition(self):
        jp = np.array([[1.0, 0.2], [1.2, 0.4]], complex)
        jm = np.array([[0.4, -0.2], [0.6, 0.0]], complex)
        even, odd = ds.even_odd(jp, jm)
        np.testing.assert_allclose(even, 0.5 * (jp + jm))
        np.testing.assert_allclose(odd, 0.5 * (jp - jm))
        d_odd = odd * 0.0 + (odd - odd)
        self.assertEqual(str(cmp.JOBS[0][0]), "coh_p")

    def test_qed_delta_definition_present(self):
        qed = cmp.load_qed()
        self.assertAlmostEqual(qed["max_abs"]["dJ_odd_x"], 1.2230424761838776e-07, delta=1e-18)
        np.testing.assert_allclose(qed["dJ_odd"], qed["J_odd_ds"] - qed["J_odd_coh"])

    def test_envelope_and_k(self):
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        self.assertAlmostEqual(env.s(0.0), 0.0)
        self.assertAlmostEqual(env.s(env.t_ramp), 1.0)
        self.assertEqual(tuple(sh.K_FRAC), (0.173, 0.271))
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        self.assertEqual(k.shape, (2,))
