#!/usr/bin/env python3
"""Static checks for 20261107 nmax=10 checkpoint resume. No long run."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_coh_dt_20261105 as coh
import q8_m0_qinput_shared_kk_ds_dt_20261104 as dth
import q8_m0_qinput_shared_kk_ds_nmax10_20261106 as n10
import q8_m0_qinput_shared_kk_ds_nmax10_resume_20261107 as rs
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106"


class ResumeStaticTests(unittest.TestCase):
    def test_scope_and_caps(self):
        self.assertEqual(rs.JOBS, (("ds_p", 1), ("ds_m", -1)))
        self.assertEqual(rs.WALL_CAP_S, 240.0)
        self.assertEqual(n10.NMAX, 10)
        self.assertEqual(n10.N_STEPS, 2570)
        self.assertEqual(pf.ORDER, 16)
        self.assertEqual(ds.THETA, 0.0)
        self.assertEqual(rs.TARGET_ABS, 1e-8)
        self.assertEqual(rs.CLEARLY_OVER, 1e-7)
        self.assertEqual(rs.DIM, 14256)

    def test_does_not_overwrite_trunc10_or_old_archives(self):
        self.assertTrue(rs.TRUNC10.is_file())
        self.assertIn(rs.TRUNC10, rs.OLD_ARCHIVES)
        self.assertIn(n10.DT_DS, rs.OLD_ARCHIVES)
        self.assertIn(n10.COH_DT, rs.OLD_ARCHIVES)
        dest = CKPT / "trunc10_resume.json"
        self.assertNotEqual(rs.TRUNC10.resolve(), dest.resolve())
        self.assertNotEqual(n10.DT_DS.resolve(), dest.resolve())
        self.assertNotEqual(n10.COH_DT.resolve(), dest.resolve())

    def test_prefix_count_at_timeout_next_j(self):
        n_new, js = dth.sample_steps_half_dt(1285, 32)
        self.assertEqual(n_new, 2570)
        self.assertEqual(sum(1 for j in js if j < 2080), 33)
        self.assertEqual(sum(1 for j in js if j < 2087), 33)
        self.assertEqual(js[32], 2049)
        self.assertEqual(js[33], 2113)
        self.assertEqual(len(js), 42)

    def test_ckpt_integrity_no_propagation(self):
        n_new, js = dth.sample_steps_half_dt(1285, 32)
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        t_all = dth.times_from_steps(n_new, env.t_ramp, js)
        rec_p = rs.inspect_one(CKPT, "ds_p", 2080, js, t_all)
        rec_m = rs.inspect_one(CKPT, "ds_m", 2087, js, t_all)
        self.assertTrue(rec_p["ok"], rec_p)
        self.assertTrue(rec_m["ok"], rec_m)
        self.assertEqual(rec_p["psi_shape"], [14256])
        self.assertEqual(rec_m["psi_shape"], [14256])
        self.assertEqual(rec_p["n_samples"], 33)
        self.assertEqual(rec_m["n_samples"], 33)
        self.assertEqual(rec_p["remaining_steps"], 490)
        self.assertEqual(rec_m["remaining_steps"], 483)
        self.assertLess(rec_p["abs_norm_minus_1"], 1e-12)
        self.assertLess(rec_m["abs_norm_minus_1"], 1e-12)

    def test_prefix_compare_zero_on_identical(self):
        ds8 = coh.load_dt_ds(n10.DT_DS)
        n = 33
        fake_p = [{"t": float(ds8["t"][i]),
                   "mean_J": [{"re": float(ds8["J_odd_ds_half"][i, 0].real), "im": 0.0},
                              {"re": float(ds8["J_odd_ds_half"][i, 1].real), "im": 0.0}]}
                  for i in range(n)]
        fake_m = [{"t": float(ds8["t"][i]),
                   "mean_J": [{"re": -float(ds8["J_odd_ds_half"][i, 0].real), "im": 0.0},
                              {"re": -float(ds8["J_odd_ds_half"][i, 1].real), "im": 0.0}]}
                  for i in range(n)]
        cmp = rs.prefix_compare(ds8, fake_p, fake_m)
        self.assertEqual(cmp["n_prefix"], 33)
        self.assertLess(cmp["max_abs_dJ_odd_ds_10_minus_8_x"], 1e-15)
        self.assertTrue(cmp["resume_allowed"])
        self.assertFalse(cmp["clearly_over_1e-7"])

    def test_clearly_over_blocks_resume(self):
        ds8 = coh.load_dt_ds(n10.DT_DS)
        n = 5
        fake_p = [{"t": float(ds8["t"][i]),
                   "mean_J": [{"re": 1e-3, "im": 0.0}, {"re": 0.0, "im": 0.0}]}
                  for i in range(n)]
        fake_m = [{"t": float(ds8["t"][i]),
                   "mean_J": [{"re": -1e-3, "im": 0.0}, {"re": 0.0, "im": 0.0}]}
                  for i in range(n)]
        ds8s = {k: (v[:n] if k in ("t", "J_odd_ds_half", "J_even_ds_half") else v) for k, v in ds8.items()}
        cmp = rs.prefix_compare(ds8s, fake_p, fake_m)
        self.assertGreater(cmp["max_abs_dJ_odd_ds_10_minus_8_x"], rs.CLEARLY_OVER)
        self.assertTrue(cmp["clearly_over_1e-7"])
        self.assertFalse(cmp["resume_allowed"])

    def test_no_nmax12_or_coherent_jobs(self):
        self.assertNotIn("coh_p", dict(rs.JOBS))
        self.assertNotIn("coh_m", dict(rs.JOBS))
        self.assertEqual(rs.WALL_CAP_S, 240.0)
        self.assertTrue(rs.n10.NMAX == 10)


if __name__ == "__main__":
    unittest.main()
