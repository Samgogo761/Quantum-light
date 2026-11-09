#!/usr/bin/env python3
"""Static checks for the 20261106 DS nmax=10 truncation driver. No long run."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_coh_dt_20261105 as coh
import q8_m0_qinput_shared_kk_ds_dt_20261104 as dth
import q8_m0_qinput_shared_kk_ds_nmax10_20261106 as n10
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf

ROOT = Path(__file__).resolve().parents[2]


class Nmax10StaticTests(unittest.TestCase):
    def test_jobs_are_ds_both_neel_only(self):
        self.assertEqual(n10.JOBS, (("ds_p", 1), ("ds_m", -1)))
        self.assertEqual(n10.NMAX, 10)
        self.assertEqual(n10.N_STEPS, 2570)
        self.assertEqual(n10.N_STEPS, dth.N_STEPS)
        self.assertEqual(n10.WALL_CAP_S, 360.0)
        self.assertEqual(ds.THETA, 0.0)
        self.assertEqual(pf.ORDER, 16)

    def test_does_not_reopen_theta_cn_or_nmax12(self):
        self.assertNotIn("coh_p", dict(n10.JOBS))
        dest = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106/trunc10.json"
        self.assertNotEqual(n10.ARCHIVE_8.resolve(), dest.resolve())
        self.assertNotEqual(n10.DT_DS.resolve(), dest.resolve())
        self.assertNotEqual(n10.COH_DT.resolve(), dest.resolve())
        self.assertNotEqual(n10.TRUNC_12.resolve(), dest.resolve())
        self.assertEqual(n10.TARGET_ABS, 1e-8)

    def test_sample_offset_matches_dt_half(self):
        n_new, js = dth.sample_steps_half_dt(1285, 32)
        self.assertEqual(n_new, 2570)
        self.assertEqual(js[0], 1)
        self.assertEqual(js[1], 65)
        self.assertEqual(js[-1], 2569)

    def test_locked_dt_matches_archives(self):
        ds8 = coh.load_dt_ds(n10.DT_DS)
        coh8 = n10.load_coh_dt(n10.COH_DT)
        self.assertEqual(ds8["status"], "ok")
        self.assertEqual(coh8["status"], "ok")
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        dt_h = env.t_ramp / n10.N_STEPS
        self.assertAlmostEqual(dt_h, ds8["dt"], places=15)
        self.assertAlmostEqual(dt_h, coh8["dt"], places=15)
        n_new, js = dth.sample_steps_half_dt(1285, 32)
        t = dth.times_from_steps(n_new, env.t_ramp, js)
        np.testing.assert_allclose(t, ds8["t"], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(t, coh8["t"], atol=1e-12, rtol=0.0)

    def test_photon_static_nmax10_vs_8_12(self):
        ph = n10.photon_static_compare()
        t8, t10, t12 = ph["table"]
        self.assertEqual(len(ph["nmax10"]["unnormalized"]["p_n"]), 11)
        self.assertEqual(len(ph["nmax8"]["unnormalized"]["p_n"]), 9)
        self.assertEqual(len(ph["nmax12"]["unnormalized"]["p_n"]), 13)
        self.assertLess(t10["large80_P_n_gt_nmax"], t8["large80_P_n_gt_nmax"])
        self.assertGreater(t10["large80_P_n_gt_nmax"], t12["large80_P_n_gt_nmax"])
        self.assertLess(t10["p_boundary"], t8["p_boundary"])
        self.assertAlmostEqual(t8["mean_n"], 0.34269368516480214, places=12)
        self.assertAlmostEqual(t12["mean_n"], 0.34273207277125034, places=12)
        self.assertGreater(t10["mean_n"], t8["mean_n"])
        self.assertLess(abs(t10["mean_n"] - t12["mean_n"]), abs(t8["mean_n"] - t12["mean_n"]))

    def test_checkpoint_roundtrip(self):
        folder = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106"
        folder.mkdir(parents=True, exist_ok=True)
        psi = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(complex)
        samples = [{"t": 1.0, "norm": 1.0}]
        dth.save_ckpt(folder, "_unit_n10", psi, 7, samples)
        got = dth.load_ckpt(folder, "_unit_n10")
        self.assertIsNotNone(got)
        psi2, j2, sm2 = got
        np.testing.assert_allclose(psi2, psi)
        self.assertEqual(j2, 7)
        self.assertEqual(sm2[0]["t"], 1.0)
        for p in dth.ckpt_paths(folder, "_unit_n10"):
            p.unlink()

    def test_old_archives_protected(self):
        self.assertTrue(n10.ARCHIVE_8.is_file())
        self.assertTrue(n10.DT_DS.is_file())
        self.assertTrue(n10.COH_DT.is_file())
        self.assertIn(n10.ARCHIVE_8, n10.OLD_ARCHIVES)
        self.assertIn(n10.DT_DS, n10.OLD_ARCHIVES)
        self.assertIn(n10.COH_DT, n10.OLD_ARCHIVES)
        self.assertIn(n10.TRUNC_12, n10.OLD_ARCHIVES)


if __name__ == "__main__":
    unittest.main()
