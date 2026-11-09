#!/usr/bin/env python3
"""Static checks for the 20261105 nmax=8 dt-halved coherent J_odd / ΔJ_odd driver."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_coh_dt_20261105 as coh
import q8_m0_qinput_shared_kk_ds_dt_20261104 as dt
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf

ROOT = Path(__file__).resolve().parents[2]


class CohDtHalfStaticTests(unittest.TestCase):
    def test_jobs_are_coherent_both_neel_only(self):
        self.assertEqual(coh.JOBS, (("coh_p", 1), ("coh_m", -1)))
        self.assertEqual(coh.NMAX, 8)
        self.assertEqual(coh.N_STEPS, 2570)
        self.assertEqual(coh.N_STEPS_OLD, 1285)
        self.assertEqual(coh.WALL_CAP_S, 360.0)
        self.assertEqual(float(ds.ALPHA.real), 0.5)
        self.assertEqual(float(ds.ALPHA.imag), 0.0)

    def test_does_not_reopen_ds_or_nmax12(self):
        self.assertNotIn("ds_p", dict(coh.JOBS))
        dest = ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105/coh_dt.json"
        self.assertNotEqual(coh.ARCHIVE_8.resolve(), dest.resolve())
        self.assertNotEqual(coh.DT_DS.resolve(), dest.resolve())
        self.assertNotEqual(coh.TRUNC_12.resolve(), dest.resolve())
        self.assertEqual(coh.N_STEPS, dt.N_STEPS)

    def test_sample_offset_matches_ds_dt(self):
        n_new, js = dt.sample_steps_half_dt(1285, 32)
        self.assertEqual(n_new, 2570)
        self.assertEqual(js[0], 1)
        self.assertEqual(js[1], 65)
        self.assertEqual(js[-1], 2569)

    def test_half_dt_times_match_both_archives(self):
        arch = coh.load_compare8(coh.ARCHIVE_8)
        ds_half = coh.load_dt_ds(coh.DT_DS)
        self.assertEqual(ds_half["status"], "ok")
        n_new, js = dt.sample_steps_half_dt(arch["n_steps"], arch["sample_every"])
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        t = dt.times_from_steps(n_new, env.t_ramp, js)
        np.testing.assert_allclose(t, arch["t"], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(t, ds_half["t"], atol=1e-12, rtol=0.0)
        self.assertAlmostEqual(env.t_ramp / n_new, arch["dt"] / 2.0, places=15)

    def test_delta_is_ds_minus_coh(self):
        arch = coh.load_compare8(coh.ARCHIVE_8)
        np.testing.assert_allclose(arch["dJ_odd"], arch["J_odd_ds"] - arch["J_odd_coh"],
                                   atol=1e-18, rtol=0.0)
        self.assertAlmostEqual(float(np.max(np.abs(arch["dJ_odd"][:, 0]))),
                               arch["max_abs"]["dJ_odd_x"], places=18)

    def test_photon_raw_is_coherent_not_ds(self):
        c = ds.photon_raw("coherent", 8)
        d = ds.photon_raw("ds", 8, 0.0)
        self.assertGreater(float(np.linalg.norm(c - d)), 1e-6)
        self.assertEqual(len(c), 9)

    def test_checkpoint_roundtrip(self):
        folder = ROOT / "results/q_line/qinput_a_shared_kk_coh_dt_20261105"
        folder.mkdir(parents=True, exist_ok=True)
        psi = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(complex)
        samples = [{"t": 1.0, "norm": 1.0}]
        dt.save_ckpt(folder, "_unit_coh", psi, 7, samples)
        got = dt.load_ckpt(folder, "_unit_coh")
        self.assertIsNotNone(got)
        psi2, j2, sm2 = got
        np.testing.assert_allclose(psi2, psi)
        self.assertEqual(j2, 7)
        self.assertEqual(sm2[0]["t"], 1.0)
        for p in dt.ckpt_paths(folder, "_unit_coh"):
            p.unlink()

    def test_old_archives_protected(self):
        self.assertTrue(coh.ARCHIVE_8.is_file())
        self.assertTrue(coh.DT_DS.is_file())
        self.assertIn(coh.ARCHIVE_8, coh.OLD_ARCHIVES)
        self.assertIn(coh.DT_DS, coh.OLD_ARCHIVES)
        self.assertIn(coh.TRUNC_12, coh.OLD_ARCHIVES)


if __name__ == "__main__":
    unittest.main()
