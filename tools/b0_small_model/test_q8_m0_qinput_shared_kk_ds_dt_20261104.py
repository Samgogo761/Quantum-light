#!/usr/bin/env python3
"""Static checks for the 20261104 nmax=8 dt-halved DS J_odd driver. No long run."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_ds_dt_20261104 as dt
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf

ROOT = Path(__file__).resolve().parents[2]


class DtHalfStaticTests(unittest.TestCase):
    def test_jobs_are_ds_both_neel_only(self):
        self.assertEqual(dt.JOBS, (("ds_p", 1), ("ds_m", -1)))
        self.assertEqual(dt.NMAX, 8)
        self.assertEqual(dt.N_STEPS, 2570)
        self.assertEqual(dt.N_STEPS_OLD, 1285)
        self.assertEqual(dt.WALL_CAP_S, 360.0)
        self.assertEqual(ds.THETA, 0.0)

    def test_does_not_reopen_coherent_or_nmax12(self):
        self.assertNotIn("coh_p", dict(dt.JOBS))
        self.assertEqual(dt.TARGET_ABS, 1e-8)
        dest = ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104/dt.json"
        self.assertNotEqual(dt.ARCHIVE_8.resolve(), dest.resolve())
        self.assertNotEqual(dt.TRUNC_12.resolve(), dest.resolve())

    def test_sample_offset_is_not_naive_every_64(self):
        n_new, js = dt.sample_steps_half_dt(1285, 32)
        self.assertEqual(n_new, 2570)
        self.assertEqual(js[0], 1)
        self.assertEqual(js[1], 65)
        self.assertEqual(js[-1], 2569)
        self.assertNotEqual(js[0], 0)
        self.assertNotEqual(js[1], 64)

    def test_half_dt_times_match_archive(self):
        arch = dt.load_archive8(dt.ARCHIVE_8)
        self.assertEqual(arch["n_steps"], 1285)
        self.assertEqual(arch["sample_every"], 32)
        n_new, js = dt.sample_steps_half_dt(arch["n_steps"], arch["sample_every"])
        env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
        t = dt.times_from_steps(n_new, env.t_ramp, js)
        np.testing.assert_allclose(t, arch["t"], atol=1e-12, rtol=0.0)
        self.assertAlmostEqual(env.t_ramp / n_new, arch["dt"] / 2.0, places=15)

    def test_checkpoint_roundtrip(self):
        folder = ROOT / "results/q_line/qinput_a_shared_kk_ds_dt_20261104"
        folder.mkdir(parents=True, exist_ok=True)
        psi = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(complex)
        samples = [{"t": 1.0, "norm": 1.0}]
        dt.save_ckpt(folder, "_unit", psi, 7, samples)
        got = dt.load_ckpt(folder, "_unit")
        self.assertIsNotNone(got)
        psi2, j2, sm2 = got
        np.testing.assert_allclose(psi2, psi)
        self.assertEqual(j2, 7)
        self.assertEqual(sm2[0]["t"], 1.0)
        for p in dt.ckpt_paths(folder, "_unit"):
            p.unlink()

    def test_even_odd_reused(self):
        jp = np.array([[1.0, 0.2]], complex)
        jm = np.array([[0.4, -0.2]], complex)
        even, odd = ds.even_odd(jp, jm)
        np.testing.assert_allclose(even, [[0.7, 0.0]])
        np.testing.assert_allclose(odd, [[0.3, 0.2]])

    def test_old_archives_protected(self):
        self.assertTrue(dt.ARCHIVE_8.is_file())
        self.assertIn(dt.ARCHIVE_8, dt.OLD_ARCHIVES)
        self.assertIn(dt.TRUNC_12, dt.OLD_ARCHIVES)


if __name__ == "__main__":
    unittest.main()
