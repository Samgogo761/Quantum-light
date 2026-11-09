#!/usr/bin/env python3
"""Static gates for 20261111 archive-side error decomposition. No propagation."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_shared_kk_errdecomp_20261111 as ed
import q8_m0_qinput_shared_kk_mtef_compare_20261109 as cmp
import q8_m0_qinput_shared_kk_mtef_gh5_20261110 as gh5
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp

ROOT = Path(__file__).resolve().parents[2]


class ErrdecompStaticTests(unittest.TestCase):
    def test_frozen_and_new_dest(self):
        self.assertEqual(ed.RECORD_DATE, "2026-11-11")
        self.assertEqual(ed.MODEL_ID, "M0_shared_mode_k_minus_k_errdecomp_v1")
        dest = ed.OUT_DIR / "errdecomp.json"
        self.assertTrue(str(ed.OUT_DIR).endswith("qinput_a_shared_kk_errdecomp_20261111"))
        for p in ed.OLD_ARCHIVES:
            self.assertTrue(p.is_file(), msg=str(p))
            self.assertNotEqual(p.resolve(), dest.resolve())
        self.assertNotEqual(gh5.OUT_DIR.resolve(), ed.OUT_DIR.resolve())
        self.assertTrue(ed.ARCHIVE_GH5.is_file())
        self.assertIn(ed.ARCHIVE_GH5, ed.OLD_ARCHIVES)

    def test_channel_identities(self):
        rng = np.random.default_rng(3)
        mc = rng.normal(size=(8, 2)) + 1j * 0
        md = rng.normal(size=(8, 2)) + 1j * 0
        qc = rng.normal(size=(8, 2)) + 1j * 0
        qd = rng.normal(size=(8, 2)) + 1j * 0
        ch = ed.channels(mc, md, qc, qd)
        self.assertTrue(ed.channel_identities_ok(ch))
        np.testing.assert_allclose(ch["e_delta"], (md - mc) - (qd - qc))
        np.testing.assert_allclose(ch["e_common"], 0.5 * (ch["e_c"] + ch["e_ds"]))

    def test_times_lock_without_interpolation(self):
        qed = cmp.load_qed()
        g5 = ed.load_gh5_series()
        g3 = ed.load_gh3_series()
        np.testing.assert_allclose(qed["t"], g5["t"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(qed["t"], g3["t"], rtol=0, atol=1e-12)
        self.assertEqual(len(qed["t"]), 42)
        np.testing.assert_allclose(qed["dJ_odd"], qed["J_odd_ds"] - qed["J_odd_coh"])
        np.testing.assert_allclose(g5["dJ_odd"], g5["J_odd_ds"] - g5["J_odd_coh"])

    def test_product_state_purities_are_one(self):
        ok, p = ed.product_state_purity_gate()
        self.assertTrue(ok)
        self.assertAlmostEqual(p["purity_photon"], 1.0, places=12)
        self.assertAlmostEqual(p["purity_k"], 1.0, places=12)
        self.assertAlmostEqual(p["purity_el_joint"], 1.0, places=12)

    def test_checkpoint_inventory_paths(self):
        qed_dir = ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102"
        npys = list(qed_dir.glob("*.npy")) + list(qed_dir.glob("*.npz"))
        self.assertEqual(npys, [])
        for key, folder, tag, n_max, neel, n_exp, note in ed.CKPT_SPEC:
            npz, js = (folder / f"ckpt_{tag}.npz", folder / f"ckpt_{tag}_samples.json")
            self.assertTrue(npz.is_file(), msg=f"{key} {npz}")
            self.assertTrue(js.is_file(), msg=f"{key} {js}")
            self.assertEqual(n_exp, 2570)
            self.assertIn(n_max, (8, 10))

    def test_summarize_common_mode_ratio(self):
        t = np.linspace(0, 1, 5)
        qed_c = np.stack([np.sin(t), 0.4 * np.sin(t)], axis=1).astype(complex)
        qed_d = 0.8 * qed_c
        mtef_c = 0.38 * qed_c
        mtef_d = 0.38 * qed_d
        ch = ed.channels(mtef_c, mtef_d, qed_c, qed_d)
        s = ed.summarize(ch, qed_c, qed_d, qed_d - qed_c, mtef_d - mtef_c)
        self.assertGreater(s["corr"]["eC_eDS_x"], 0.999)
        self.assertLess(s["corr"]["eC_qedC_x"], -0.999)
        self.assertAlmostEqual(s["scale"]["max_eCommon_over_max_eC_x"], 0.9, places=14)
        self.assertLess(s["scale"]["max_eDelta_over_max_eC_x"], 0.3)

    def test_entangled_reduced_state_against_explicit_partial_trace(self):
        # Nontrivial phases catch transpose/conjugation mistakes hidden by products.
        rng = np.random.default_rng(19)
        V = rng.normal(size=(2, 2, 3)) + 1j*rng.normal(size=(2, 2, 3))
        V /= np.linalg.norm(V)
        full = np.outer(V.ravel(), V.ravel().conj()).reshape(2, 2, 3, 2, 2, 3)
        rp = np.einsum('abmabn->mn', full)
        re = np.einsum('abmcdm->abcd', full).reshape(4, 4)
        p = ed.reduced_purities(V.ravel(), el=2)
        self.assertAlmostEqual(p['purity_photon'], np.trace(rp@rp).real, places=13)
        self.assertAlmostEqual(p['purity_el_joint'], np.trace(re@re).real, places=13)
        bell = np.zeros((2, 2, 2), complex)
        bell[0, 0, 0] = 1/np.sqrt(2)
        bell[1, 0, 1] = 1j/np.sqrt(2)
        self.assertAlmostEqual(ed.reduced_purities(bell, el=2)['purity_photon'], 0.5)

    def test_factorized_current_product_state(self):
        model = ed.build_models({(2, 1)})[(2, 1)]
        rng = np.random.default_rng(33)
        parts = [rng.normal(size=d)+1j*rng.normal(size=d) for d in (ed.EL, ed.EL, 3)]
        parts = [v/np.linalg.norm(v) for v in parts]
        psi = np.kron(np.kron(parts[0], parts[1]), parts[2])
        j = ed.sh.apply_current(model, psi, 0.7)
        direct = np.array([np.vdot(psi, a) for a in j])
        np.testing.assert_allclose(ed.factorized_current(model, psi, 0.7), direct, atol=1e-13, rtol=0)

    def test_peak_ratio_does_not_certify_waveform_match(self):
        t = np.linspace(0, 2*np.pi, 101, endpoint=False)
        q, m = np.sin(t), 0.4*np.cos(t)
        fit = ed.fit_scale(q, m)
        self.assertAlmostEqual(fit['scale_through_zero'], 0, places=13)
        self.assertAlmostEqual(fit['relative_rms_after_scale_fit'], 0.4, places=13)

    def test_factorized_current_entangled_state_against_schmidt_mixture(self):
        model = ed.build_models({(2, 1)})[(2, 1)]
        rng = np.random.default_rng(71)
        M = rng.normal(size=(ed.EL**2, 3)) + 1j*rng.normal(size=(ed.EL**2, 3))
        M /= np.linalg.norm(M)
        left, sv, right = np.linalg.svd(M, full_matrices=False)
        reference = np.zeros(2, complex)
        for i, wi in enumerate(sv**2):
            for j, wj in enumerate(sv**2):
                prod = np.kron(left[:, i], right[j, :])
                reference += wi*wj*np.array([np.vdot(prod, v) for v in ed.sh.apply_current(model, prod, 0.7)])
        np.testing.assert_allclose(ed.factorized_current(model, M.ravel(), 0.7), reference, atol=1e-13, rtol=0)

    def test_sha_helper_and_jsonable(self):
        h = mp.sha256_file(ed.ARCHIVE_GH5)
        self.assertEqual(len(h), 64)
        packed = mp.jsonable({"z": 1 + 0j, "a": np.float64(1.5)})
        self.assertEqual(packed["z"]["re"], 1.0)
        self.assertEqual(packed["a"], 1.5)


if __name__ == "__main__":
    unittest.main()
