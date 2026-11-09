"""Batch-1 tests: endpoint Fourier, BAC closed form, and runner gates.

No 200-step smoke and no real QED/GH5 propagation.
"""
from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qinput_spectrum_reanalysis_20261028 import spectrum
import q8_m0_qinput_1600nm_protocolA_qed_postprocess_20261102 as pp
import q8_m0_qinput_1600nm_protocolA_qed_fourangle_20261102 as runner
import q8_m0_qinput_1600nm_protocolA_qed_fockaudit_20261102 as fa


class Fourier775Tests(unittest.TestCase):
    def test_constant_false_h2_vanishes_on_production_grid(self):
        t, _ = pp.load_series(pp.QED_OUT, "p_0")
        rec = pp.constant_false_h2_on_grid(t)
        self.assertTrue(rec["legacy_leaks"])
        self.assertGreater(rec["legacy_abs_h2"], 1e-3)
        self.assertTrue(rec["linear_exact_ok"])
        self.assertLess(rec["linear_exact_abs_max"], 1e-12)

    def test_known_sign_on_775_grid(self):
        t, _ = pp.load_series(pp.QED_OUT, "p_0")
        omega = pp.OMEGA0
        period = pp.PERIOD
        phase = 0.73
        y = 2 * np.exp(1j * phase) * np.exp(-2j * omega * t)
        got = spectrum(t, y, 2 * period, 4 * period, [2], method="linear_exact", omega=omega)[0, 0]
        want = 2 * np.exp(1j * phase)
        # 350-point piecewise-linear reconstruction is not machine-precise.
        # Dense-grid tests already use 2e-6; this production grid uses 2e-2.
        self.assertLess(abs(got - want), 2e-2)
        self.assertLess(abs(np.angle(got * np.conj(want))), 0.05)


class BACTests(unittest.TestCase):
    def test_closed_form_matches_lstsq(self):
        B, A, C = 1.2 + 0.3j, 0.4 - 0.25j, 0.05 + 0.02j
        zs = [B + A * np.exp(1j * th) + C * np.exp(-1j * th) for th in pp.THETA_ORDER]
        fit = pp.fit_from_z(zs, 1e-16)
        self.assertTrue(np.allclose(fit["coeff"], [B, A, C]))
        pred = B + A * np.exp(1j * pp.HOLD_THETA) + C * np.exp(-1j * pp.HOLD_THETA)
        self.assertLess(abs(fit["holdout_prediction"] - pred), 1e-12)

    def test_training_rejects_pi4(self):
        pp.assert_training_not_holdout()
        with self.assertRaisesRegex(RuntimeError, "pi/4"):
            pp.assert_training_not_holdout(theta_order=(0.0, 0.25 * np.pi, np.pi, 1.5 * np.pi))
        with self.assertRaisesRegex(RuntimeError, "holdout"):
            pp.assert_training_not_holdout(training_tags=("p_0", "p_q"))


def _ok_job(neel, theta, n_full=11154, top_n_steps=None, alarm=False, status="ok"):
    rec = {
        "status": status,
        "fock_alarm_0p05": alarm,
        "omega0": 0.775,
        "n_max": 8,
        "used_pf_OMEGA0": False,
        "did_not_scale_g": True,
        "finite_state": True,
        "covers_requested": True,
        "error": None,
        "neel": neel,
        "theta": theta,
        "definition": {"n_full": n_full, "omega0": 0.775, "g": 0.005, "n_max": 8},
        "ledger": {"steps_completed": n_full},
    }
    if top_n_steps is not None:
        rec["n_steps"] = top_n_steps
    return rec


class RunnerGateTests(unittest.TestCase):
    def setUp(self):
        self.out = HERE / "_tmp_gate_tests"
        if self.out.exists():
            shutil.rmtree(self.out, ignore_errors=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.old = {
            "OUT": runner.OUT,
            "ANALYSIS": runner.ANALYSIS,
            "FIT_PATH": runner.FIT_PATH,
            "FIT_SHA_PATH": runner.FIT_SHA_PATH,
            "STAGE2_BUDGET_PATH": runner.STAGE2_BUDGET_PATH,
        }
        runner.OUT = self.out
        runner.ANALYSIS = self.out / "analysis_endpoint_v1"
        runner.FIT_PATH = self.out / "fourangle_fit.json"
        runner.FIT_SHA_PATH = self.out / "fourangle_fit.sha256"
        runner.STAGE2_BUDGET_PATH = self.out / "stage2_budget.json"
        runner.bind_fockaudit_namespace()

    def tearDown(self):
        runner.OUT = self.old["OUT"]
        runner.ANALYSIS = self.old["ANALYSIS"]
        runner.FIT_PATH = self.old["FIT_PATH"]
        runner.FIT_SHA_PATH = self.old["FIT_SHA_PATH"]
        runner.STAGE2_BUDGET_PATH = self.old["STAGE2_BUDGET_PATH"]
        runner.bind_fockaudit_namespace()
        shutil.rmtree(self.out, ignore_errors=True)

    def _write_job(self, tag, rec, series=True):
        (self.out / f"job_{tag}.json").write_text(json.dumps(rec), encoding="utf-8")
        if series:
            (self.out / f"series_{tag}.npz").write_bytes(b"npz")

    def test_official_n_full_missing_top_level(self):
        rec = _ok_job(1, 0.0)
        self.assertEqual(runner.official_n_full(rec), 11154)
        rec["n_steps"] = 11154
        self.assertEqual(runner.official_n_full(rec), 11154)
        rec["n_steps"] = 200
        self.assertIsNone(runner.official_n_full(rec))

    def test_n_steps_flag_refused(self):
        rc = runner.main(["--analyze-only", "--n-steps", "200"])
        self.assertEqual(rc, 3)
        why = json.loads((self.out / "refused_not_authorized.json").read_text(encoding="utf-8"))
        self.assertIn("n-steps", why["why"])

    def test_completed_axis_does_not_call_fockaudit(self):
        self._write_job("p_0", _ok_job(1, 0.0))
        with mock.patch.object(fa, "run_one_job", side_effect=AssertionError("must not propagate")):
            rec = runner.run_one_job("p_0")
        self.assertEqual(rec["status"], "ok")
        self.assertTrue(runner.job_ok_no_alarm("p_0"))

    def test_launch_serial_stops_after_first_failure(self):
        self._write_job("p_0", _ok_job(1, 0.0, status="running"), series=False)
        self._write_job("m_0", _ok_job(-1, 0.0, status="running"), series=False)
        calls = []

        def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None):
            calls.append(cmd)
            tag = cmd[-1]
            self._write_job(tag, _ok_job(1, 0.0, status="failed"), series=False)
            return mock.Mock(returncode=1)

        with mock.patch("subprocess.run", side_effect=fake_run):
            out = runner.launch_serial(("p_0", "m_0"), "stage1")
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["stopped_after"], "p_0")
        self.assertNotIn("m_0", out["returncodes"])

    def test_pi4_without_freeze_refused(self):
        for tag, neel, th in (("p_0", 1, 0.0), ("m_0", -1, 0.0), ("p_h", 1, 0.5 * np.pi),
                              ("m_h", -1, 0.5 * np.pi), ("p_pi", 1, np.pi), ("m_pi", -1, np.pi),
                              ("p_t", 1, 1.5 * np.pi), ("m_t", -1, 1.5 * np.pi)):
            self._write_job(tag, _ok_job(neel, th))
        os.environ[runner.AUTH_ENV] = "stage2"
        try:
            rec = runner.run_one_job("p_q")
        finally:
            os.environ.pop(runner.AUTH_ENV, None)
        self.assertEqual(rec["status"], "refused_not_authorized")
        self.assertIn("fit", rec["why"])

    def test_freeze_fit_refused_without_eight(self):
        self._write_job("p_0", _ok_job(1, 0.0))
        rec = runner.freeze_fit()
        self.assertEqual(rec["status"], "refused_not_authorized")


class ProductionReadOnlyTests(unittest.TestCase):
    def test_real_p0_is_ok_without_top_level_n_steps(self):
        runner.bind_fockaudit_namespace()
        rec = runner.load_job("p_0")
        self.assertIsNone(rec.get("n_steps"))
        self.assertEqual(runner.official_n_full(rec), 11154)
        self.assertTrue(runner.job_ok_no_alarm("p_0"))
        with mock.patch.object(fa, "run_one_job", side_effect=AssertionError("no")):
            got = runner.run_one_job("p_0")
        self.assertEqual(got["status"], "ok")


if __name__ == "__main__":
    unittest.main()
