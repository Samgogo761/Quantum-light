#!/usr/bin/env python3
"""Unit tests for strict shared-node +N/-N post-processing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "analysis" / "combine_pmN_nodes_strict.py"
SPEC = importlib.util.spec_from_file_location("combine_pmN_nodes_strict", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _table(domain: str) -> np.ndarray:
    """Two nodes and two harmonics with an analytically simple decomposition."""
    if domain not in {"plus", "minus"}:
        raise ValueError(domain)
    rows = []
    for node_id, weight, intensity, phase in (
        (1, 0.25, 1.0e11, 0.0),
        (2, 0.75, 2.0e11, 1.0),
    ):
        for order in (2, 5):
            if domain == "plus":
                jx = 3.0 if node_id == 1 else 5.0
            else:
                jx = 1.0
            rows.append(
                [node_id, weight, intensity, phase, order, jx, 0.0, 0.0, 0.0, jx * jx]
            )
    return np.asarray(rows, dtype=float)


class StrictPairingTests(unittest.TestCase):
    def _write(self, path: Path, data: np.ndarray) -> None:
        np.savetxt(path, data, fmt="%.17e")

    def test_weighted_summary_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plus_path = root / "plus.dat"
            minus_path = root / "minus.dat"
            out_path = root / "node_je_jo.dat"
            summary_path = root / "summary.dat"
            self._write(plus_path, _table("plus"))
            self._write(minus_path, _table("minus")[::-1])  # row reordering is harmless

            plus = MODULE.load_modes(plus_path)
            minus = MODULE.load_modes(minus_path)
            aligned = MODULE.align_and_validate_pair(
                plus, minus, plus_path, minus_path, rtol=1.0e-12, atol=1.0e-12
            )
            jxp, jyp = MODULE._complex_components(plus)
            jxm, jym = MODULE._complex_components(aligned)
            jxe, jye = 0.5 * (jxp + jxm), 0.5 * (jyp + jym)
            jxo, jyo = 0.5 * (jxp - jxm), 0.5 * (jyp - jym)
            MODULE.write_node_output(out_path, plus, jxe, jye, jxo, jyo)
            MODULE.write_summary(
                summary_path, plus, jxp, jyp, jxm, jym, jxe, jye, jxo, jyo
            )

            node_rows = np.loadtxt(out_path, comments="#", ndmin=2)
            summary_rows = np.loadtxt(summary_path, comments="#", ndmin=2)
            self.assertEqual(node_rows.shape, (4, 15))
            self.assertEqual(summary_rows.shape[0], 2)

            # Summary layout: 3 leading columns, 16 complex-component columns,
            # then 15 scalar diagnostics.
            row_h2 = summary_rows[summary_rows[:, 0] == 2][0]
            self.assertAlmostEqual(row_h2[3], 4.5)       # Re mu_plus_x
            self.assertAlmostEqual(row_h2[7], 1.0)       # Re mu_minus_x
            self.assertAlmostEqual(row_h2[11], 2.75)     # Re Je_bar_x
            self.assertAlmostEqual(row_h2[15], 1.75)     # Re Jo_bar_x
            scalar = row_h2[19:]
            self.assertAlmostEqual(scalar[0], 21.0)      # +N ICS
            self.assertAlmostEqual(scalar[1], 20.25)     # +N CS
            self.assertAlmostEqual(scalar[3], 1.0)       # -N ICS
            self.assertAlmostEqual(scalar[6], 7.75)      # trajectory-even ICS
            self.assertAlmostEqual(scalar[9], 3.25)      # trajectory-odd ICS
            self.assertAlmostEqual(scalar[12], 11.0)     # mean domain ICS
            self.assertAlmostEqual(scalar[13], 11.0)     # even+odd ICS
            self.assertLess(scalar[14], 1.0e-14)         # identity residual

    def test_duplicate_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.dat"
            data = _table("plus")
            data = np.vstack((data, data[0]))
            self._write(path, data)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.load_modes(path)

    def test_negative_weight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.dat"
            data = _table("plus")
            data[data[:, 0] == 1, 1] = -0.25
            self._write(path, data)
            with self.assertRaisesRegex(ValueError, "negative quadrature weight"):
                MODULE.load_modes(path)

    def test_plus_minus_metadata_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plus_path = root / "plus.dat"
            minus_path = root / "minus.dat"
            plus_data = _table("plus")
            minus_data = _table("minus")
            minus_data[(minus_data[:, 0] == 2) & (minus_data[:, 4] == 5), 3] += 0.1
            self._write(plus_path, plus_data)
            self._write(minus_path, minus_data)
            plus = MODULE.load_modes(plus_path)
            # The within-file check catches harmonic-dependent metadata first.
            with self.assertRaisesRegex(ValueError, "harmonic-dependent phase"):
                MODULE.load_modes(minus_path)

    def test_node_manifest_sha_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plus_nodes = root / "plus_nodes.dat"
            minus_nodes = root / "minus_nodes.dat"
            plus_nodes.write_text("1 1.0 0.0 0.0 1e11 0.0\n", encoding="utf-8")
            minus_nodes.write_text("1 1.0 0.0 0.0 1e11 0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.require_identical_node_manifests(plus_nodes, minus_nodes)
            same = MODULE.require_identical_node_manifests(plus_nodes, plus_nodes)
            self.assertEqual(len(same), 64)

    def test_missing_manifest_is_rejected(self) -> None:
        missing = Path("/no/such/manifest_for_a0_test.dat")
        with self.assertRaisesRegex(ValueError, "missing"):
            MODULE.require_identical_node_manifests(missing, missing)

    def test_manifest_modes_crosscheck_and_twopi_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            man = root / "nodes.dat"
            # phase written as ~2π must match modes phase 0
            man.write_text(
                "1 0.25 0.0 0.0 1.0e11 6.283185307179586\n"
                "2 0.75 0.0 0.0 2.0e11 1.0\n",
                encoding="utf-8",
            )
            plus = _table("plus")
            plus[plus[:, 0] == 1, 3] = 0.0
            plus_path = root / "plus.dat"
            self._write(plus_path, plus)
            manifest = MODULE.load_node_manifest(man)
            MODULE.crosscheck_modes_against_manifest(
                plus, plus_path, manifest,
                rtol=1.0e-12, atol=1.0e-12, phase_atol=1.0e-6,
            )
            # intensity mismatch must fail
            bad = plus.copy()
            bad[0, 2] *= 1.01
            bad_path = root / "bad.dat"
            self._write(bad_path, bad)
            with self.assertRaisesRegex(ValueError, "intensity mismatch"):
                MODULE.crosscheck_modes_against_manifest(
                    bad, bad_path, manifest,
                    rtol=1.0e-12, atol=1.0e-12, phase_atol=1.0e-6,
                )

    def test_inf_in_modes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inf.dat"
            data = _table("plus")
            data[0, 5] = np.inf
            self._write(path, data)
            with self.assertRaisesRegex(ValueError, "NaN|Inf|non-finite|finite"):
                MODULE.load_modes(path)

    def test_run_metadata_must_match_physics_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plus = root / "plus_meta.txt"
            minus = root / "minus_meta.txt"
            common = (
                "nk=20\nnkx=20\nnky=20\ndt=0.35\nT2_cycles=0.5\nwvl_nm=3200.0\n"
                "squeeze_r=2.5\nsqueeze_theta_deg=0\nharmonics=2,5,7,9,10\n"
                "source_sha256=aaa\nbinary_sha256=bbb\nnodes_sha256=ccc\n"
            )
            plus.write_text(common + "tb_sha256=tb_plus\ncase=plusN\n", encoding="utf-8")
            minus.write_text(common + "tb_sha256=tb_minus\ncase=minusN\n", encoding="utf-8")
            matched = MODULE.compare_run_metadata_pair(plus, minus)
            self.assertEqual(matched["nk"], "20")
            self.assertEqual(matched["nodes_sha256"], "ccc")
            bad = root / "bad_meta.txt"
            bad.write_text(common.replace("nk=20", "nk=40") + "tb_sha256=x\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch on nk"):
                MODULE.compare_run_metadata_pair(plus, bad)


if __name__ == "__main__":
    unittest.main()
