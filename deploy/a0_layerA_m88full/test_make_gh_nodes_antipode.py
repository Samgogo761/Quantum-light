#!/usr/bin/env python3
"""Unit tests for ES25.17 GH manifests + exact antipode enforcement."""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gh_nodes import ALPHA0_TOL, TWOPI, make_sv_nodes, write_manifest  # noqa: E402


def _load_rows(path: Path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        rows.append(
            {
                "id": int(f[0]),
                "w": float(f[1]),
                "re": float(f[2]),
                "im": float(f[3]),
                "I": float(f[4]),
                "phi": float(f[5]),
            }
        )
    return rows


class TestMakeGhNodesAntipode(unittest.TestCase):
    def test_antipodes_exact_after_es25_roundtrip(self):
        rows = make_sv_nodes(2.5, 0.0, 1.0e11, 3)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nodes.dat"
            write_manifest(path, rows, 2.5, 0.0, 3, 1.0e11)
            text = path.read_text(encoding="utf-8")
            self.assertIn("ES25.17", text)
            loaded = _load_rows(path)

        by_alpha = {}
        for r in loaded:
            key = (round(r["re"], 12), round(r["im"], 12))
            by_alpha[key] = r

        max_dphi = 0.0
        for r in loaded:
            a = complex(r["re"], r["im"])
            if abs(a) <= ALPHA0_TOL:
                self.assertLessEqual(abs(r["I"]), 1e-30)
                self.assertLessEqual(abs(r["phi"]), 1e-30)
                continue
            partner = by_alpha[(round(-r["re"], 12), round(-r["im"], 12))]
            self.assertAlmostEqual(r["w"], partner["w"], places=16)
            self.assertAlmostEqual(r["I"], partner["I"], places=16)
            self.assertAlmostEqual(r["re"] + partner["re"], 0.0, places=15)
            self.assertAlmostEqual(r["im"] + partner["im"], 0.0, places=15)
            dphi = abs(((partner["phi"] - (r["phi"] + math.pi)) + math.pi) % TWOPI - math.pi)
            max_dphi = max(max_dphi, dphi)
            self.assertLessEqual(dphi, 1e-15)
        self.assertLess(max_dphi, 1e-14)


if __name__ == "__main__":
    unittest.main()
