#!/usr/bin/env python3
"""Strict JSON / finite / rebuild tests for full112 top diagnosis v2."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_full112_top import (  # noqa: E402
    DEFAULT_DATA,
    OUT_DEFAULT,
    SCHEMA,
    VERDICT,
    build_report,
    dumps_strict,
    nearest_match_errors,
    ordered_unique_errors,
    to_jsonable,
)

import numpy as np

QL = Path(__file__).resolve().parents[2]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def strict_loads(text: str) -> dict:
    require("NaN" not in text, "bare NaN token")
    require("Infinity" not in text, "Infinity token")

    def _reject(name: str):
        raise ValueError(f"non-finite JSON constant {name}")

    return json.loads(text, parse_constant=_reject)


def assert_finite(obj, path="$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        require(math.isfinite(obj), f"non-finite at {path}")


def test_strict_committed_json() -> dict:
    text = OUT_DEFAULT.read_text(encoding="utf-8")
    payload = strict_loads(text)
    assert_finite(payload)
    require(payload["schema"] == SCHEMA, payload.get("schema"))
    for key, val in VERDICT.items():
        require(payload["verdict"][key] == val, f"verdict {key}")
    leftover = payload["leftover_by_trial_P"]
    require(leftover["proxy_band_index_fixed_QE169_200"] is True, "proxy not fixed 169-200")
    require(leftover["proxy_band_index_min"] == 169, "proxy min")
    require(leftover["proxy_band_index_max"] == 200, "proxy max")
    ordered = payload["path_match"]["method_ordered_unique"]["bands_105_112"]
    nearest = payload["path_match"]["method_nearest_neighbor"]["bands_105_112"]
    require(abs(ordered["MAE_eV"] - 0.015) < 0.002, f"ordered MAE {ordered['MAE_eV']}")
    require(abs(ordered["MAX_eV"] - 0.040) < 0.005, f"ordered MAX {ordered['MAX_eV']}")
    require(nearest["MAE_eV"] < ordered["MAE_eV"] + 0.005, "nearest MAE")
    require("content_sha256" in payload, "missing content_sha256")
    require("input_sha256" in payload["provenance"], "missing input sha")
    require("generator_sha256" in payload["provenance"], "missing generator sha")
    return payload


def test_dumps_rejects_nan() -> None:
    try:
        dumps_strict({"x": float("nan")})
    except (ValueError, TypeError):
        return
    raise RuntimeError("dumps_strict accepted NaN")


def test_ordered_unique_synthetic() -> None:
    # Cluster pairing: 8-nearest degenerates to 2-nearest here.
    e_w = np.array([[0.00, 0.10], [0.02, 0.12]])
    e_qe = np.array([[0.00, 0.10], [0.02, 0.12], [5.00, 5.12]])
    near = nearest_match_errors(e_w, e_qe)
    ord_ = ordered_unique_errors(e_w, e_qe)
    require(float(near.max()) < 1e-15, "nearest should be exact")
    require(float(ord_.max()) < 1e-15, "cluster ordered should stay in-shell")
    # A far QE band must not be pulled in.
    require(float(ord_.mean()) < 0.5, "cluster must ignore the 5 eV manifold")


def test_rebuild_if_data_present() -> None:
    if not (DEFAULT_DATA / "wannier" / "CrI3.eig").is_file():
        print("SKIP rebuild: data-root missing")
        return
    report = build_report(DEFAULT_DATA, QL)
    committed = strict_loads(OUT_DEFAULT.read_text(encoding="utf-8"))
    for key in (
        "schema",
        "verdict",
        "path_match",
        "leftover_by_trial_P",
        "truncation_gap_path",
        "qe_mp_grid",
    ):
        a = dumps_strict(to_jsonable(report[key]))
        b = dumps_strict(committed[key])
        require(a == b, f"rebuild mismatch in {key}")


def main() -> int:
    test_dumps_rejects_nan()
    test_ordered_unique_synthetic()
    test_strict_committed_json()
    test_rebuild_if_data_present()
    print("PASS test_diagnose_full112_top")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
