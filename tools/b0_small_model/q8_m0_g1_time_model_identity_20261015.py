"""Identity of the saved fourth-order rows, and the time-extrapolation budget.

Does not rebuild rows and does not overwrite them. The even-power fit used
earlier is reported beside an h^4+h^5 fit. Neither fit is a symmetry proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gamma12_time_order_20261015"
OUT = OUTDIR / "time_model_and_cache_identity.json"
SCRIPT = ROOT / "tools/b0_small_model/q8_m0_g1_gamma12_ir_tail_20261015.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_from(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def cache_report(path, macro, period):
    with np.load(path, allow_pickle=False) as saved:
        report = {"sha256": sha256(path), "bytes": path.stat().st_size, "keys": saved.files}
        width = period / macro
        for label in ("d", "u"):
            step = np.array(saved[f"step_{label}"])
            time = np.array(saved[f"time_{label}"])
            force = saved[f"force_{label}"]
            row = saved[f"row_{label}"]
            rounded = np.round(step, 12)
            values, counts = np.unique(rounded, return_counts=True)
            report[label] = {
                "row_shape": list(row.shape),
                "force_shape": list(force.shape),
                "row_finite": bool(np.isfinite(row).all()),
                "step_values": values.tolist(),
                "step_counts": counts.astype(int).tolist(),
                "time_min": float(time.min()),
                "time_max": float(time.max()),
            }
        same_time = bool(np.array_equal(saved["time_d"], saved["time_u"])
                         and np.array_equal(saved["step_d"], saved["step_u"]))
    expected_max = (9 * macro - 1) * width + 0.5 * width
    report["same_driven_undriven_grid"] = same_time
    report["expected_macro_step"] = width
    report["expected_half_step"] = 0.5 * width
    report["expected_time_max"] = expected_max
    report["time_max_error"] = abs(report["d"]["time_max"] - expected_max)
    report["covered_periods"] = 9
    report["matches_expected_grid"] = bool(
        same_time and report["time_max_error"] < 1e-9
        and report["d"]["row_shape"] == [4, 4610]
        and report["d"]["force_shape"] == [3 * 9 * macro, 4, 2]
        and report["d"]["step_counts"] == [2 * 9 * macro, 9 * macro]
    )
    return report


def fit_h4_h5(fine, mid, coarse):
    """R(tau)=R0+A tau^4+B tau^5, tau=1,2,4 at the three stored levels."""
    d1 = mid - fine
    d2 = coarse - fine
    odd = (15.0 * d2 - 255.0 * d1) / 7440.0
    even = (d1 - 31.0 * odd) / 15.0
    return fine - even - odd, even, odd


def fit_h4_h6(fine, mid, coarse):
    d1 = mid - fine
    d2 = coarse - fine
    higher = (d2 - 17.0 * d1) / 3024.0
    even = (d1 - 63.0 * higher) / 15.0
    return fine - even - higher, even, higher


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prot = p1_a2_protocol(0.8)
    period = float(prot.period)
    caches = {}
    for macro in (1024, 2048):
        caches[str(macro)] = cache_report(OUTDIR / f"adjoint_rows_macro{macro}.npz", macro, period)
    levels = {}
    for macro, name, key in (
        (512, "richardson_corrected_macro512.json", "final_nine_panels"),
        (1024, "richardson_corrected_macro1024.json", "final_nine_panels"),
        (2048, "richardson_macro2048_final.json", "final_nine"),
    ):
        block = json.loads((OUTDIR / name).read_text(encoding="utf-8"))[key]
        levels[macro] = {kind: matrix_from(block[kind]) for kind in ("normal", "anomalous")}
    models = {}
    for kind in ("normal", "anomalous"):
        fine, mid, coarse = levels[2048][kind], levels[1024][kind], levels[512][kind]
        step = mid - fine
        predicted_h4 = mid + 16.0 * step
        predicted_h5 = mid + 32.0 * step
        limit_h4 = fine - step / 15.0
        limit_h5 = fine - step / 31.0
        limit_45, even4, odd5 = fit_h4_h5(fine, mid, coarse)
        limit_46, even4b, even6 = fit_h4_h6(fine, mid, coarse)
        probe = np.array([-1.7e-4, -1.6e-4, -1.2e-4], float)
        synthetic = probe[0] + 1.0e-6 * np.array([1.0, 16.0, 256.0]) + 2.0e-8 * np.array([1.0, 32.0, 1024.0])
        got, _, _ = fit_h4_h5(synthetic[0], synthetic[1], synthetic[2])
        if abs(got - probe[0]) > 1e-15:
            raise AssertionError("h4+h5 solver failed its exact example")
        models[kind] = {
            "held_out_512_error_if_leading_h4": pack(predicted_h4 - coarse),
            "held_out_512_error_if_leading_h5": pack(predicted_h5 - coarse),
            "two_point_limit_h4": pack(limit_h4),
            "two_point_limit_h5": pack(limit_h5),
            "two_point_limit_difference": pack(limit_h4 - limit_h5),
            "three_point_h4_h5_limit": pack(limit_45),
            "three_point_h4_h5_h5_coefficient": pack(odd5),
            "three_point_h4_h6_limit": pack(limit_46),
            "three_point_h4_h6_h6_coefficient": pack(even6),
            "h4_h5_versus_h4_h6": pack(limit_45 - limit_46),
        }
    script_hash = sha256(SCRIPT)
    published = json.loads((OUTDIR / "richardson_ir_tail_macro2048.json").read_text(encoding="utf-8"))
    recorded = published["provenance_sha256"][str(SCRIPT.resolve())]
    write_new_json(OUT, {
        "pulse_periods": int(prot.n_cycles),
        "sampled_periods_in_adjoint_cache": 9,
        "pulse_end": float(prot.t_pulse),
        "caches": caches,
        "baseline_without_infrared_tail": models,
        "infrared_script_hash_matches_published_json": script_hash == recorded,
        "loader_does_not_check_this_record": True,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    normal = models["normal"]
    print("grid", caches["1024"]["matches_expected_grid"], caches["2048"]["matches_expected_grid"])
    print("h4 holdout", normal["held_out_512_error_if_leading_h4"]["maxabs"])
    print("h5 holdout", normal["held_out_512_error_if_leading_h5"]["maxabs"])
    print("limits differ", normal["two_point_limit_difference"]["maxabs"],
          normal["h4_h5_versus_h4_h6"]["maxabs"])


if __name__ == "__main__":
    main()
