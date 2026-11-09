"""Element sources of the infrared-baseline residual.

Frequency bins select nodes by location. They are not complete quadrature
panels. A large bin sum is a contribution, not an integration error.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json

OUT = Path("results/q_line/gamma12_output_transpose_input_refine_20261008")
EDGES = (1e-4, 0.001, 0.02, 0.07, 0.13, 0.3, 1.0, 2.5, 6.0, 12.0)


def pack(m):
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def split_bands(path):
    z = np.load(path)
    u0, du, v, om = z["u0"], z["delta_u"], z["V"], z["oms"]
    u = u0 + du
    w = np.repeat(om, 4)
    bounds = [0.0, *EDGES, 1e9]
    labels = ["<1e-4", "1e-4--0.001", "0.001--0.02", "0.02--0.07", "0.07--0.13",
              "0.13--0.3", "0.3--1", "1--2.5", "2.5--6", "6--12", ">12"]
    bands = {}
    for label, a, b in zip(labels, bounds[:-1], bounds[1:]):
        ids = np.flatnonzero((w >= a) & (w < b))
        if ids.size == 0:
            continue
        bands[label] = _block(u0[:, ids], du[:, ids], v[:, ids], int(ids.size))
    dark = np.arange(len(w), u.shape[1])
    bands["dark"] = _block(u0[:, dark], du[:, dark], v[:, dark], int(dark.size))
    total = u @ u.conj().T - v @ v.conj().T - np.eye(2)
    return {"bands": bands, "normal": pack(total),
            "anomalous": pack(u @ v.T - v @ u.T)}


def _block(u0, du, v, n):
    cross = u0 @ du.conj().T + du @ u0.conj().T
    quad = du @ du.conj().T
    vv = v @ v.conj().T
    normal = (u0 + du) @ (u0 + du).conj().T - vv
    return {"columns": n, "normal_00": float(normal[0, 0].real),
            "cross_00": float(cross[0, 0].real),
            "quad_00": float(quad[0, 0].real),
            "VV_00": float(vv[0, 0].real)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "residual_source_4096.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use --output with a new path: {args.output}")
    report = {"band_split_4096_omega_lo_1e-4": split_bands(OUT / "endpoint_low.npz"),
              "provenance_sha256": provenance([__file__, OUT / "endpoint_low.npz"]),
              "node_grouping_not_panel_integral": True,
              "note": "Band split of the omega_lo=1e-4, ntpc=4096 output. Bins are node-location groups, not complete quadrature panels. A trial internal b-bdag gram was removed because it is not the output CCR."}
    write_new_json(args.output, report)
    bands = report["band_split_4096_omega_lo_1e-4"]["bands"]
    for name, rec in bands.items():
        print(f"{name:12} cols {rec['columns']:6d} normal00 {rec['normal_00']: .6f} "
              f"cross {rec['cross_00']: .4f} quad {rec['quad_00']: .4f} VV {rec['VV_00']: .4f}")


if __name__ == "__main__":
    main()
