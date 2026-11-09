"""Timestep check on the infrared-extended quadrature. No new electron propagation.

Reuses the saved output-transpose rows at each ntpc. The input measure is the
wing-factor-2, core-160 quadrature with omega_lo=1e-4 and omega_hi=12.
"""
from pathlib import Path
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import q8_m0_gamma12_input_wing_20261008 as wing
import q8_m0_gamma12_transpose_input_refine_20261008 as base
import q8_m0_output_transpose_20261008 as tr

OUT = base.OUT
BACK = base.ROOT / "results/q_line/gamma12_output_transpose_20261008"
OLD = base.ROOT / "results/q_line/gamma12_output_diagnostic_20261007"


def load_rows(ntpc, manifest):
    directory = BACK / f"ntpc_{ntpc}"
    meta = json.loads((directory / "backward.json").read_text(encoding="utf-8"))
    path = directory / "backward.npz"
    if base.prep.sha256_file(path) != meta["sha256"]:
        raise ValueError(f"Backward hash mismatch at ntpc={ntpc}")
    if ntpc == 1024:
        h = float(manifest["time"]["h"])
        nstep = int(manifest["time"]["nstep"])
    else:
        info = json.loads((OLD / f"time_{ntpc}" / "prepared.json").read_text(encoding="utf-8"))
        h, nstep = float(info["h"]), int(info["nstep"])
        if info.get("source_manifest") != manifest["manifest_sha256"]:
            raise ValueError("Refined vertices were not built from the frozen +N manifest")
        if meta["identity"].get("refined_prepared_sha256") != base.prep.sha256_file(OLD / f"time_{ntpc}" / "prepared.json"):
            raise ValueError("Backward rows do not match the saved refined vertices")
    with np.load(path, allow_pickle=False) as z:
        rows = {}
        for label in ("driven", "undriven"):
            forcing = z[label + "_forcing"]
            if forcing.shape[0] != nstep:
                raise ValueError(f"{label} forcing steps {forcing.shape[0]} != {nstep}")
            rows[label] = tr.BackwardRows(z[label + "_row0"], forcing, 2, nstep, h, meta["timing_s"][label])
    return rows, h, nstep


def pack(m):
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist()}


def pieces(vals):
    u0, du, v = vals["u0"], vals["delta_u"], vals["V"]
    u = u0 + du
    normal = u @ u.conj().T - v @ v.conj().T - np.eye(2)
    anomalous = u @ v.T - v @ u.T
    cross = u0 @ du.conj().T + du @ u0.conj().T
    quad = du @ du.conj().T
    vv = v @ v.conj().T
    return {
        "normal": pack(normal),
        "anomalous": pack(anomalous),
        "cross": pack(cross),
        "quad_minus_VV": pack(quad - vv),
        "normal_00": float(normal[0, 0].real),
        "normal_maxabs": float(np.max(np.abs(normal))),
        "anomalous_maxabs": float(np.max(np.abs(anomalous))),
        "U0_defect": float(np.max(np.abs(u0 @ u0.conj().T - np.eye(2)))),
        "cross_00": float(cross[0, 0].real),
        "quad_00": float(quad[0, 0].real),
        "VV_00": float(vv[0, 0].real),
        "pump_00": float((cross + quad - vv)[0, 0].real),
        "n_trace": float(np.trace(vals["n"]).real),
    }


def run(ntpc):
    out = OUT / f"lowend_time_matrix_ntpc_{ntpc}.json"
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    manifest, cache, _rows4096, _info, _identity = base.load_frozen()
    rows, h, nstep = load_rows(ntpc, manifest)
    gamma = float(cache["gamma"])
    pol = np.array(cache["pol"], copy=True)
    prot = base.p1_a2_protocol(base.q.A_STAR)
    op = base.q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, norm = base.q.normalize_phi(op, base.q.phi_unnormalized(op, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    oms, wts, counts = wing.wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0)
    ndark = int(manifest["physical_scope"]["n_dark_columns"])
    old_nph = int(manifest["physical_scope"]["n_photon_columns"])
    print(f"ntpc={ntpc}: {len(oms)} frequencies, {nstep} steps", flush=True)
    vals, timings = wing.contract(
        cache, rows, gamma, pol, phi_of, eta, 0.0, oms, wts, old_nph, ndark)
    summary = pieces(vals)
    summary.update(ntpc=ntpc, h=h, nstep=nstep, n_nodes=int(len(oms)),
                   omega_lo=1.0e-4, omega_hi=12.0, wing_factor=2, core_n_panel=160,
                   panel_request_counts=counts, timings=timings,
                   total_wall_s=time.perf_counter() - started,
                   normal_definition="U @ U.conj().T - V @ V.conj().T - I",
                   production_qualification=False, physical_conclusion=False)
    base.prep.write_json_atomic(out, summary)
    print(json.dumps({k: summary[k] for k in (
        "ntpc", "n_trace", "normal_maxabs", "anomalous_maxabs", "pump_00", "total_wall_s"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntpc", type=int, choices=(1024, 2048), required=True)
    run(parser.parse_args().ntpc)
