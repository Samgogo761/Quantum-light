"""Time-step ladder at fixed two-k production quadrature.

Same 526 frequencies, same initial columns, same eight-cycle P1.
Reports residuals against I, against the actual U0 gram, and the
drive-induced change. Does not mix other frequency sets into the extrapolation.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step
from q8_m0_g1_production_time_integral_20261014 import prepare, vertices
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_two_k_timestep_20261014/timestep_ladder.json"


def err(a):
    return float(np.max(np.abs(np.asarray(a))))


def pack(m):
    m = np.asarray(m)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(), "maxabs": err(m)}


def propagate(prep, verts, h):
    state = prep["initial"].copy()
    pol, gamma = prep["pol"], prep["gamma"]
    omega, free = prep["omega"], prep["free"]
    times = (np.arange(len(verts)) + 0.5) * h
    k_all, _ = kernel_at_times(gamma, prep["drive"], times)
    acc_u = np.zeros((2, state.b.shape[1]), complex)
    acc_v = np.zeros_like(acc_u)
    for step, (u, d) in enumerate(verts):
        t = step * h
        prev = state
        fa = free * (np.exp(-1j * omega * (t + h)) - np.exp(-1j * omega * t))
        dreal = np.real(d)
        state = midpoint_step(state, u, dreal, h, gamma, free_A_increment=fa)
        mid = ColumnState(0.5 * (prev.b + state.b), 0.5 * (prev.bdag + state.bdag),
                          0.5 * (prev.A + state.A))
        src = pol @ current(mid, u, dreal)
        acc_u += h * k_all[step] * src
        acc_v += h * k_all[step] * np.conj(src)
    return acc_u, acc_v


def matrices(prep, du, dv):
    u = prep["u0"] + du
    uu = u @ u.conj().T
    vv = dv @ dv.conj().T
    gram = uu - vv
    g_u0 = prep["u0"] @ prep["u0"].conj().T
    eye = np.eye(2)
    return {
        "against_I": gram - eye,
        "against_U0_gram": gram - g_u0,
        "against_mass_identity": gram - prep["mass"] * eye,
        "anomalous": u @ dv.T - dv @ u.T,
    }


def safe_ratio(num, den):
    out = np.zeros(num.shape, complex)
    mask = np.abs(den) > 1e-10
    out[mask] = num[mask] / den[mask]
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prep = prepare()
    if prep["n_freq"] != 526:
        raise AssertionError(f"Frequency count changed: {prep['n_freq']}")
    g_u0 = prep["u0"] @ prep["u0"].conj().T
    levels = {}
    for ntpc in (128, 256, 512, 1024):
        h = float(prep["drive"].period) / ntpc
        nstep = 8 * ntpc
        print("ntpc", ntpc, "steps", nstep, flush=True)
        quiet = vertices(prep, 0.0, 0.0, nstep, h)
        driven = vertices(prep, 0.8, 0.0, nstep, h)
        uu, uv = propagate(prep, quiet, h)
        du, dv = propagate(prep, driven, h)
        got = matrices(prep, du - uu, dv - uv)
        levels[str(ntpc)] = {name: pack(mat) for name, mat in got.items()}
        write_new_json(OUT.with_name(f"level_ntpc_{ntpc}.json"),
                       {"ntpc": ntpc, "h": h, "steps": nstep, **levels[str(ntpc)]})
        print("  RN", got["against_I"].real[0, 0], "dC", got["against_U0_gram"].real[0, 0],
              "RA", err(got["anomalous"]), flush=True)
    order = ["128", "256", "512", "1024"]

    def series(name):
        return [np.array(levels[k][name]["real"]) + 1j * np.array(levels[k][name]["imag"])
                for k in order]

    extrap = {}
    for name in ("against_I", "against_U0_gram", "against_mass_identity", "anomalous"):
        mats = series(name)
        d1, d2, d3 = mats[1] - mats[0], mats[2] - mats[1], mats[3] - mats[2]
        extrap[name] = {
            "difference_128_to_256": pack(d1),
            "difference_256_to_512": pack(d2),
            "difference_512_to_1024": pack(d3),
            "ratio_first_over_second": pack(safe_ratio(d1, d2)),
            "ratio_second_over_third": pack(safe_ratio(d2, d3)),
            "richardson_512_1024": pack(mats[3] + d3 / 3.0),
        }
    report = {
        "frequencies": prep["n_freq"],
        "packet_mass": prep["mass"],
        "U0_gram": pack(g_u0),
        "U0_gram_minus_mass_identity": pack(g_u0 - prep["mass"] * np.eye(2)),
        "cycles": 8,
        "levels": levels,
        "extrapolation": extrap,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__, Path(__file__).with_name(
            "q8_m0_g1_production_time_integral_20261014.py")]),
    }
    write_new_json(OUT, report)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
