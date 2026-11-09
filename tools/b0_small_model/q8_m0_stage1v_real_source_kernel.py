#!/usr/bin/env python3
"""Stage 1V: photon-kernel benchmark using actual p-h current waveforms X(t).

This is not a production calculation. It directly addresses the audit point:
"don't guess omega_max; diagnose the spectrum of X(t), then use real electronic
waveforms as prescribed currents to test the photon basis."

The script only checks the Maxwell bridge: for representative X(t) channels,
can the photon quadrature basis still recover the causal self-field
  dot(kappa_self) = gamma I(t),   kappa_self(0)=0 ?
It does not run the coupled p-h/photon propagation and does not infer areas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_cjj import propagate_cjj  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import (  # noqa: E402
    compare, field_from_kernel, photon_retarded_kernel,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1v_real_source_kernel_20260923"
MU = -0.009010620372198276
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])


def open_grid(omega_max: float, n: int):
    """Positive-frequency Gauss-Legendre grid on (0, omega_max]."""
    x, w = leggauss(int(n))
    om = 0.5 * omega_max * (x + 1.0)
    wt = 0.5 * omega_max * w
    return om, wt


def gauss_two_segment(lo, mid, hi, n_lo, n_hi):
    """Return (omega, weights) with IR emphasis and separate UV wing."""
    o1, w1 = open_grid(mid - lo, n_lo)
    o1 = o1 + lo
    o2, w2 = open_grid(hi - mid, n_hi)
    o2 = o2 + mid
    return np.r_[o1, o2], np.r_[w1, w2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntpc", type=int, default=256)
    parser.add_argument("--omega-max", type=float, nargs="+",
                        default=[8.0, 16.0, 32.0, 64.0, 128.0])
    parser.add_argument("--n-per-segment", type=int, default=48)
    args = parser.parse_args()

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    _, e_perp = parallel_perp_basis(geo)

    print(f"propagate_cjj (2 k, P1, ntpc={args.ntpc}) ...", flush=True)
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=args.ntpc, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=False,
    )
    times = np.asarray(rec["times"])
    dt = float(rec["dt"])
    x = np.asarray(rec["X_full"])  # (nt, nk, 2, c, v)
    nt = len(times)
    lag = dt * np.arange(nt)

    # choose representative channels: strongest total power, plus the top few
    # of the second k-point (where runaway was observed).
    tot = np.sum(np.abs(x)**2, axis=0)
    strongest = np.unravel_index(np.argmax(tot), tot.shape)
    channels = [("strongest", strongest)]
    # top-5 channels in k=1
    tot_k1 = tot[1].flatten()
    top5 = np.argsort(tot_k1)[-5:][::-1]
    for rank, flat in enumerate(top5, start=1):
        a, c, v = np.unravel_index(flat, (2, 4, 4))
        channels.append((f"k1_top{rank}_a{a}_c{c}_v{v}", (1, int(a), int(c), int(v))))

    # target causal kernel: dot(kappa)=gamma I(t), kappa(0)=0
    gamma = 1.0  # arbitrary; compare is scale-invariant for A_err
    target_kernel = np.r_[0.0, np.full(nt - 1, gamma)]

    rows = []
    for label, ch in channels:
        kidx, a, c, v = ch
        # build 2D prescribed current in the perp direction for this channel
        src_complex = x[:, kidx, a, c, v]
        src = np.column_stack([
            np.real(src_complex) * e_perp[0] - np.imag(src_complex) * e_perp[1],
            np.real(src_complex) * e_perp[1] + np.imag(src_complex) * e_perp[0],
        ])
        I_t = np.einsum("ta,a->t", src, e_perp)
        # diagnostic: spectral content of this channel (cheap DFT)
        spec = np.fft.rfft(I_t)
        freq = np.fft.rfftfreq(nt, dt) * 2 * np.pi
        pow_spec = np.abs(spec)**2
        cum = np.cumsum(pow_spec) / np.sum(pow_spec)
        def pct(th):
            i = min(int(np.searchsorted(cum, th)), len(freq) - 1)
            return float(freq[i])
        spec_edges = {f"{int(th*100)}pct": pct(th) for th in [0.9, 0.99, 0.999, 0.9999]}

        for omax in args.omega_max:
            om, wt = gauss_two_segment(0.0, 0.5 * omax * prot.omega0,
                                         omax * prot.omega0,
                                         args.n_per_segment, args.n_per_segment)
            k = photon_retarded_kernel(lag, om, wt, gamma)
            from_photons = field_from_kernel(src, k, dt)
            target = field_from_kernel(src, target_kernel, dt)
            kcheck_target = compare(from_photons, target, src, dt, gamma)
            row = {
                "channel": label,
                "k": int(kidx), "a": int(a), "c": int(c), "v": int(v),
                "omega_max_over_omega0": float(omax),
                "n_nodes": int(len(om)),
                "A_rel_max": kcheck_target["A_rel_max"],
                "A_rel_l2": kcheck_target["A_rel_l2"],
                "dA_rel_max": kcheck_target["dA_rel_max"],
                "A_final_over_Maxwell_final": kcheck_target["A_final_over_Maxwell_final"],
                "spectral_edges": spec_edges,
                "kernel_qualified": bool(
                    kcheck_target["A_rel_max"] < 0.01 and
                    kcheck_target["dA_rel_max"] < 0.01
                ),
            }
            rows.append(row)
            print(f"{label} omax={omax} Aerr={row['A_rel_max']:.4g} "
                  f"dAerr={row['dA_rel_max']:.4g} qual={row['kernel_qualified']}",
                  flush=True)

    payload = {
        "status": "real_source_kernel_benchmark_only",
        "theory": "P1_dressed_particle_hole_photon_canonical_Gaussian",
        "n_k": 2,
        "k_frac": K_FRAC.tolist(),
        "ntpc": args.ntpc,
        "n_t": nt,
        "dt": dt,
        "gamma_unit_test": gamma,
        "channels_benchmarked": len(channels),
        "rows": rows,
        "qualification_note": "Kernel check uses representative p-h waveforms "
                              "as prescribed currents. Coupled propagation not run.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "stage1v_real_source_kernel.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
