#!/usr/bin/env python3
"""Stage 1AE: real-source photon-continuum qualification, axes separated.

Uses the converged X_ref(t) stored by stage 1AD as a prescribed c-number
current (no back-coupling to electrons).  The two axes that earlier runs
mixed together are separated:

1. UV cutoff omega_max at saturated quadrature density;
2. quadrature node density at fixed cutoff;
3. time-grid sampling of the source itself (X_ref subsampled);
4. analytic truncation estimate: source power above each cutoff.

Gate: the strongest real X(t) channel must recover the causal Maxwell kernel
(dot kappa = gamma I, kappa(0)=0) before any coupled run is qualified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_model import build_geometry  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import (  # noqa: E402
    compare, field_from_kernel, photon_retarded_kernel,
)
from q8_m0_stage1v_real_source_kernel import gauss_two_segment, open_grid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1ae_source_qual_20260923"


def load_xref(a_star: float, ntpc: int):
    path = REPO / "results/q_line/stage1ad_xref_20260923" / (
        f"xref_a{a_star:g}_ntpc{ntpc}.npz")
    if not path.exists():
        raise FileNotFoundError(f"run stage1ad first: {path}")
    z = np.load(path)
    return {"times": z["times"], "X_full": z["X_full"], "dt": float(z["dt"]),
            "omega0": float(z["omega0"]), "ntpc": int(z["ntpc"])}


def strongest_channel(x):
    tot = np.sum(np.abs(x) ** 2, axis=0)
    return tuple(int(i) for i in np.unravel_index(np.argmax(tot), tot.shape))


def channel_current(x, ch, e_perp):
    src_complex = x[(slice(None),) + ch]
    return np.column_stack([
        np.real(src_complex) * e_perp[0] - np.imag(src_complex) * e_perp[1],
        np.real(src_complex) * e_perp[1] + np.imag(src_complex) * e_perp[0],
    ])


def kernel_error(current, dt, omega, weight, gamma=1.0):
    nt = current.shape[0]
    lag = dt * np.arange(nt)
    k = photon_retarded_kernel(lag, omega, weight, gamma)
    target = field_from_kernel(current, np.r_[0.0, np.full(nt - 1, gamma)], dt)
    field = field_from_kernel(current, k, dt)
    return compare(field, target, current, dt, gamma)


def uv_grid(omega0, cutoff_over, n_low=96, uv_per_octave=384):
    """Low band (0, 8 w0] fixed; UV wing (8 w0, cutoff] density-controlled."""
    if cutoff_over <= 8:
        return open_grid(cutoff_over * omega0, n_low)
    octaves = max(1, int(round(np.log2(cutoff_over / 8.0))))
    return gauss_two_segment(0.0, 8.0 * omega0, cutoff_over * omega0,
                             n_low, uv_per_octave * octaves)


def tail_power_fraction(current, dt, omega0, cutoffs):
    nt = current.shape[0]
    freq = np.fft.rfftfreq(nt, dt) * 2.0 * np.pi / omega0
    power = (np.abs(np.fft.rfft(current[:, 0])) ** 2
             + np.abs(np.fft.rfft(current[:, 1])) ** 2)
    tot = np.sum(power)
    return {f"{c:g}": float(np.sum(power[freq > c]) / tot) for c in cutoffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-star", type=float, default=0.8)
    ap.add_argument("--ntpc", type=int, default=None,
                    help="X_ref grid; default = finest available")
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[8, 16, 32, 64, 128, 192, 256])
    ap.add_argument("--subsample", type=int, nargs="+", default=[4, 2, 1],
                    help="source time-grid thinning factors")
    args = ap.parse_args()

    ref_dir = REPO / "results/q_line/stage1ad_xref_20260923"
    avail = sorted(int(p.stem.split("ntpc")[1]) for p in ref_dir.glob(
        f"xref_a{args.a_star:g}_ntpc*.npz"))
    if not avail:
        raise SystemExit("no X_ref stored; run q8_m0_stage1ad_xref.py first")
    ntpc = args.ntpc or avail[-1]
    ref = load_xref(args.a_star, ntpc)
    geo = build_geometry()
    _, e_perp = parallel_perp_basis(geo)
    x, dt, omega0 = ref["X_full"], ref["dt"], ref["omega0"]
    ch = strongest_channel(x)
    current_full = channel_current(x, ch, e_perp)
    print(f"X_ref ntpc={ntpc} channel={ch} dt={dt:.5f} "
          f"(Nyq {np.pi / dt / omega0:.0f} w0)", flush=True)

    rows = []
    tails = {}
    for sub in args.subsample:
        cur = current_full[::sub]
        dts = dt * sub
        tails[f"sub{sub}"] = tail_power_fraction(cur, dts, omega0,
                                                 args.cutoffs)
        for cutoff in args.cutoffs:
            omega, weight = uv_grid(omega0, cutoff)
            met = kernel_error(cur, dts, omega, weight)
            row = {"subsample": sub,
                   "eff_ntpc": ntpc // sub,
                   "cutoff_over_omega0": cutoff,
                   "n_nodes": int(len(omega)),
                   "tail_power_above_cutoff": tails[f"sub{sub}"][f"{cutoff:g}"],
                   "A_rel_max": met["A_rel_max"],
                   "A_rel_l2": met["A_rel_l2"],
                   "dA_rel_max": met["dA_rel_max"],
                   "A_source_norm_l2": met["A_source_norm_l2"]}
            rows.append(row)
            print(f"sub={sub} cut={cutoff:>5.0f} n={row['n_nodes']:>5d} "
                  f"Arel={met['A_rel_max']:.4g} dArel={met['dA_rel_max']:.4g} "
                  f"tail={row['tail_power_above_cutoff']:.3g}", flush=True)

    # axis 2: node-density doubling at fixed cutoffs, finest source grid
    density_rows = []
    for cutoff in (64.0, 128.0):
        for nuv in (192, 384, 768, 1536):
            omega, weight = gauss_two_segment(
                0.0, 8.0 * omega0, cutoff * omega0, 96, nuv)
            met = kernel_error(current_full, dt, omega, weight)
            density_rows.append({"cutoff_over_omega0": cutoff,
                                 "n_uv": nuv, "n_nodes": int(len(omega)),
                                 "A_rel_max": met["A_rel_max"],
                                 "dA_rel_max": met["dA_rel_max"],
                                 "A_source_norm_l2": met["A_source_norm_l2"]})
            print(f"density cut={cutoff:.0f} n_uv={nuv} "
                  f"Arel={met['A_rel_max']:.4g}", flush=True)

    payload = {
        "status": "prescribed_real_source_qualification_axes_separated",
        "xref_ntpc": ntpc, "a_star": args.a_star, "channel": ch,
        "source_nyquist_over_omega0": float(np.pi / dt / omega0),
        "cutoff_rows": rows,
        "density_rows": density_rows,
        "tail_power_fraction": tails,
        "qualification_note": "kernel qualified requires A_rel_max and "
                              "dA_rel_max < 1% at saturated node density",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "stage1ae_source_qual.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
