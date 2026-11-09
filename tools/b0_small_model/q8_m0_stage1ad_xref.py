#!/usr/bin/env python3
"""Stage 1AD: electron-background reference X_ref(t), D_ref(t) qualification.

No photons, no occupation.  The electronic vertex X(t) and contact D(t) are
propagated on a staircase of time grids; consecutive grids are compared at
their common sample times, and the cumulative spectrum edges of X are tracked.
The finest grid whose trajectory and spectrum are stable against the previous
one becomes X_ref and is stored as npz for all later stage-1 use.

This is the first of the staged qualification gates: the earlier claim of a
"low-step stable plateau" was an artefact of endpoint averaging, and the
99.9% spectral edge measured at ntpc=256 sits at 111.6 omega0, close to that
grid's Nyquist (128 omega0), so the ntpc=256 spectrum is not final.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1ad_xref_20260923"
MU = -0.009010620372198276
K_FRAC = np.array([[0.0, 0.0], [0.17, 0.41]])


def spectrum_edges(x, dt, omega0, thresholds=(0.5, 0.9, 0.99, 0.999, 0.9999)):
    """Total-power cumulative edges across all channels, in units of omega0."""
    nt = x.shape[0]
    freq_full = np.fft.fftfreq(nt, dt) * 2.0 * np.pi
    flat = x.reshape(nt, -1)
    power_full = np.zeros(nt)
    for j in range(flat.shape[1]):
        power_full += np.abs(np.fft.fft(flat[:, j])) ** 2
    pos = freq_full >= 0.0
    freq = freq_full[pos]
    power = power_full[pos]
    cum = np.cumsum(power) / np.sum(power)
    edges = {}
    for th in thresholds:
        i = min(int(np.searchsorted(cum, th)), len(freq) - 1)
        edges[f"{th:g}"] = float(freq[i] / omega0)
    return edges, float(freq[-1] / omega0)


def compare_grids(coarse, fine):
    """Rel difference of X at common sample times; fine must be a refinement."""
    xc, xf = coarse["X_full"], fine["X_full"]
    ratio = (xf.shape[0] - 1) // (xc.shape[0] - 1)
    xf_sub = xf[::ratio]
    den = max(float(np.max(np.abs(xf_sub))), 1e-30)
    max_abs = float(np.max(np.abs(xf_sub - xc)))
    l2 = float(np.linalg.norm((xf_sub - xc).ravel()) /
               max(np.linalg.norm(xf_sub.ravel()), 1e-30))
    return {"refinement_ratio": int(ratio),
            "max_abs_diff_over_max_X": max_abs / den,
            "l2_rel_diff": l2}


def run_one(a_star, ntpc, geo, save):
    t0 = time.perf_counter()
    prot = p1_a2_protocol(a_star)
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=ntpc, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=K_FRAC, save_full_x=True, save_mean_D=True,
    )
    wall = time.perf_counter() - t0
    edges, nyq = spectrum_edges(rec["X_full"], float(rec["dt"]), prot.omega0)
    if save:
        np.savez_compressed(
            OUT / f"xref_a{a_star:g}_ntpc{ntpc}.npz",
            times=rec["times"], X_full=rec["X_full"],
            mean_D_t=rec["mean_D_t"], w_k=rec["w_k"],
            dt=float(rec["dt"]), omega0=prot.omega0, ntpc=ntpc, a_star=a_star,
        )
    return rec, {"ntpc": ntpc, "n_t": int(len(rec["times"])),
                 "dt": float(rec["dt"]),
                 "nyquist_over_omega0": nyq,
                 "spectrum_edges_over_omega0": edges,
                 "wall_seconds": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-star", type=float, nargs="+", default=[0.8])
    ap.add_argument("--ntpc", type=int, nargs="+", default=[256, 512, 1024])
    args = ap.parse_args()

    geo = build_geometry()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "stage1ad_xref.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) \
        if summary_path.exists() else {
            "status": "electron_background_reference_only_no_photons",
            "model": "frozen M0/R0, two k probe points",
            "k_frac": K_FRAC.tolist(), "mu": MU, "runs": {}}
    for a_star in args.a_star:
        key = f"a{a_star:g}"
        runs = payload["runs"].get(key, [])
        recs = {}
        for ntpc in args.ntpc:
            if any(r["ntpc"] == ntpc for r in runs):
                recs[ntpc] = None
                continue
            rec, row = run_one(a_star, ntpc, geo, save=True)
            recs[ntpc] = rec
            runs.append(row)
            runs.sort(key=lambda r: r["ntpc"])
            print(f"a*={a_star} ntpc={ntpc} 99.9% edge "
                  f"{row['spectrum_edges_over_omega0']['0.999']:.1f} w0 "
                  f"(Nyq {row['nyquist_over_omega0']:.0f} w0) "
                  f"wall={row['wall_seconds']:.0f}s",
                  flush=True)
        # grid-to-grid comparison at common times (recompute coarse grids
        # only if they were skipped above; otherwise load stored npz)
        stored = {}
        for row in runs:
            ntpc = row["ntpc"]
            data = recs.get(ntpc)
            if data is None:
                npz = np.load(OUT / f"xref_a{a_star:g}_ntpc{ntpc}.npz")
                data = {"X_full": npz["X_full"]}
            stored[ntpc] = data
        nts = sorted(stored)
        for a, b in zip(nts[:-1], nts[1:]):
            cmp_row = {"pair": [a, b], **compare_grids(stored[a], stored[b])}
            runs[next(i for i, r in enumerate(runs) if r["ntpc"] == b)][
                "vs_previous_grid"] = cmp_row
            print(f"  pair {a}->{b}: maxrel={cmp_row['max_abs_diff_over_max_X']:.3e} "
                  f"l2={cmp_row['l2_rel_diff']:.3e}", flush=True)
        payload["runs"][key] = runs
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
