#!/usr/bin/env python3
"""Stage 1AE2: test the error ~ Omega_source / omega_max hypothesis.

Stage 1AE showed the real-source kernel error is NOT explained by source
spectral tail, quadrature density, or source time sampling.  The remaining
candidate is the finite kernel rise time ~1/omega_max convolved with the
source's own fast variation: for an oscillatory source at frequency Omega
the instantaneous-kernel target is amplified by ~Omega/omega_max.

Predictions tested here:
1. slow real channels (small spectral edge) pass at much lower cutoff;
2. low-passing the strongest channel at Omega_lp makes the error scale like
   Omega_lp / omega_max on the SAME time grid and SAME quadrature.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_model import build_geometry  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1s_continuum import compare, field_from_kernel, \
    photon_retarded_kernel  # noqa: E402
from q8_m0_stage1ae_source_qual import (  # noqa: E402
    channel_current, kernel_error, load_xref, uv_grid,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1ae_source_qual_20260923"


def lowpass(current, dt, omega0, lp_over):
    nt = current.shape[0]
    freq = np.fft.rfftfreq(nt, dt) * 2.0 * np.pi / omega0
    keep = freq <= lp_over
    out = np.zeros_like(current)
    for a in range(2):
        spec = np.fft.rfft(current[:, a])
        spec[~keep] = 0.0
        out[:, a] = np.fft.irfft(spec, n=nt)
    return out


def main():
    ref = load_xref(0.8, 2048)
    geo = build_geometry()
    _, e_perp = parallel_perp_basis(geo)
    x, dt, omega0 = ref["X_full"], ref["dt"], ref["omega0"]

    tot = np.sum(np.abs(x) ** 2, axis=0)
    flat = np.argsort(tot.ravel())[::-1]
    chans = [tuple(int(i) for i in np.unravel_index(f, tot.shape))
             for f in flat[:6]]
    cutoffs = [64.0, 128.0, 192.0]

    chan_rows = []
    for ch in chans:
        cur = channel_current(x, ch, e_perp)
        # spectral edges of this channel
        spec = np.abs(np.fft.rfft(cur[:, 0])) ** 2
        freq = np.fft.rfftfreq(len(cur), dt) * 2 * np.pi / omega0
        cum = np.cumsum(spec) / np.sum(spec)
        edge99 = float(freq[min(int(np.searchsorted(cum, 0.99)),
                                len(freq) - 1)])
        for cutoff in cutoffs:
            om, wt = uv_grid(omega0, cutoff)
            met = kernel_error(cur, dt, om, wt)
            chan_rows.append({"channel": ch, "edge99_over_omega0": edge99,
                              "cutoff_over_omega0": cutoff,
                              "A_rel_max": met["A_rel_max"],
                              "dA_rel_max": met["dA_rel_max"],
                              "A_source_norm_l2": met["A_source_norm_l2"]})
            print(f"ch={ch} edge99={edge99:6.1f} cut={cutoff:5.0f} "
                  f"Arel={met['A_rel_max']:.4g} "
                  f"srcL2={met['A_source_norm_l2']:.4g}", flush=True)

    # low-pass the strongest channel: error must track Omega_lp
    ch0 = chans[0]
    cur0 = channel_current(x, ch0, e_perp)
    lp_rows = []
    for lp in (8.0, 16.0, 32.0, 64.0):
        cur_lp = lowpass(cur0, dt, omega0, lp)
        for cutoff in (64.0, 128.0):
            om, wt = uv_grid(omega0, cutoff)
            met = kernel_error(cur_lp, dt, om, wt)
            lp_rows.append({"lowpass_over_omega0": lp,
                            "cutoff_over_omega0": cutoff,
                            "A_rel_max": met["A_rel_max"],
                            "dA_rel_max": met["dA_rel_max"],
                            "A_source_norm_l2": met["A_source_norm_l2"]})
            print(f"lp={lp:5.0f} cut={cutoff:4.0f} "
                  f"Arel={met['A_rel_max']:.4g} "
                  f"srcL2={met['A_source_norm_l2']:.4g}", flush=True)

    payload = {
        "status": "error_scaling_diagnostic_prescribed_sources_only",
        "hypothesis": "kernel error ~ Omega_source/omega_max (finite kernel "
                      "rise time), not tail/density/sampling",
        "channel_rows": chan_rows,
        "lowpass_rows": lp_rows,
    }
    target = OUT / "stage1ae2_channel_scan.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
