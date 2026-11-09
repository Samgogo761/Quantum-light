#!/usr/bin/env python3
"""Zero-pump M0 electronic spectral weights vs frequency and grid.

Algebraic identity (K^R-K^A vs S) is recorded separately from finite-grid
/ eta stability of smooth-window integrals. Not photon number, not P1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, window_fourier, x_from_bases
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases

REPO = Path(__file__).resolve().parents[2]
MU = -0.009010620372198276
OMEGA0_OVER_TH = float(p1_a2_protocol(0.0).omega0)

OUT = REPO / "results/q_line/stage1ao_electronic_spectrum_20261004"
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_128x128", 128, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)
ETA_IDENTITY = 1e-4
ETA_SHAPE = (1e-3, 3e-4, 1e-4)
N_BINS = 64
HIST_MAX = 8.0
N_SHAPE = 24
WEIGHT_ONCE = (
    "C_ab,p = w_k X_a*(kcv) X_b(kcv); gaps keep every (k,c,v). "
    "Do not also multiply sqrt(w) on the columns."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_matrix(m):
    m = np.asarray(m)
    herm = 0.5 * (m + m.conj().T)
    return {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "maxabs": float(np.max(np.abs(m))),
        "hermitian_eigenvalues": np.linalg.eigvalsh(herm).tolist(),
        "hermitian_maxabs": float(np.max(np.abs(herm))),
        "anti_hermitian_maxabs": float(np.max(np.abs(m - herm))),
    }


def pack_c(m):
    rec = pack_matrix(m)
    rec["trace_real"] = float(np.trace(np.asarray(m)).real)
    return rec


def spectral_data(gap, x, weights):
    """C_ab = sum w X_a* X_b, retaining every c,v transition separately."""
    gap = np.asarray(gap, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(gap <= 0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Positive occupied-empty gaps and normalized weights required")
    c = np.einsum("k,kacv,kbcv->abkcv", weights, x.conj(), x)
    return gap.ravel(), c.reshape(2, 2, -1)


def kernel_any(d, gap, c, z):
    z = complex(z)
    return (np.asarray(d, complex)
            + np.einsum("abp,p->ab", c, 1 / (z - gap))
            - np.einsum("abp,p->ab", c.conj(), 1 / (z + gap)))


def lorentzian_S(gap, c, omega, eta, *, greater):
    omega = float(omega)
    eta = float(eta)
    if greater:
        return np.einsum("abp,p->ab", c, 2.0 * eta / ((omega - gap) ** 2 + eta ** 2))
    return np.einsum("abp,p->ab", c.conj(), 2.0 * eta / ((omega + gap) ** 2 + eta ** 2))


def h2_weight_interpolator(omega0, tp):
    """Same H2 |phi|^2 as q8_m0_k1_packet.packet_on_grid, without importing it."""
    omega_c = 2.0 * float(omega0)
    dw = (2.0 * np.pi / float(tp)) / 64.0
    wmax = 12.0 * float(omega0)
    omega = np.arange(0.0, wmax + 0.5 * dw, dw)
    raw = float(tp) * window_fourier(omega_c - omega, float(tp))
    mass = float(np.trapezoid(np.abs(raw) ** 2, omega))
    phi = raw / np.sqrt(mass)
    wgt = np.abs(phi) ** 2

    def weight(omega_q):
        q = np.asarray(omega_q, float)
        out = np.interp(q, omega, wgt, left=0.0, right=0.0)
        out = np.where(q > 0.0, out, 0.0)
        return out

    return {
        "omega": omega,
        "weight": wgt,
        "omega_c": omega_c,
        "l2": float(np.trapezoid(wgt, omega)),
        "fn": weight,
        "not_photon_number": True,
        "not_a_connected_photon_mode": True,
        "role": "smooth_frequency_window_only",
    }


def histogram_trace(gap, c, edges):
    tr = np.real(np.einsum("aap->p", c))
    mass, _ = np.histogram(gap, bins=edges, weights=tr)
    return mass


def assemble(geo, name, nk, shift, h2):
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
    min_gap = float(np.min(gap))
    max_gap = float(np.max(gap))
    equal = np.einsum("abp->ab", c)
    w_h2 = h2["fn"](gap)
    h2_unb = np.einsum("abp,p->ab", c, w_h2)
    edges = np.linspace(0.0, HIST_MAX, N_BINS + 1)
    hist = histogram_trace(gap, c, edges)
    identity = []
    probes = [0.0, 0.5 * min_gap, 0.9 * min_gap, 1.1 * min_gap, float(h2["omega_c"]), 1.0]
    for w in probes:
        sg = lorentzian_S(gap, c, w, ETA_IDENTITY, greater=True)
        sl = lorentzian_S(gap, c, w, ETA_IDENTITY, greater=False)
        kr = kernel_any(d, gap, c, w + 1j * ETA_IDENTITY)
        ka = kernel_any(d, gap, c, w - 1j * ETA_IDENTITY)
        lhs = kr - ka
        rhs = -1j * (sg - sl)
        identity.append({
            "omega": float(w),
            "below_gap": bool(abs(w) < min_gap),
            "identity_maxabs": float(np.max(np.abs(lhs - rhs))),
            "S_greater_trace": float(np.trace(sg).real),
        })
    shape_omega = np.unique(np.concatenate((
        np.array([0.0, 0.5 * min_gap, 0.9 * min_gap], float),
        np.linspace(min_gap, min(2.0, max_gap), N_SHAPE),
        np.array([float(h2["omega_c"])], float),
    )))
    eta_rows = []
    for eta in ETA_SHAPE:
        traces = []
        for w in shape_omega:
            sg = lorentzian_S(gap, c, float(w), eta, greater=True)
            traces.append(float(np.trace(sg).real))
        eta_rows.append({
            "eta": float(eta),
            "omega": shape_omega.tolist(),
            "S_greater_trace": traces,
            "below_gap_max_trace": max(
                tr for w, tr in zip(shape_omega, traces) if abs(w) < min_gap
            ),
        })
    below = int(np.count_nonzero(gap < min_gap - 1e-18))
    return {
        "grid": name,
        "n_k": int(len(kc)),
        "shift": shift,
        "n_transitions": int(gap.size),
        "min_direct_gap": min_gap,
        "max_direct_gap": max_gap,
        "n_transitions_strictly_below_min_gap": below,
        "equal_time_C_greater": pack_c(equal),
        "h2_unbroadened_weighted_C": pack_c(h2_unb),
        "histogram_edges": edges.tolist(),
        "histogram_trace_C": hist.tolist(),
        "histogram_mass_sum": float(np.sum(hist)),
        "identity_eta": ETA_IDENTITY,
        "spectral_identity": identity,
        "identity_maxabs_all_probes": max(row["identity_maxabs"] for row in identity),
        "eta_packet_integrals": eta_rows,
        "weight_counted_once": WEIGHT_ONCE,
    }


def _eta_row(grid_rec, eta):
    return next(row for row in grid_rec["eta_packet_integrals"] if abs(row["eta"] - eta) < 1e-18)


def _eta_l1(a, b, eta):
    ta = np.asarray(_eta_row(a, eta)["S_greater_trace"], float)
    tb = np.asarray(_eta_row(b, eta)["S_greater_trace"], float)
    n = min(len(ta), len(tb))
    return float(np.sum(np.abs(ta[:n] - tb[:n])) / max(np.sum(tb[:n]), 1e-30))


def _eta_pair_l1(grid_rec, eta_a, eta_b):
    ta = np.asarray(_eta_row(grid_rec, eta_a)["S_greater_trace"], float)
    tb = np.asarray(_eta_row(grid_rec, eta_b)["S_greater_trace"], float)
    return float(np.sum(np.abs(ta - tb)) / max(np.sum(tb), 1e-30))


def relative_trace(a, b):
    ta = float(a["trace_real"])
    tb = float(b["trace_real"])
    return abs(ta - tb) / max(abs(tb), 1e-30)


def compare(rows):
    by = {row["grid"]: row for row in rows}
    g96, g128, g192, h192 = (by[n] for n in (
        "Gamma_96x96", "Gamma_128x128", "Gamma_192x192", "halfshift_192x192"
    ))
    hist_l1 = {}
    for a, b, key in (
        (g96, g128, "96_to_128"),
        (g128, g192, "128_to_192"),
        (g192, h192, "192_to_halfshift"),
    ):
        ha = np.asarray(a["histogram_trace_C"], float)
        hb = np.asarray(b["histogram_trace_C"], float)
        n = min(len(ha), len(hb))
        hist_l1[key] = float(np.sum(np.abs(ha[:n] - hb[:n])) / max(np.sum(hb[:n]), 1e-30))
    return {
        "equal_time_trace_rel_96_to_192": relative_trace(
            g96["equal_time_C_greater"], g192["equal_time_C_greater"]
        ),
        "equal_time_trace_rel_128_to_192": relative_trace(
            g128["equal_time_C_greater"], g192["equal_time_C_greater"]
        ),
        "equal_time_trace_rel_192_to_halfshift": relative_trace(
            g192["equal_time_C_greater"], h192["equal_time_C_greater"]
        ),
        "h2_unbroadened_trace_rel_96_to_192": relative_trace(
            g96["h2_unbroadened_weighted_C"], g192["h2_unbroadened_weighted_C"]
        ),
        "h2_unbroadened_trace_rel_128_to_192": relative_trace(
            g128["h2_unbroadened_weighted_C"], g192["h2_unbroadened_weighted_C"]
        ),
        "h2_unbroadened_trace_rel_192_to_halfshift": relative_trace(
            g192["h2_unbroadened_weighted_C"], h192["h2_unbroadened_weighted_C"]
        ),
        "histogram_l1_rel": hist_l1,
        "fixed_eta_1e-3_trace_l1_128_to_192": _eta_l1(g128, g192, 1e-3),
        "fixed_eta_1e-3_trace_l1_192_to_halfshift": _eta_l1(g192, h192, 1e-3),
        "eta_1e-3_vs_1e-4_on_192_l1": _eta_pair_l1(g192, 1e-3, 1e-4),
        "algebra_is_not_continuum": True,
    }


def verdict(cmp_row, ident_max):
    algebra = bool(ident_max < 1e-9)
    mass_ok = bool(cmp_row["equal_time_trace_rel_128_to_192"] < 1e-3
                   and cmp_row["equal_time_trace_rel_192_to_halfshift"] < 1e-3)
    h2_ok = bool(cmp_row["h2_unbroadened_trace_rel_128_to_192"] < 5e-3
                 and cmp_row["h2_unbroadened_trace_rel_192_to_halfshift"] < 5e-3)
    hist_ok = bool(cmp_row["histogram_l1_rel"]["128_to_192"] < 5e-2
                   and cmp_row["histogram_l1_rel"]["192_to_halfshift"] < 5e-2)
    eta_grid_ok = bool(cmp_row["fixed_eta_1e-3_trace_l1_128_to_192"] < 5e-2
                       and cmp_row["fixed_eta_1e-3_trace_l1_192_to_halfshift"] < 5e-2)
    return {
        "algebraic_identity": "pass" if algebra else "fail",
        "equal_time_mass_grid": "stable_finite" if mass_ok else "not_stable",
        "h2_window_unbroadened": "stable_finite" if h2_ok else "not_stable",
        "histogram_shape": "stable_finite" if hist_ok else "not_stable",
        "fixed_eta_shape_grid": "stable_finite" if eta_grid_ok else "not_stable",
        "eta_narrowing_on_192_is_not_continuum": True,
        "continuum_spectrum_claimed": False,
        "photon_mode_connected": False,
        "full_incoming_channel_attached": False,
    }


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    prot = p1_a2_protocol(0.0)
    h2 = h2_weight_interpolator(prot.omega0, prot.t_pulse)
    h2_meta = {k: v for k, v in h2.items() if k != "fn"}
    h2_meta["omega"] = [float(h2_meta["omega"][0]), float(h2_meta["omega"][-1]),
                        int(len(h2_meta["omega"]))]
    h2_meta["weight_max"] = float(np.max(h2["weight"]))
    del h2_meta["weight"]
    rows = []
    for name, nk, shift in GRIDS:
        rec = assemble(geo, name, nk, shift, h2)
        rows.append(rec)
        print(name, "min_gap", rec["min_direct_gap"],
              "identity", rec["identity_maxabs_all_probes"],
              "C_tr", rec["equal_time_C_greater"]["trace_real"],
              "H2_tr", rec["h2_unbroadened_weighted_C"]["trace_real"],
              flush=True)
    cmp_row = compare(rows)
    ident_max = max(r["identity_maxabs_all_probes"] for r in rows)
    status_flags = verdict(cmp_row, ident_max)
    payload = {
        "status": "zero_pump_electronic_spectrum_finite_grid_diagnosis_only",
        "definition": {
            "C_greater": "sum_kcv w_k X_a* X_b at omega=+Delta",
            "S_lorentzian": "2 eta / ((omega-Delta)^2+eta^2) times C, identity only",
            "K": "stage1ai/ak kernel_any, original D kept",
            "identity": "K(w+i eta)-K(w-i eta)=-i(S>-S<)",
            "weight_once": WEIGHT_ONCE,
            "h2_window": "fixed |phi|^2 from K1 H2 packet; electronic source diagnosis only",
        },
        "omega0": OMEGA0_OVER_TH,
        "h2_window": h2_meta,
        "grids": rows,
        "grid_eta_comparison": cmp_row,
        "separate_status": status_flags,
        "identity_maxabs_all_grids": ident_max,
        "P1_not_run": True,
        "photon_statistics_not_run": True,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_cjj.py",
                REPO / "tools/b0_small_model/q8_m0_response.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ak_intrinsic_noise.py",
                REPO / "tools/b0_small_model/q8_m0_k1_packet.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "electronic_spectrum.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "separate_status": status_flags,
        "identity_maxabs_all_grids": ident_max,
        "comparison": cmp_row,
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
