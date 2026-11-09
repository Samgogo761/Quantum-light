#!/usr/bin/env python3
"""One finite H2-window BZ quadrature for the zero-pump electronic source.

M_ab = sum_{kcv} w_k X_a* X_b F(Delta_kcv). Not photon number, not P1,
not a full incoming-channel attach. Old stage1ao JSON is not overwritten.
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

REPO = Path(__file__).resolve().parents[2]
MU = -0.009010620372198276
OUT = REPO / "results/q_line/stage1ap_h2_window_quadrature_20261004"
OLD_JSON = "results/q_line/stage1ao_electronic_spectrum_20261004/electronic_spectrum.json"
TARGET = 0.005
CHUNK = 8192
N_CELL = 32
ETA_ALIGN = 1e-3
OMEGA_ALIGN = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_c(m):
    m = np.asarray(m, complex)
    herm = 0.5 * (m + m.conj().T)
    evals = np.linalg.eigvalsh(herm)
    return {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "maxabs": float(np.max(np.abs(m))),
        "trace_real": float(np.trace(m).real),
        "frobenius": float(np.linalg.norm(m)),
        "hermitian_eigenvalues": evals.tolist(),
        "anti_hermitian_maxabs": float(np.max(np.abs(m - herm))),
        "min_hermitian_eig": float(evals[0]),
    }


def frobenius(a, b=None):
    a = np.asarray(a, complex)
    if b is None:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - np.asarray(b, complex)))


def rel_frob(a, b):
    den = max(frobenius(b), 1e-30)
    return frobenius(a, b) / den


def component_abs(a, b):
    d = np.asarray(a, complex) - np.asarray(b, complex)
    return {
        "real": np.real(d).tolist(),
        "imag": np.imag(d).tolist(),
        "maxabs": float(np.max(np.abs(d))),
    }


def h2_packet(omega0, tp):
    omega_c = 2.0 * float(omega0)
    dw = (2.0 * np.pi / float(tp)) / 64.0
    wmax = 12.0 * float(omega0)
    omega = np.arange(0.0, wmax + 0.5 * dw, dw)
    raw = float(tp) * window_fourier(omega_c - omega, float(tp))
    mass = float(np.trapezoid(np.abs(raw) ** 2, omega))
    wgt = np.abs(raw) ** 2 / mass
    omega_ext = np.arange(0.0, 2.0 + 0.5 * dw, dw)
    raw_ext = float(tp) * window_fourier(omega_c - omega_ext, float(tp))
    mass_ext = float(np.trapezoid(np.abs(raw_ext) ** 2, omega_ext))
    samples = {}
    for w in (0.05, 0.093, 0.10, 0.12, 0.20, 0.30, 0.45, 0.60, 0.80, 1.0):
        rd = float(tp) * window_fourier(omega_c - w, float(tp))
        samples[str(w)] = {
            "F_direct": float(np.abs(rd) ** 2 / mass),
            "F_interp": float(np.interp(w, omega, wgt, left=0.0, right=0.0)),
        }

    def F_interp(q):
        q = np.asarray(q, float)
        out = np.interp(q, omega, wgt, left=0.0, right=0.0)
        return np.where(q > 0.0, out, 0.0)

    def F_direct(q):
        q = np.asarray(q, float)
        rd = float(tp) * window_fourier(omega_c - q, float(tp))
        out = np.abs(rd) ** 2 / mass
        return np.where(q > 0.0, out, 0.0)

    return {
        "omega_c": omega_c,
        "mass_0_to_0p6": mass,
        "mass_0_to_2": mass_ext,
        "mass_rel_tail": abs(mass_ext - mass) / mass,
        "omega_grid": omega,
        "wgt": wgt,
        "F_interp": F_interp,
        "F_direct": F_direct,
        "samples": samples,
        "not_photon_number": True,
        "not_a_physical_cutoff": True,
    }


def evaluate(geo, kf, weights, Ffn, *, want_eta=False):
    kf = np.asarray(kf, float)
    weights = np.asarray(weights, float)
    n = len(kf)
    M = np.zeros((2, 2), complex)
    min_gap = np.empty(n)
    contrib = np.empty(n)
    s_eta = np.zeros((2, 2), complex) if want_eta else None
    for i0 in range(0, n, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, n))
        kc = geo.k_cart(kf[sl])
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
        x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
        gap = en[:, 4:, None] - en[:, None, :4]
        f = Ffn(gap)
        mk = np.einsum("kacv,kbcv,kcv->kab", np.conj(x), x, f)
        w = weights[sl]
        M += np.einsum("k,kab->ab", w, mk)
        min_gap[sl] = gap.reshape(len(kc), -1).min(axis=1)
        contrib[sl] = w * np.real(mk[:, 0, 0] + mk[:, 1, 1])
        if want_eta:
            lor = 2.0 * ETA_ALIGN / ((OMEGA_ALIGN - gap) ** 2 + ETA_ALIGN ** 2)
            ck = np.einsum("kacv,kbcv,kcv->kab", np.conj(x), x, lor)
            s_eta += np.einsum("k,kab->ab", w, ck)
    out = {"M": M, "min_gap": min_gap, "contrib": contrib,
           "weight_sum": float(np.sum(weights)), "n_k": int(n)}
    if want_eta:
        out["S_eta_omega_0p1"] = s_eta
    return out


def inversion_ok(kf, weights):
    """Integer-lattice partner test. Half-integers are snapped by doubling.

    Uniform and half-shift hybrid grids pair. A Γ hybrid child can invert
    into a neighbouring parent; one-pass close_mask does not close that set.
    Do not treat a leftover miss count as a float-key false positive.
    """
    kf = np.mod(np.asarray(kf, float), 1.0)
    w = np.asarray(weights, float)
    coords = np.unique(np.concatenate((kf[:, 0], kf[:, 1])))
    du = np.diff(np.sort(coords))
    du = du[du > 1e-14]
    scale = float(np.min(du)) if du.size else 1.0
    period = int(round(1.0 / scale))
    qtest = kf / scale
    if float(np.max(np.abs(qtest - np.round(qtest)))) > 0.1:
        scale *= 0.5
        period *= 2
        qtest = kf / scale
    q = np.mod(np.round(qtest).astype(np.int64), period)
    table = {}
    for i, (a, b) in enumerate(q):
        table[(int(a), int(b))] = i
    miss = 0
    wdiff = 0.0
    inv = np.mod(-q, period)
    for i, (a, b) in enumerate(inv):
        j = table.get((int(a), int(b)))
        if j is None:
            miss += 1
            continue
        wdiff = max(wdiff, abs(w[i] - w[j]))
    return {
        "missing_partners": int(miss),
        "max_weight_mismatch": float(wdiff),
        "lattice_scale": scale,
        "lattice_period": int(period),
        "method": "integer_grid_rounding_halfinteger_safe",
    }


def partner_ij(i, j, nk, shift):
    if abs(shift) < 1e-15:
        return (-i) % nk, (-j) % nk
    return (nk - 1 - i) % nk, (nk - 1 - j) % nk


def close_mask(mask, nk, shift):
    """Keep the hybrid point set inversion-paired.

    shift=1/2 subpoints invert inside the area-inverse parent cell.
    shift=0 corner subpoints also need the neighbouring parent cells.
    """
    out = np.array(mask, bool, copy=True)
    for i in range(nk):
        for j in range(nk):
            if not mask[i, j]:
                continue
            if abs(shift) < 1e-15:
                partners = (
                    ((-i) % nk, (-j) % nk),
                    ((-i - 1) % nk, (-j) % nk),
                    ((-i) % nk, (-j - 1) % nk),
                    ((-i - 1) % nk, (-j - 1) % nk),
                )
            else:
                partners = (partner_ij(i, j, nk, shift),)
            for pi, pj in partners:
                out[pi, pj] = True
    return out


def hybrid_grid(nk0, shift, mask, r):
    mask = close_mask(mask, nk0, shift)
    kf = []
    w = []
    wc = 1.0 / (nk0 * nk0)
    n_ref = 0
    for i in range(nk0):
        for j in range(nk0):
            if mask[i, j]:
                n_ref += 1
                for a in range(r):
                    for b in range(r):
                        kf.append([
                            (i + (a + shift) / r) / nk0,
                            (j + (b + shift) / r) / nk0,
                        ])
                        w.append(wc / (r * r))
            else:
                kf.append([(i + shift) / nk0, (j + shift) / nk0])
                w.append(wc)
    return np.asarray(kf, float), np.asarray(w, float), mask, n_ref


def cell_id(kf, ncell=N_CELL):
    ij = np.floor(np.mod(kf, 1.0) * ncell).astype(int) % ncell
    return ij[:, 0] * ncell + ij[:, 1]


def accumulate_cells(ids, contrib, m_k, weights, ncell=N_CELL):
    n = ncell * ncell
    mass = np.zeros(n)
    for i, c in zip(ids, contrib):
        mass[i] += c
    return mass


def uniform(geo, nk, shift, Ffn, **kw):
    kf, _, w = k_grid(geo, nk, shift=shift)
    rec = evaluate(geo, kf, w, Ffn, **kw)
    rec["kf"] = kf
    rec["weights"] = w
    rec["grid"] = f"{'halfshift' if shift else 'Gamma'}_{nk}x{nk}"
    rec["shift"] = float(shift)
    rec["nk"] = int(nk)
    return rec


def summarize(rec, extra=None):
    inv = inversion_ok(rec["kf"], rec["weights"])
    out = {
        "grid": rec["grid"],
        "n_k": rec["n_k"],
        "shift": rec["shift"],
        "weight_sum": rec["weight_sum"],
        "M": pack_c(rec["M"]),
        "inversion": inv,
        "contrib_sum": float(np.sum(rec["contrib"])),
        "min_direct_gap": float(np.min(rec["min_gap"])),
    }
    if "S_eta_omega_0p1" in rec:
        out["S_eta_omega_0p1"] = pack_c(rec["S_eta_omega_0p1"])
    if extra:
        out.update(extra)
    return out


def compare_pair(a, b, name):
    ma, mb = a["M"], b["M"]
    return {
        "pair": name,
        "trace_rel": abs(np.trace(ma - mb).real) / max(abs(np.trace(mb).real), 1e-30),
        "frobenius_rel": rel_frob(ma, mb),
        "component_abs": component_abs(ma, mb),
        "trace_a": float(np.trace(ma).real),
        "trace_b": float(np.trace(mb).real),
    }


def cell_table(rec, ncell=N_CELL):
    ids = cell_id(rec["kf"], ncell)
    mass = np.zeros(ncell * ncell)
    for i, c in zip(ids, rec["contrib"]):
        mass[i] += c
    return mass


def select_cells(mass_a, mass_b, *, ncell=N_CELL, cover=0.85, min_frac=0.002):
    diff = np.abs(mass_a - mass_b)
    total_diff = float(np.sum(diff))
    total_mass = float(max(np.sum(np.abs(mass_a)), 1e-30))
    order = np.argsort(-diff)
    chosen = np.zeros(ncell * ncell, bool)
    acc = 0.0
    for idx in order:
        if acc >= cover * total_diff and diff[idx] < min_frac * total_mass:
            break
        chosen[idx] = True
        acc += diff[idx]
    for idx, m in enumerate(mass_a):
        if abs(m) >= min_frac * total_mass:
            chosen[idx] = True
    closed = chosen.copy()
    nsq = ncell * ncell
    grown = chosen.copy()
    for idx in np.nonzero(chosen)[0]:
        cx, cy = divmod(int(idx), ncell)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                grown[((cx + di) % ncell) * ncell + (cy + dj) % ncell] = True
    closed = grown.copy()
    for idx in np.nonzero(grown)[0]:
        cx, cy = divmod(int(idx), ncell)
        closed[((-cx - 1) % ncell) * ncell + ((-cy - 1) % ncell)] = True
    del nsq
    rows = []
    for idx in np.nonzero(closed)[0]:
        cx, cy = divmod(int(idx), ncell)
        rows.append({
            "cell": [int(cx), int(cy)],
            "mass_a": float(mass_a[idx]),
            "mass_b": float(mass_b[idx]),
            "abs_diff": float(diff[idx]),
        })
    rows.sort(key=lambda r: -r["abs_diff"])
    return {
        "n_selected": int(np.count_nonzero(closed)),
        "n_cell": int(ncell * ncell),
        "cover_of_abs_diff": float(np.sum(diff[closed]) / max(total_diff, 1e-30)),
        "mass_fraction_selected": float(np.sum(np.abs(mass_a[closed])) / total_mass),
        "total_abs_diff": total_diff,
        "mask": closed.reshape(ncell, ncell),
        "top_cells": rows[:24],
        "all_selected_cells": [[int(i // ncell), int(i % ncell)] for i in np.nonzero(closed)[0]],
    }


def mask_192_from_cells(nk, ncell, cell_mask):
    out = np.zeros((nk, nk), bool)
    for i in range(nk):
        for j in range(nk):
            out[i, j] = cell_mask[int(i * ncell / nk), int(j * ncell / nk)]
    return out


def hybrid_from_cells(nk, shift, cell_mask, r):
    return hybrid_grid(nk, shift, mask_192_from_cells(nk, cell_mask.shape[0], cell_mask), r)


def estimate_points(nk, n_ref, r):
    n_coarse = nk * nk - n_ref
    return n_coarse + n_ref * r * r


def main() -> int:
    t0 = time.perf_counter()
    geo = build_geometry()
    prot = p1_a2_protocol(0.0)
    h2 = h2_packet(prot.omega0, prot.t_pulse)
    Fdir = h2["F_direct"]
    Fint = h2["F_interp"]

    print("cost_estimate_before_run", {
        "uniform_grids": [96, 128, 192, "half192"],
        "prev_192_seconds_scale": 1.5,
        "hybrid_unknown_until_cell_select": True,
        "local_only": True,
    }, flush=True)

    g128i = uniform(geo, 128, 0.0, Fint)
    g96 = uniform(geo, 96, 0.0, Fdir)
    g128 = uniform(geo, 128, 0.0, Fdir, want_eta=True)
    g192 = uniform(geo, 192, 0.0, Fdir, want_eta=True)
    h192 = uniform(geo, 192, 0.5, Fdir, want_eta=True)

    f_impl = {
        "interp_vs_direct_on_128": compare_pair(g128i, g128, "F_interp_vs_F_direct_Gamma128"),
        "mass_rel_tail_0_to_2_vs_0_to_0p6": h2["mass_rel_tail"],
        "sample_F": h2["samples"],
        "not_a_physical_cutoff": True,
    }

    mass_g = cell_table(g192)
    mass_h = cell_table(h192)
    chosen = select_cells(mass_g, mass_h)
    r = 4
    n_ref_est = int(chosen["n_selected"] * (192 // N_CELL) ** 2)
    n_hyb = estimate_points(192, n_ref_est, r)
    print("cell_select", {
        "n_selected_32x32": chosen["n_selected"],
        "n_refined_192_cells_est": n_ref_est,
        "hybrid_n_k_est": n_hyb,
        "seconds_est": 1.5 * n_hyb / (192 * 192),
        "cover_of_abs_diff": chosen["cover_of_abs_diff"],
    }, flush=True)

    kf_a, w_a, mask_a, n_ref_a = hybrid_from_cells(192, 0.0, chosen["mask"], r)
    kf_b, w_b, mask_b, n_ref_b = hybrid_from_cells(192, 0.5, chosen["mask"], r)
    hyb_g = evaluate(geo, kf_a, w_a, Fdir, want_eta=True)
    hyb_g.update({"kf": kf_a, "weights": w_a, "grid": "hybrid_Gamma192_r4_selected32",
                  "shift": 0.0, "nk": 192})
    hyb_h = evaluate(geo, kf_b, w_b, Fdir, want_eta=True)
    hyb_h.update({"kf": kf_b, "weights": w_b, "grid": "hybrid_halfshift192_r4_selected32",
                  "shift": 0.5, "nk": 192})

    comparisons = [
        compare_pair(g128, g192, "Gamma128_to_Gamma192"),
        compare_pair(g192, h192, "Gamma192_to_halfshift192"),
        compare_pair(hyb_g, hyb_h, "hybrid_Gamma_to_hybrid_halfshift"),
        compare_pair(hyb_g, g192, "hybrid_Gamma_to_Gamma192"),
        compare_pair(hyb_h, h192, "hybrid_halfshift_to_halfshift192"),
        compare_pair(g128i, g128, "F_interp_vs_direct_Gamma128"),
    ]
    hyb_rel = comparisons[2]["frobenius_rel"]
    hyb_tr = comparisons[2]["trace_rel"]
    passed = bool(hyb_rel < TARGET and hyb_tr < TARGET)

    eta_align = {
        "omega": OMEGA_ALIGN,
        "eta": ETA_ALIGN,
        "same_absolute_frequency": True,
        "trace": {
            "Gamma_128x128": float(np.trace(g128["S_eta_omega_0p1"]).real),
            "Gamma_192x192": float(np.trace(g192["S_eta_omega_0p1"]).real),
            "halfshift_192x192": float(np.trace(h192["S_eta_omega_0p1"]).real),
            "hybrid_Gamma": float(np.trace(hyb_g["S_eta_omega_0p1"]).real),
            "hybrid_halfshift": float(np.trace(hyb_h["S_eta_omega_0p1"]).real),
        },
    }
    eta_align["rel_128_to_192"] = abs(
        eta_align["trace"]["Gamma_128x128"] - eta_align["trace"]["Gamma_192x192"]
    ) / max(abs(eta_align["trace"]["Gamma_192x192"]), 1e-30)
    eta_align["rel_192_to_half"] = abs(
        eta_align["trace"]["Gamma_192x192"] - eta_align["trace"]["halfshift_192x192"]
    ) / max(abs(eta_align["trace"]["halfshift_192x192"]), 1e-30)
    eta_align["rel_hybrid"] = abs(
        eta_align["trace"]["hybrid_Gamma"] - eta_align["trace"]["hybrid_halfshift"]
    ) / max(abs(eta_align["trace"]["hybrid_halfshift"]), 1e-30)

    hyb_extra_a = {
        "n_refined_parent_cells": int(n_ref_a),
        "refine_factor": r,
        "parent_nk": 192,
        "selected_32x32_cells": chosen["n_selected"],
        "cover_of_abs_diff": chosen["cover_of_abs_diff"],
        "mass_fraction_selected": chosen["mass_fraction_selected"],
    }
    hyb_extra_b = dict(hyb_extra_a)
    hyb_extra_b["n_refined_parent_cells"] = int(n_ref_b)

    status = (
        "h2_electronic_source_window_finite_qualification"
        if passed else
        "h2_electronic_source_window_not_yet_stable"
    )
    payload = {
        "status": status,
        "all_checks_pass": passed,
        "target_relative": TARGET,
        "definition": {
            "M": "sum_kcv w_k X_a* X_b F(Delta_kcv)",
            "F": "H2 |phi|^2 from window_fourier, L2-normalized on the original 0..0.6 grid",
            "weight_once": "w_k only; no extra sqrt(w) on vertices",
            "not_photon_number": True,
            "not_photon_output": True,
            "not_full_ground": True,
            "not_incoming_attach": True,
        },
        "F_implementation": f_impl,
        "uniform_grids": [
            summarize(g96),
            summarize(g128),
            summarize(g192),
            summarize(h192),
        ],
        "hybrid_grids": [
            summarize(hyb_g, hyb_extra_a),
            summarize(hyb_h, hyb_extra_b),
        ],
        "comparisons": comparisons,
        "cell_partition": {
            "n_cell": N_CELL,
            "n_selected": chosen["n_selected"],
            "cover_of_abs_diff": chosen["cover_of_abs_diff"],
            "mass_fraction_selected": chosen["mass_fraction_selected"],
            "total_abs_diff": chosen["total_abs_diff"],
            "top_cells": chosen["top_cells"],
            "all_selected_cells": chosen["all_selected_cells"],
            "note": "Selected from Gamma192 vs halfshift192 cell masses; inversion-closed.",
        },
        "aligned_eta_omega_0p1": eta_align,
        "old_22pct_shape_l1": {
            "status": "misaligned_frequency_samples_do_not_use",
            "replacement": "aligned_eta_omega_0p1",
        },
        "qualification": {
            "h2_window_matrix_target_0p5pct": passed,
            "continuum_spectrum_claimed": False,
            "photon_output_claimed": False,
            "full_ground_claimed": False,
            "pointwise_spectrum_claimed": False,
        },
        "inversion_diagnostic": {
            "same_quadrature_not_a_second_physics_round": True,
            "float_key_false_positive_on_Gamma_hybrid": False,
            "integer_grid_rounding_used": True,
            "gamma_hybrid_missing_partners_are_real": True,
            "halfshift_hybrid_missing_partners": 0,
            "note": (
                "hybrid Gamma reports 1452 missing inversion partners under "
                "both float-key and integer-grid rounding (scale 1/768). "
                "A refined shift=0 child inverts into a neighbouring parent; "
                "one-pass close_mask does not close the set. Uniform grids "
                "and the half-shift hybrid are paired. This is a Gamma-hybrid "
                "point-set caveat, not a second physics round."
            ),
        },
        "old_json_not_overwritten": [OLD_JSON],
        "P1_not_run": True,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_cjj.py",
                REPO / "tools/b0_small_model/q8_m0_classical.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ao_electronic_spectrum_20261004.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h2_window_quadrature.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    table = []
    for row in chosen["top_cells"]:
        table.append(f"{row['cell'][0]:2d} {row['cell'][1]:2d}  "
                     f"{row['mass_a']:.6e}  {row['mass_b']:.6e}  {row['abs_diff']:.6e}")
    (OUT / "h2_cell_contributions.txt").write_text(
        "cx cy  mass_Gamma192  mass_half192  abs_diff\n" + "\n".join(table) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "passed": passed,
        "hybrid_frobenius_rel": hyb_rel,
        "hybrid_trace_rel": hyb_tr,
        "n_k_hybrid": hyb_g["n_k"],
        "n_refined": n_ref_a,
        "F_interp_vs_direct": f_impl["interp_vs_direct_on_128"]["frobenius_rel"],
        "eta_aligned": {
            "128_to_192": eta_align["rel_128_to_192"],
            "192_to_half": eta_align["rel_192_to_half"],
            "hybrid": eta_align["rel_hybrid"],
        },
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0 if passed else 0


if __name__ == "__main__":
    raise SystemExit(main())

