#!/usr/bin/env python3
"""Zero-cost P-conjugate r-precision audit (no production TB overwrite).

Builds float64 -N r from +N; compares to production -N (E18.8); forms ES25.17
round-trip of exact r; decomposes ΔD_a(k) and evaluates Δv_a = -i[ΔD_a, H].

If ΔD ≈ q_a I then [ΔD,H]=0 and write-format r error cannot affect propagation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from construct_p_conjugate_tb import (
    build_minus_hr,
    build_minus_rr,
    build_permutation,
    build_wf_metadata,
    neg_R_index,
    parse_final_centers,
    parse_lattice_ang,
    read_header_text,
    read_tb_blocks,
)

ANG_TO_BOHR = 1.0 / 0.529177210903


def round_fmt(z: complex, fmt: str) -> complex:
    if fmt == "E18.8":
        return float(f"{z.real:18.8E}") + 1j * float(f"{z.imag:18.8E}")
    if fmt == "ES25.17":
        return float(f"{z.real:25.17E}") + 1j * float(f"{z.imag:25.17E}")
    raise ValueError(fmt)


def roundtrip_array(arr: np.ndarray, fmt: str, sample: int | None = None) -> np.ndarray:
    """Format-roundtrip; optional random sample for speed on huge arrays."""
    if sample is None or sample >= arr.size:
        out = np.empty_like(arr)
        flat_i = arr.ravel()
        flat_o = out.ravel()
        for i, c in enumerate(flat_i):
            flat_o[i] = round_fmt(complex(c), fmt)
        return out
    # sample-only estimate of max truncation (not for Fourier)
    rng = np.random.default_rng(0)
    idx = rng.choice(arr.size, size=sample, replace=False)
    flat = arr.ravel()
    err = [abs(flat[i] - round_fmt(complex(flat[i]), fmt)) for i in idx]
    return np.array(err, dtype=float)


def fourier_op(irvec, ndegen, mat_r, kfrac):
    """mat_r: (nr, nw, nw) or (nr, nw, nw, 3)."""
    phase = np.exp(2j * np.pi * (irvec @ np.asarray(kfrac, dtype=float))) / ndegen
    return np.tensordot(phase, mat_r, axes=(0, 0))


def hermitize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.conj().T)


def decompose_delta(delta: np.ndarray) -> dict:
    """Δ = q I + diag_TL + off; returns max-abs norms."""
    nw = delta.shape[0]
    tr = np.trace(delta)
    q = tr / float(nw)
    ident = q * np.eye(nw, dtype=np.complex128)
    diag = np.diag(np.diag(delta))
    diag_tl = diag - ident
    off = delta - diag
    return {
        "q": complex(q),
        "max_abs_qI": float(np.max(np.abs(ident))),
        "max_abs_diag_TL": float(np.max(np.abs(diag_tl))),
        "max_abs_off": float(np.max(np.abs(off))),
        "max_abs_total": float(np.max(np.abs(delta))),
        "frac_qI": float(np.max(np.abs(ident)) / max(np.max(np.abs(delta)), 1e-300)),
    }


def write_hybrid_tb(
    prod_minus: Path,
    out_path: Path,
    irvec: np.ndarray,
    h_prod: np.ndarray,
    r_hi: np.ndarray,
    header_lines: list[str],
) -> None:
    """Write -N TB: H from production arrays with E18.8; r with ES25.17."""
    # Reuse writer but patch formats: temporarily monkey via local copy of write logic
    nwann = h_prod.shape[1]
    n_grid, m_grid = np.meshgrid(
        np.arange(1, nwann + 1), np.arange(1, nwann + 1), indexing="ij"
    )
    m_flat = m_grid.reshape(-1)
    n_flat = n_grid.reshape(-1)
    order = np.lexsort((m_flat, n_flat))
    m_seq = m_flat[order]
    n_seq = n_flat[order]

    if "v2" in out_path.name.lower():
        comment = (
            " P-conjugate -N TB-v2: H=E18.8 as production; r=ES25.17 strict P-conjugate "
            f"(from {prod_minus.name}; official freeze, does not overwrite production)"
        )
    else:
        comment = (
            " P-conjugate -N TEMP: H=E18.8 as production; r=ES25.17 audit only "
            f"(from {prod_minus.name}; DO NOT use as production TB)"
        )
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(comment + "\n")
        fh.write("\n".join(header_lines[1:]) + "\n")
        for ir in range(irvec.shape[0]):
            fh.write("\n")
            fh.write(f"{irvec[ir,0]:5d}{irvec[ir,1]:5d}{irvec[ir,2]:5d}\n")
            h = h_prod[ir]
            re = h.real.T.reshape(-1)
            im = h.imag.T.reshape(-1)
            lines = [
                f"{m_seq[i]:5d}{n_seq[i]:5d}{re[i]:18.8E}{im[i]:18.8E}"
                for i in range(len(m_seq))
            ]
            fh.write("\n".join(lines) + "\n")
        for ir in range(irvec.shape[0]):
            fh.write("\n")
            fh.write(f"{irvec[ir,0]:5d}{irvec[ir,1]:5d}{irvec[ir,2]:5d}\n")
            rx = r_hi[ir, :, :, 0].T.reshape(-1)
            ry = r_hi[ir, :, :, 1].T.reshape(-1)
            rz = r_hi[ir, :, :, 2].T.reshape(-1)
            lines = [
                f"{m_seq[i]:5d}{n_seq[i]:5d}"
                f"{rx[i].real:25.17E}{rx[i].imag:25.17E}"
                f"{ry[i].real:25.17E}{ry[i].imag:25.17E}"
                f"{rz[i].real:25.17E}{rz[i].imag:25.17E}"
                for i in range(len(m_seq))
            ]
            fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plus-tb", type=Path, required=True)
    ap.add_argument("--minus-tb-prod", type=Path, required=True)
    ap.add_argument("--plus-win", type=Path, required=True)
    ap.add_argument("--plus-wout", type=Path, required=True)
    ap.add_argument("--center", default="0.499998583,0.333333067,0.372004766")
    ap.add_argument("--nk", type=int, default=20)
    ap.add_argument("--write-temp-tb", type=Path, default=None,
                    help="Optional hybrid TB path (H=E18.8 prod, r=ES25.17); not production")
    ap.add_argument("--skip-k-audit", action="store_true",
                    help="Skip full-k ΔD/Δv (use when only writing hybrid TB after a prior audit)")
    ap.add_argument("--r-sample-roundtrip", type=int, default=50000,
                    help="elements to sample for format truncation floor estimate")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    center = np.array([float(x) for x in args.center.split(",")], dtype=float)
    lattice = parse_lattice_ang(args.plus_win)
    c_cart = center @ lattice
    centers_cart = parse_final_centers(args.plus_wout)
    centers_frac = (centers_cart @ np.linalg.inv(lattice)) % 1.0
    nwann = centers_frac.shape[0]
    meta = build_wf_metadata(nwann)
    perm, pdiag = build_permutation(centers_frac, meta, center)
    if pdiag["duplicate_targets"] or pdiag["non_involutive"]:
        raise SystemExit("permutation not involution")

    print("[1] reading +N ...", flush=True)
    ir_p, nd_p, h_p, r_p = read_tb_blocks(args.plus_tb, True)
    print("[2] reading production -N ...", flush=True)
    ir_m, nd_m, h_m_prod, r_m_prod = read_tb_blocks(args.minus_tb_prod, True)
    if not np.array_equal(ir_p, ir_m):
        raise SystemExit("irvec mismatch")
    neg_idx = neg_R_index(ir_p)

    print("[3] construct float64 -N H/r ...", flush=True)
    h_exact = build_minus_hr(h_p, perm, meta["parity"], neg_idx)
    r_exact = build_minus_rr(r_p, perm, meta["parity"], neg_idx, ir_p, c_cart)

    dH = float(np.max(np.abs(h_m_prod - h_exact)))
    dR_prod_vs_exact = float(np.max(np.abs(r_m_prod - r_exact)))

    # Format floors (sampled) — E18.8 vs ES25.17 truncation of exact r
    print("[4] format truncation floors (sampled) ...", flush=True)
    err_e188 = roundtrip_array(r_exact, "E18.8", sample=args.r_sample_roundtrip)
    err_es25 = roundtrip_array(r_exact, "ES25.17", sample=args.r_sample_roundtrip)
    fmt_floor = {
        "E18.8_max_abs_sample": float(np.max(err_e188)),
        "ES25.17_max_abs_sample": float(np.max(err_es25)),
        "n_sample": int(args.r_sample_roundtrip),
    }

    # Physical Δr for upgrading beyond production: float64 exact vs production disk.
    # ES25.17 ≈ float64 for Angstrom Wannier matrix elements, so this is the
    # discriminant without rewriting a 1GB TB unless --write-temp-tb is set.
    print("[5] Δr = r_exact - r_prod ...", flush=True)
    delta_r = r_exact - r_m_prod
    max_abs_delta_r = float(np.max(np.abs(delta_r)))
    r_hi = r_exact

    hybrid_reread = None
    if args.write_temp_tb is not None:
        print(f"[5b] writing hybrid temp TB -> {args.write_temp_tb}", flush=True)
        header = read_header_text(args.minus_tb_prod, ir_m.shape[0])
        # Store ES25.17 roundtrip for written r (vectorized via sample check only;
        # write uses ES25.17 formatting of exact).
        args.write_temp_tb.parent.mkdir(parents=True, exist_ok=True)
        write_hybrid_tb(
            args.minus_tb_prod, args.write_temp_tb, ir_m, h_m_prod, r_exact, header
        )
        print("[5c] re-read hybrid r ...", flush=True)
        _, _, h_tmp, r_tmp = read_tb_blocks(args.write_temp_tb, True)
        dH_tmp = float(np.max(np.abs(h_tmp - h_m_prod)))
        dR_tmp = float(np.max(np.abs(r_tmp - r_exact)))
        hybrid_reread = {
            "max_abs_dH_vs_prod": dH_tmp,
            "max_abs_dR_vs_exact": dR_tmp,
        }
        delta_r = r_tmp - r_m_prod
        max_abs_delta_r = float(np.max(np.abs(delta_r)))
        r_hi = r_tmp

    print("[6] full k20 ΔD / Δv ...", flush=True)
    # Use production H for commutator (what solver sees)
    max_qI = max_tl = max_off = max_dD = 0.0
    max_dv = max_v = 0.0
    worst_k = None
    per_cart = []
    dv_rel = 0.0
    if args.skip_k_audit:
        print("[6] skipped (--skip-k-audit)", flush=True)
        decision = "WRITE_ONLY_NO_K_AUDIT"
        action = "Hybrid TB write path; see prior AUDIT_PCONJ_R_PRECISION.json for decision."
        exclude_e188 = False
        need_hi_traj = False
    else:
        nk = args.nk
        for a in range(3):
            max_qI_a = max_tl_a = max_off_a = max_dD_a = 0.0
            max_dv_a = max_v_a = 0.0
            for ikx in range(nk):
                for iky in range(nk):
                    kf = (ikx / nk, iky / nk, 0.0)
                    Hk = hermitize(fourier_op(ir_m, nd_m, h_m_prod, kf))
                    Dk_prod = fourier_op(ir_m, nd_m, r_m_prod[:, :, :, a], kf)
                    Dk_hi = fourier_op(ir_m, nd_m, r_hi[:, :, :, a], kf)
                    dD = Dk_hi - Dk_prod
                    dec = decompose_delta(dD)
                    max_qI_a = max(max_qI_a, dec["max_abs_qI"])
                    max_tl_a = max(max_tl_a, dec["max_abs_diag_TL"])
                    max_off_a = max(max_off_a, dec["max_abs_off"])
                    max_dD_a = max(max_dD_a, dec["max_abs_total"])

                    v_prod = -1j * (Dk_prod @ Hk - Hk @ Dk_prod)
                    dv = -1j * (dD @ Hk - Hk @ dD)
                    nv = float(np.max(np.abs(v_prod)))
                    ndv = float(np.max(np.abs(dv)))
                    max_v_a = max(max_v_a, nv)
                    if ndv >= max_dv_a:
                        max_dv_a = ndv
                        if a == 0:
                            worst_k = {"kfrac": kf, "cart": a, "max_abs_dv": ndv}
            max_qI = max(max_qI, max_qI_a)
            max_tl = max(max_tl, max_tl_a)
            max_off = max(max_off, max_off_a)
            max_dD = max(max_dD, max_dD_a)
            max_dv = max(max_dv, max_dv_a)
            max_v = max(max_v, max_v_a)
            per_cart.append(
                {
                    "cart": "xyz"[a],
                    "max_abs_qI": max_qI_a,
                    "max_abs_diag_TL": max_tl_a,
                    "max_abs_off": max_off_a,
                    "max_abs_dD": max_dD_a,
                    "max_abs_dv": max_dv_a,
                    "max_abs_v_prod": max_v_a,
                    "max_rel_dv": max_dv_a / max(max_v_a, 1e-300),
                }
            )

        # Decision thresholds
        # "near machine precision" for effective operators after identity removal
        eff_max = max(max_tl, max_off)
        dv_rel = max_dv / max(max_v, 1e-300)
        # Exclude E18.8 if effective ΔD and Δv are tiny vs production scales
        exclude_e188 = (eff_max <= 1e-14) or (dv_rel <= 1e-14 and max_dv <= 1e-14)
        # Physical residual warranting trajectories: ~1e-9 effective
        need_hi_traj = (eff_max >= 1e-10) or (dv_rel >= 1e-10)

        if exclude_e188:
            decision = "EXCLUDE_E18.8_r_write_as_CEP_driver"
            action = "Do not run high-precision TB trajectories; proceed to GH5 freeze checklist."
        elif need_hi_traj:
            decision = "RUN_TWO_TRAJECTORIES_WITH_HI_R_TB"
            action = (
                "Effective traceless/off or Δv ~>=1e-10: run same antipode pair on part_2 "
                "with hybrid ES25.17-r TB."
            )
        else:
            decision = "BORDERLINE_REVIEW"
            action = "Intermediate residual; review numbers before GH5 or hi-r trajectories."

    report = {
        "gate_note": (
            "T2 branch closed. This audit asks whether E18.8 r-write truncation "
            "yields a physically active ΔD/Δv (not merely qI)."
        ),
        "cep_labels": {
            "CEP_STRICT_V2": "FAIL",
            "CEP_RELATIVE_COVARIANCE": "PASS",
            "CEP_PHYSICS_CONTAMINATION": "PASS",
            "CEP_MACHINE_EXACTNESS": "UNRESOLVED",
        },
        "production_minus_tb": str(args.minus_tb_prod),
        "dH_prod_vs_exact": dH,
        "dR_prod_vs_exact": dR_prod_vs_exact,
        "format_truncation_floor_Ang": fmt_floor,
        "max_abs_delta_r_hi_minus_prod_Ang": max_abs_delta_r,
        "hybrid_temp_tb": str(args.write_temp_tb) if args.write_temp_tb else None,
        "hybrid_reread": hybrid_reread,
        "k20_decomposition_Ang": {
            "max_abs_qI": max_qI,
            "max_abs_diag_TL": max_tl,
            "max_abs_off": max_off,
            "max_abs_dD": max_dD,
            "per_cartesian": per_cart,
        },
        "k20_velocity": {
            "max_abs_dv": max_dv,
            "max_abs_v_prod": max_v,
            "max_rel_dv": dv_rel,
            "worst_k": worst_k,
            "note": "Δv=-i[ΔD,H] with H,D from TB (D in Angstrom as stored)",
        },
        "decision": decision,
        "action": action,
        "exclude_e188": exclude_e188,
        "need_hi_traj": need_hi_traj,
        "gh5": "HOLD_until_this_audit_acted_on",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "max_abs_diag_TL": max_tl,
        "max_abs_off": max_off,
        "max_rel_dv": dv_rel,
        "max_abs_dv": max_dv,
        "dR_prod_vs_exact": dR_prod_vs_exact,
        "report": str(args.report),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
