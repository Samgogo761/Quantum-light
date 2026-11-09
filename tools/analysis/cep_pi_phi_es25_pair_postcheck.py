#!/usr/bin/env python3
"""Post-check for ES25 CEP-π pair probe — CEP V2 metric definitions.

Fixes the previous bug where e_abs == e_rel == d / local_amp.

CEP V2:
  d_n     = || J_{+} + J_{-} ||
  A_n     = max over full GH3 reference nodes of ||J|| at harmonic n
  e_abs   = d_n / A_n
  e_rel   = d_n / max(||J_+||, ||J_-||)   (local; strong-node gate uses this)

Also:
  FFT linearity: F[W(J_++J_-)] = F[W J_+] + F[W J_-]
  Time metric:   A_t = max_t (||J_+||+||J_-||)/2
                 e_t,∞ = max_t ||J_++J_-|| / A_t
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_full112_gh3_candidates import (
    HARD_ORDERS,
    amp_xy,
    load_manifest,
    load_node_modes,
    pair_nodes_by_neg_alpha,
)
from hhg_fft_utils import harmonic_fft_index, omega0_from_wvl_nm, parse_input_nml

DIAG_ORDERS = (10,)
CEP_V2_ABS_ATOL = 1.0e-12
CEP_V2_REL_ATOL = 1.0e-3
CEP_V2_ETA = 1.0e-10
# Canonical schema: FFT linearity uses component-scale denominator, not the
# cancelled residual. Do not mix with pre-20260812 reports that omit
# fft_linearity_abs / use residual-as-den (those can look like ~1e-5).
SCHEMA_VERSION = "cep_v2_fft_component_den_20260812"


def scale_A_n(modes_p: dict, modes_m: dict, pairs: list[dict], order: int) -> float:
    """Match cep_pi_gate_v2.scale_A_n (avoid importing compare_v3 via cep_pi_gate_v2)."""
    A = 0.0
    for pr in pairs:
        ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
        for nid in {ia, ib}:
            A = max(A, amp_xy(modes_p[(nid, order)]["jx"], modes_p[(nid, order)]["jy"]))
            A = max(A, amp_xy(modes_m[(nid, order)]["jx"], modes_m[(nid, order)]["jy"]))
    return A if A > 0.0 else 1.0e-300


def jones(jx: complex, jy: complex) -> float:
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def load_jt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (it, Jx, Jy) from Jt_node_XXXX.dat."""
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        rows.append([float(f[0]), float(f[2]), float(f[3])])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def hann_window(nt: int) -> np.ndarray:
    # Match mod_hhg.f90: 0.5*(1-cos(2π (it-1)/(nt-1))), it=1..nt
    it = np.arange(nt, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * math.pi * it / float(nt - 1)))


def fft_complex_amp(jx: np.ndarray, jy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalized FFT of Hann-windowed current (FFTW/numpy equivalent for this check)."""
    nt = jx.size
    w = hann_window(nt)
    jwx = np.fft.fft(jx * w)
    jwy = np.fft.fft(jy * w)
    # positive freqs including DC: length nt//2+1 — use full then index 0..n_omega-1
    return jwx, jwy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plus-dir", type=Path, required=True)
    ap.add_argument("--minus-dir", type=Path, required=True)
    ap.add_argument("--alpha-id", type=int, default=4)
    ap.add_argument("--anti-id", type=int, default=6)
    ap.add_argument(
        "--ref-plus",
        type=Path,
        required=True,
        help="Full GH3 +N run dir (all nodes) for A_n scale",
    )
    ap.add_argument(
        "--ref-minus",
        type=Path,
        required=True,
        help="Full GH3 -N run dir (all nodes) for A_n scale",
    )
    ap.add_argument("--abs-atol", type=float, default=CEP_V2_ABS_ATOL)
    ap.add_argument("--rel-atol", type=float, default=CEP_V2_REL_ATOL)
    ap.add_argument("--eta", type=float, default=CEP_V2_ETA)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    # --- A_n from full reference GH3 ---
    man = load_manifest(args.ref_plus / "nodes_manifest.input.dat")
    pairs = pair_nodes_by_neg_alpha(man)
    modes_rp = load_node_modes(args.ref_plus / "HHG_nodes_modes.dat")
    modes_rm = load_node_modes(args.ref_minus / "HHG_nodes_modes.dat")

    mp = load_modes_pair(args.plus_dir / "HHG_nodes_modes.dat")
    mm = load_modes_pair(args.minus_dir / "HHG_nodes_modes.dat")

    orders = tuple(HARD_ORDERS) + DIAG_ORDERS
    rows = []
    for h in orders:
        A_n = scale_A_n(modes_rp, modes_rm, pairs, h)
        jp = mp[(args.alpha_id, h)]
        jm = mm[(args.anti_id, h)]
        d = jones(jp["jx"] + jm["jx"], jp["jy"] + jm["jy"])
        amp_p = jones(jp["jx"], jp["jy"])
        amp_m = jones(jm["jx"], jm["jy"])
        den_local = max(amp_p, amp_m)
        e_abs = d / max(A_n, 1.0e-300)
        e_rel = d / max(den_local, 1.0e-300)
        strong = den_local >= args.eta * A_n
        abs_fail = e_abs > args.abs_atol
        rel_fail = bool(strong and e_rel > args.rel_atol)
        rows.append(
            {
                "harmonic": h,
                "d": d,
                "A_n": A_n,
                "amp_plus": amp_p,
                "amp_minus": amp_m,
                "den_local": den_local,
                "e_abs": e_abs,
                "e_rel": e_rel,
                "strong": strong,
                "abs_fail": abs_fail,
                "rel_fail": rel_fail,
                "legacy_bug_d_over_local": d / max(den_local, 1.0e-300),
                "phi_plus": jp["phi"],
                "phi_minus": jm["phi"],
                "dphi_minus_plus_pi": abs(
                    ((jm["phi"] - (jp["phi"] + math.pi)) + math.pi) % (2 * math.pi) - math.pi
                ),
            }
        )

    hard_rows = [r for r in rows if r["harmonic"] in HARD_ORDERS]
    max_e_abs = max(r["e_abs"] for r in hard_rows)
    max_e_rel_strong = max(
        (r["e_rel"] for r in hard_rows if r["strong"]), default=0.0
    )
    abs_pass = not any(r["abs_fail"] for r in hard_rows)
    rel_pass = not any(r["rel_fail"] for r in hard_rows)

    # --- J(t) metrics ---
    jt_p = args.plus_dir / f"Jt_node_{args.alpha_id:04d}.dat"
    jt_m = args.minus_dir / f"Jt_node_{args.anti_id:04d}.dat"
    time_rep: dict = {"status": "UNAVAILABLE"}
    fft_rep: dict = {"status": "UNAVAILABLE"}

    if jt_p.is_file() and jt_m.is_file():
        _, jxp, jyp = load_jt(jt_p)
        _, jxm, jym = load_jt(jt_m)
        assert jxp.shape == jxm.shape
        nt = int(jxp.size)
        nml = parse_input_nml(args.plus_dir / "input.nml")
        dt = float(nml["dt_au"])
        wvl = float(nml.get("wvl_nm", 3200.0))
        omega0 = float(nml.get("omega0", omega0_from_wvl_nm(wvl)))

        # Time-domain e_t,∞
        amp_p_t = np.sqrt(jxp * jxp + jyp * jyp)
        amp_m_t = np.sqrt(jxm * jxm + jym * jym)
        sum_t = np.sqrt((jxp + jxm) ** 2 + (jyp + jym) ** 2)
        A_t = float(np.max(0.5 * (amp_p_t + amp_m_t)))
        e_t_inf = float(np.max(sum_t) / max(A_t, 1.0e-300))
        time_rep = {
            "status": "OK",
            "nt": nt,
            "A_t": A_t,
            "max_norm_sum": float(np.max(sum_t)),
            "e_t_inf": e_t_inf,
            "definition": "A_t=max_t(||J+||+||J-||)/2; e_t_inf=max||J++J-||/A_t",
        }

        # FFT linearity at hard harmonic bins
        jwx_p, jwy_p = fft_complex_amp(jxp, jyp)
        jwx_m, jwy_m = fft_complex_amp(jxm, jym)
        jwx_s, jwy_s = fft_complex_amp(jxp + jxm, jyp + jym)
        fft_rows = []
        max_lin_rel = 0.0
        for h in orders:
            iw = harmonic_fft_index(nt, dt, omega0, h)  # 1-based
            i0 = iw - 1
            sum_from_parts_x = jwx_p[i0] + jwx_m[i0]
            sum_from_parts_y = jwy_p[i0] + jwy_m[i0]
            dx = sum_from_parts_x - jwx_s[i0]
            dy = sum_from_parts_y - jwy_s[i0]
            num = jones(dx, dy)
            # Denominator: component scale, not the cancelled sum (avoids /~0 blow-up).
            den = max(
                jones(jwx_p[i0], jwy_p[i0]),
                jones(jwx_m[i0], jwy_m[i0]),
                jones(jwx_s[i0], jwy_s[i0]),
                1.0e-300,
            )
            lin_rel = num / den
            max_lin_rel = max(max_lin_rel, lin_rel)
            d_fft = jones(sum_from_parts_x, sum_from_parts_y)
            fft_rows.append(
                {
                    "harmonic": h,
                    "iw_1based": iw,
                    "fft_linearity_abs": num,
                    "fft_linearity_rel": lin_rel,
                    "d_from_fft_sum": d_fft,
                }
            )
        fft_rep = {
            "status": "OK",
            "max_linearity_rel": max_lin_rel,
            "pass_linearity": max_lin_rel <= 1.0e-12,
            "harmonics": fft_rows,
            "note": (
                "numpy FFT of Hann-windowed J; linearity den = max(|Jw+|,|Jw-|,|Jw_sum|), "
                "not the cancelled residual."
            ),
        }

    gate = "CEP_ABS_METRIC_CORRECTED"
    if abs_pass and rel_pass:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "metric_fix": (
            "e_abs=d/A_n with A_n from full GH3 reference; "
            "e_rel=d/max(|J+|,|J-|); previous script wrongly set both to d/local. "
            "FFT linearity den = max(|Jw+|,|Jw-|,|Jw_sum|), not cancelled residual."
        ),
        "pair": {"alpha_id": args.alpha_id, "anti_id": args.anti_id},
        "plus_dir": str(args.plus_dir),
        "minus_dir": str(args.minus_dir),
        "ref_plus": str(args.ref_plus),
        "ref_minus": str(args.ref_minus),
        "thresholds": {
            "abs_atol": args.abs_atol,
            "rel_atol": args.rel_atol,
            "eta": args.eta,
        },
        "harmonics": rows,
        "max_e_abs_hard": max_e_abs,
        "max_e_rel_strong_hard": max_e_rel_strong,
        "abs_pass": abs_pass,
        "rel_pass": rel_pass,
        "verdict": verdict,
        "time_domain": time_rep,
        "fft_linearity": fft_rep,
        "next_step": (
            "If abs_pass is false under corrected A_n, submit same-binary "
            "4-task dephasing on/off × ±N on part_2/OMP=8."
            if not abs_pass
            else "Corrected e_abs PASS; dephasing-off not required for this residual."
        ),
        "gh5": "HOLD",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "max_e_abs_hard": max_e_abs,
                "max_e_rel_strong_hard": max_e_rel_strong,
                "abs_pass": abs_pass,
                "rel_pass": rel_pass,
                "e_t_inf": time_rep.get("e_t_inf"),
                "fft_lin_max": fft_rep.get("max_linearity_rel"),
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


def load_modes_pair(path: Path) -> dict[tuple[int, int], dict]:
    """Lightweight modes loader (same columns as HHG_nodes_modes)."""
    out: dict[tuple[int, int], dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        nid, order = int(f[0]), int(f[4])
        out[(nid, order)] = {
            "w": float(f[1]),
            "I": float(f[2]),
            "phi": float(f[3]),
            "jx": complex(float(f[5]), float(f[6])),
            "jy": complex(float(f[7]), float(f[8])),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
