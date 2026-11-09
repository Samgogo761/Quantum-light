#!/usr/bin/env python3
"""Field-only 2ω diagnostic: H2 of E(t)^2 for the same 1600 nm protocol.

No electronic propagation. If DS θ=π/θ=0 matches the M0 odd-current ratio,
the 1600 nm compression-axis effect is mostly E^2 quadrature weighting,
not extra material frequency allocation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from q8_m0_1600nm_bz_stat_production_20261028 import (
    OUT,
    build_times,
    field_from_xp,
    gh1d_nodes,
    project_qs,
    protocol,
    xp_state,
)


def e2_h2(kind: str, theta: float, proto: dict, meta: dict, gh: int) -> dict:
    st = xp_state(kind, proto, theta)
    ghn = gh1d_nodes(st, gh)
    acc = np.zeros(meta["times"].size, dtype=float)
    acc2 = np.zeros(meta["times"].size, dtype=float)
    for n in ghn["nodes"]:
        fld = field_from_xp(meta, proto, n["X"], n["P"])
        e = fld["E_VA"]
        acc += n["weight"] * e
        acc2 += n["weight"] * e * e
    lo, hi = float(meta["times"][0]), float(meta["times"][-1])
    z_mean_e = project_qs(meta["times"], acc[:, None], meta["omega"], [1, 2], lo, hi)
    z_e2 = project_qs(meta["times"], acc2[:, None], meta["omega"], [2], lo, hi)[0, 0]
    # Also the 2ω of <E>^2 versus <E^2>
    z_meane2 = project_qs(meta["times"], (acc * acc)[:, None], meta["omega"], [2], lo, hi)[0, 0]
    return {
        "kind": kind,
        "theta": float(theta),
        "n_nodes": ghn["n_nodes"],
        "H1_of_mean_E": {"re": float(z_mean_e[0, 0].real), "im": float(z_mean_e[0, 0].imag), "abs": float(abs(z_mean_e[0, 0]))},
        "H2_of_mean_E": {"re": float(z_mean_e[1, 0].real), "im": float(z_mean_e[1, 0].imag), "abs": float(abs(z_mean_e[1, 0]))},
        "H2_<E2>": {"re": float(z_e2.real), "im": float(z_e2.imag), "abs": float(abs(z_e2)), "phase_deg": float(np.degrees(np.angle(z_e2)))},
        "H2_<E>2": {"re": float(z_meane2.real), "im": float(z_meane2.imag), "abs": float(abs(z_meane2)), "phase_deg": float(np.degrees(np.angle(z_meane2)))},
        "mean_X": st["mean_X"],
        "mean_P": st["mean_P"],
        "var_X": st["var_X"],
        "var_P": st["var_P"],
        "analytic_X2_minus_P2": float((st["mean_X"] ** 2 + st["var_X"]) - (st["mean_P"] ** 2 + st["var_P"])),
    }


def main() -> int:
    proto = protocol(1.0e10)
    meta = build_times(32)
    rows = []
    for kind, th in (("coherent", 0.0), ("bsv", 0.0), ("ds", 0.0), ("ds", np.pi)):
        rec = e2_h2(kind, th, proto, meta, 9 if kind != "coherent" else 5)
        rows.append(rec)
        print(
            f"{kind:9s} th={th:.3f} H2<E2>={rec['H2_<E2>']['abs']:.4e} "
            f"arg={rec['H2_<E2>']['phase_deg']:.2f} "
            f"H2<E>2={rec['H2_<E>2']['abs']:.4e} "
            f"X2-P2={rec['analytic_X2_minus_P2']:.4e}"
        )
    by = {f"{r['kind']}_th{int(round(r['theta']*100))}": r for r in rows}
    a0 = by["ds_th0"]["H2_<E2>"]["abs"]
    ap = by["ds_th314"]["H2_<E2>"]["abs"]
    ac = by["coherent_th0"]["H2_<E2>"]["abs"]
    ab = by["bsv_th0"]["H2_<E2>"]["abs"]
    z0 = complex(by["ds_th0"]["H2_<E2>"]["re"], by["ds_th0"]["H2_<E2>"]["im"])
    zp = complex(by["ds_th314"]["H2_<E2>"]["re"], by["ds_th314"]["H2_<E2>"]["im"])
    zc = complex(by["coherent_th0"]["H2_<E2>"]["re"], by["coherent_th0"]["H2_<E2>"]["im"])
    zb = complex(by["bsv_th0"]["H2_<E2>"]["re"], by["bsv_th0"]["H2_<E2>"]["im"])
    out = {
        "status": "field_only_E2_H2",
        "i0_W_cm2": 1.0e10,
        "note": "No electrons. H2 of the driving field intensity/E^2 only.",
        "rows": rows,
        "ds_pi_over_0_abs": float(ap / a0) if a0 else None,
        "ds_phase_diff_deg": float(np.degrees(np.angle(zp / z0))) if abs(z0) else None,
        "bsv_over_coh_abs": float(ab / ac) if ac else None,
        "bsv_vs_coh_phase_diff_deg": float(np.degrees(np.angle(zb / zc))) if abs(zc) else None,
        "analytic_ds_pi_over_0_X2mP2": float(
            by["ds_th314"]["analytic_X2_minus_P2"] / by["ds_th0"]["analytic_X2_minus_P2"]
        ),
    }
    path = OUT / "field_only_E2_H2_i1e10_gh9.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("ds_pi/0", out["ds_pi_over_0_abs"], "dphi", out["ds_phase_diff_deg"])
    print("bsv/coh", out["bsv_over_coh_abs"], "dphi", out["bsv_vs_coh_phase_diff_deg"])
    print("analytic X2-P2 pi/0", out["analytic_ds_pi_over_0_X2mP2"])
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
