#!/usr/bin/env python3
"""Read-only dump of 1600 nm production JSON for the one-page report."""
from __future__ import annotations

import cmath
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/q_line/q1600_bz_stat_production_20261028"


def zx(pack):
    return complex(pack[0]["re"], pack[0]["im"]), complex(pack[1]["re"], pack[1]["im"])


def main() -> int:
    files = sorted(OUT.glob("physics_i*_nk*_gh*.json"))
    for fn in files:
        d = json.loads(fn.read_text(encoding="utf-8"))
        print("=" * 70)
        print(
            fn.name,
            "I0",
            d["i0_W_cm2"],
            "nk",
            d["nk"],
            "gh",
            d.get("gh_order"),
            "wall",
            round(d["wall_s_total"], 2),
            "closed",
            d.get("inversion_closed"),
            "nkpts",
            d.get("n_kpts"),
        )
        proto = d.get("protocol", {})
        print(
            " nbar",
            proto.get("nbar"),
            "r_bsv",
            proto.get("r_bsv"),
            "r_ds",
            proto.get("r_ds"),
            "alpha_ds",
            proto.get("alpha_ds"),
            "alpha_coh",
            proto.get("alpha_coh"),
            "Epeak",
            proto.get("E_peak_coh_V_m"),
            "photon_eV",
            proto.get("photon_eV"),
        )
        rows = {}
        for r in d["rows"]:
            key = f"{r['kind']}_th{int(round(r['theta'] * 100))}"
            rows[key] = r
            ox, oy = zx(r["H2"]["odd"])
            ex, ey = zx(r["H2"]["even"])
            full = r["H2"]["odd_norm2D"]
            trim = r["H2_window_trim_16pct"]["odd_norm2D"]
            print(
                f"  {key:16s} odd2D={full:.6e} even2D={r['H2']['even_norm2D']:.6e} "
                f"argx={r['H2']['odd'][0]['phase_deg']:.2f} argy={r['H2']['odd'][1]['phase_deg']:.2f} "
                f"trim_rel={(abs(trim - full) / full) if full else None:.4g} nodes={r.get('n_nodes')}"
            )
            print(f"    Jx_odd={ox:.4e} Jy_odd={oy:.4e} Jx_even={ex:.4e} Jy_even={ey:.4e}")
            print(
                f"    xC={r['H2']['analyzers']['x']['contrast_coh']:.6g} "
                f"yC={r['H2']['analyzers']['y']['contrast_coh']:.6g} "
                f"45C={r['H2']['analyzers']['45deg']['contrast_coh']:.6g} "
                f"135C={r['H2']['analyzers']['135deg']['contrast_coh']:.6g}"
            )
            print(f"    Pcoh+={r['Pcoh_H2']['plus']}")
            print(f"    Pstat+={r['Pstat_H2']['plus']}")
            print(f"    Pstat-Pcoh+={r['Pstat_minus_Pcoh_H2_plus']}")
        z0, _ = zx(rows["ds_th0"]["H2"]["odd"])
        zp, _ = zx(rows["ds_th314"]["H2"]["odd"])
        zc, _ = zx(rows["coherent_th0"]["H2"]["odd"])
        zb, _ = zx(rows["bsv_th0"]["H2"]["odd"])
        e0 = rows["ds_th0"]["H2"]["even_norm2D"]
        ep = rows["ds_th314"]["H2"]["even_norm2D"]
        print("  DS |pi/0| Jx", abs(zp) / abs(z0), "dphi", cmath.phase(zp / z0) * 180 / cmath.pi)
        print("  DS odd2D pi/0", rows["ds_th314"]["H2"]["odd_norm2D"] / rows["ds_th0"]["H2"]["odd_norm2D"])
        print("  DS even2D pi/0", (ep / e0) if e0 else None)
        print("  BSV/coh |Jx odd|", abs(zb) / abs(zc), "dphi", cmath.phase(zb / zc) * 180 / cmath.pi)
        print(
            "  BSV even/odd",
            rows["bsv_th0"]["H2"]["even_norm2D"] / rows["bsv_th0"]["H2"]["odd_norm2D"],
        )
        print(
            "  coh/ds0/dsp/bsv argx",
            rows["coherent_th0"]["H2"]["odd"][0]["phase_deg"],
            rows["ds_th0"]["H2"]["odd"][0]["phase_deg"],
            rows["ds_th314"]["H2"]["odd"][0]["phase_deg"],
            rows["bsv_th0"]["H2"]["odd"][0]["phase_deg"],
        )
    kn = json.loads((OUT / "kernel_nk6.json").read_text(encoding="utf-8"))
    print("=" * 70)
    print("kernel R_mat", kn["R_mat_equal_over_split"])
    for name, rec in kn["pairs"].items():
        print(name, "odd_norm", rec["odd_norm"], rec["odd_H2_mixed_over_E2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
