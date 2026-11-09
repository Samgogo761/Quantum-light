#!/usr/bin/env python3
"""Read-only summary of 1600 nm production JSON. No new propagation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/q1600_bz_stat_production_20261028"


def zxy(pack):
    if isinstance(pack, list):
        return complex(pack[0]["re"], pack[0]["im"]), complex(pack[1]["re"], pack[1]["im"])
    return complex(pack["re"], pack["im"]), None


def main() -> int:
    files = sorted(OUT.glob("physics_i*_nk*_gh*.json"))
    for fn in files:
        d = json.loads(fn.read_text(encoding="utf-8"))
        print("=" * 72)
        print(fn.name, "I0", d["i0_W_cm2"], "nk", d["nk"], "wall_s", round(d["wall_s_total"], 2))
        rows = {}
        for r in d["rows"]:
            key = f"{r['kind']}_th{int(round(r['theta']*100))}"
            rows[key] = r
            h = r["H2"]
            ht = r["H2_window_trim_16pct"]
            ox, oy = zxy(h["odd"])
            ex, ey = zxy(h["even"])
            print(
                f"  {r['kind']:9s} th={r['theta']:.3f} odd={h['odd_norm2D']:.4e} "
                f"even={h['even_norm2D']:.4e} trim={ht['odd_norm2D']:.4e} "
                f"drel={abs(ht['odd_norm2D']-h['odd_norm2D'])/h['odd_norm2D'] if h['odd_norm2D'] else None}"
            )
            print(f"    Jx_odd={ox:.4e} Jy_odd={oy:.4e} argx={h['odd'][0]['phase_deg']:.2f} argy={h['odd'][1]['phase_deg']:.2f}")
            print(f"    Jx_even={ex:.4e} Jy_even={ey:.4e}")
            print(
                "    contrast x/y/45/135:",
                h["analyzers"]["x"]["contrast_coh"],
                h["analyzers"]["y"]["contrast_coh"],
                h["analyzers"]["45deg"]["contrast_coh"],
                h["analyzers"]["135deg"]["contrast_coh"],
            )
            print("    Pcoh+", r["Pcoh_H2"]["plus"], "Pstat+", r["Pstat_H2"]["plus"])
            print("    Pstat-Pcoh+", r["Pstat_minus_Pcoh_H2_plus"])
            harms = r["harmonics_mean"]
            for q in ("1", "2", "3", "4"):
                o = harms[q]["odd"]
                print(f"    H{q} odd |Jx|={o[0]['abs']:.3e} |Jy|={o[1]['abs']:.3e} even|Jx|={harms[q]['even'][0]['abs']:.3e}")
        ds0, dsp = rows["ds_th0"], rows["ds_th314"]
        z0, _ = zxy(ds0["H2"]["odd"])
        zp, _ = zxy(dsp["H2"]["odd"])
        import cmath
        print("  DS amp ratio pi/0", abs(zp) / abs(z0) if abs(z0) else None)
        print("  DS dphi deg", (cmath.phase(zp / z0) * 180 / 3.141592653589793) if abs(z0) else None)
    kn = json.loads((OUT / "kernel_nk6.json").read_text(encoding="utf-8"))
    print("=" * 72)
    print("kernel R_mat", kn["R_mat_equal_over_split"])
    for name, rec in kn["pairs"].items():
        print(name, rec["odd_H2_mixed_over_E2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
