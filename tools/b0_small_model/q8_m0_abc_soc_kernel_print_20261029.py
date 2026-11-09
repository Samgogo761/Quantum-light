from __future__ import annotations

import json
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "results/q_line/abc_soc_kernel_scan_20261029/soc_kernel_scan.json"
d = json.loads(p.read_text(encoding="utf-8"))
print("verdict", d["verdict"])
print("adia_span", d["adiabatic_soc_arg_span_deg"])
print("ft_span", d["finite_time_arg_span_deg"])
print("--- adiabatic SOC ---")
hdr = f"{'frac':>6} {'abs':>10} {'arg':>8} {'gap':>7} {'t0':>9} {'t1':>9} {'t2':>9} {'net/t2':>7}"
print(hdr)
for c in d["adiabatic_soc"]:
    print(
        f"{c['frac']:6.3f} {c['odd_x']['abs']:10.3e} {c['odd_x']['arg_deg']:8.2f} "
        f"{c['min_gap']:7.4f} {c['t0']['abs']:9.3e} {c['t1']['abs']:9.3e} "
        f"{c['t2']['abs']:9.3e} {c['net_over_t2']:7.3f}"
    )
print("--- FT ---")
for k, v in d["finite_time"].items():
    print(
        k,
        "frac", v.get("lam_so_frac"),
        "ft", None if v.get("finite_time_arg_deg") is None else round(v["finite_time_arg_deg"], 2),
        "ad", None if v.get("adiabatic_arg_deg") is None else round(v["adiabatic_arg_deg"], 2),
        "odd2d", v.get("odd_norm2D"),
        "off", v.get("gated_off"),
    )
print("--- adiabatic delta ---")
want = {0.0, 0.5, 1.0, 1.5, 2.0}
for c in d["adiabatic_delta"]:
    if round(c["frac"], 2) in want:
        print(
            f"d={c['frac']:.2f} abs={c['odd_x']['abs']:.3e} "
            f"arg={c['odd_x']['arg_deg']:.2f} gap={c['min_gap']:.4f}"
        )
