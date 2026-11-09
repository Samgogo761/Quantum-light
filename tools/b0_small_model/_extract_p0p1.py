#!/usr/bin/env python3
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
keep = {
    "name": d.get("name"),
    "nk": d.get("nk"),
    "ntpc": d.get("ntpc"),
    "grid_shift": d.get("grid_shift"),
    "targets_sin2": d.get("targets_sin2"),
    "traces_meta": d.get("traces_meta"),
}
ana = d.get("analysis") or {}
keep["diagnostics"] = ana.get("diagnostics")
keep["end_of_pulse"] = ana.get("end_of_pulse")
keep["subspace_reconstruct_Jplus_maxabs"] = ana.get("subspace_reconstruct_Jplus_maxabs")
w = (ana.get("windows") or {}).get("sin2") or {}
keep["subspace_harmonic_reconstruct_maxabs"] = w.get("subspace_harmonic_reconstruct_maxabs")
harm = {}
for name in ("J_e", "J_o", "J_e_vv", "J_e_cc", "J_e_vc", "J_o_vv", "J_o_cc", "J_o_vc"):
    if name not in w:
        continue
    harm[name] = {}
    for n in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        if n not in w[name]:
            continue
        e = w[name][n]
        harm[name][n] = {
            "hbar_Omega": e.get("hbar_Omega_over_th"),
            "abs_x": e.get("abs_x"),
            "abs_y": e.get("abs_y"),
            "x": e.get("x"),
            "y": e.get("y"),
        }
keep["harmonics_sin2"] = harm
print(json.dumps(keep, indent=2))
