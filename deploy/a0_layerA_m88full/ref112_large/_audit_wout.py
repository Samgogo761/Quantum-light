#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np

ROOT = Path("/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large")
PROD = Path("/public/home/wangjs/project/CrI3_TB/wannier")


def parse_eig(path, n_k, n_b):
    e = np.zeros((n_k, n_b))
    with path.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            e[int(p[1]) - 1, int(p[0]) - 1] = float(p[2])
    return e


e_new = parse_eig(ROOT / "CrI3_112large.eig", 576, 400)
e_old = parse_eig(PROD / "CrI3.eig", 576, 200)
d = np.abs(e_new[:, :200] - e_old)
imax = np.unravel_index(int(np.argmax(d)), d.shape)
print("max_abs", float(d[imax]), "ik", int(imax[0]) + 1, "ib", int(imax[1]) + 1)
print("old", float(e_old[imax]), "new", float(e_new[imax]))
print("n_gt_1meV", int(np.sum(d > 1e-3)))
print("n_gt_10ueV", int(np.sum(d > 1e-5)))
print("n_gt_1ueV", int(np.sum(d > 1e-6)))
print("max_abs_by_band_199_200", float(d[:, 198].max()), float(d[:, 199].max()))
print("max_abs_bands_1_56", float(d[:, :56].max()), "57_144", float(d[:, 56:144].max()), "145_200", float(d[:, 144:200].max()))

w = (ROOT / "CrI3_112large.wout").read_text(errors="replace")
print("\n=== 112large wout slices ===")
for pat in [
    r"Number of Wannier Functions\s+:\s+\d+",
    r"Number of bands in inner window",
    r"Number of bands in outer window",
    r"Frozen",
    r"Disentangle",
    r"converged",
    r"num_iter",
    r"write_hr",
    r"write_tb",
]:
    pass

# print key blocks
for key in ["Input file", "System", "DISENTANGLE", "WANNIERISE", "Final State", "PLOTTING"]:
    i = w.find(key)
    print(f"-- find {key!r}", i)

# extract lines around disentangle summary
lines = w.splitlines()
for i, ln in enumerate(lines):
    if any(s in ln for s in [
        "Number of Wannier",
        "Number of input",
        "Number of bands",
        "inner window",
        "outer window",
        "frozen",
        "Frozen",
        "dis_win",
        "dis_froz",
        "Disentanglement",
        "CONV",
        "delta_omega",
        "Omega I",
        "write_hr",
        "write_tb",
        "seedname",
    ]):
        if i < 250 or "Omega" in ln or "window" in ln.lower() or "Disentang" in ln or "CONV" in ln:
            print(f"{i+1:5d}|{ln}")

print("\n--- first 120 lines ---")
print("\n".join(lines[:120]))

pw = PROD / "CrI3.wout"
if pw.is_file():
    t = pw.read_text(errors="replace")
    m = re.findall(r"Final Spread \(Ang\^2\)\s+Omega Total\s+=\s+([-\d.Ee+]+)", t)
    oi = re.search(r"Omega I\s+=\s+([-\d.Ee+]+)", t)
    print("\nPROD omega_total", m[-1] if m else None, "Omega I", oi.group(1) if oi else None)
    print("PROD bytes", pw.stat().st_size)

print("\n--- extra files ---")
for p in sorted(ROOT.glob("CrI3_112large*")):
    print(p.name, p.stat().st_size)
