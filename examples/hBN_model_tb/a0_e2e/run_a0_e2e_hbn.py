#!/usr/bin/env python3
"""hBN end-to-end A0 acceptance (classical complex, coherent GH, SV smoke)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E2E = Path(__file__).resolve().parent
EXE = ROOT / ("hhg_sbe.exe" if sys.platform.startswith("win") else "hhg_sbe")
REPORT = E2E / "a0_e2e_report.md"


def run_case(name: str, nml: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    src = E2E / nml
    dst = work / "input.nml"
    shutil.copy2(src, dst)
    # copy relative TB resolution: nml points to ../hBN_model_tb.dat from a0_e2e,
    # but we run inside work subdir -> need ../../ or patch. Use absolute via rewrite.
    text = src.read_text(encoding="utf-8")
    tb = (E2E.parent / "hBN_model_tb.dat").as_posix()
    text = text.replace('wannier_tb_file = "../hBN_model_tb.dat"',
                        f'wannier_tb_file = "{tb}"')
    dst.write_text(text, encoding="utf-8")
    log = work / "run.log"
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run([str(EXE), "input.nml"], cwd=work, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{name} failed, see {log}")
    return work


def load_hhg(path: Path):
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 4]  # order, total


def load_complex_total(path: Path):
    data = np.loadtxt(path, comments="#")
    # harmonic omega ReJx ImJx ReJy ImJy |Jx|^2 |Jy|^2 total
    return data[:, 0], data[:, 8]


def main() -> int:
    if not EXE.exists():
        print(f"ERROR: solver not found: {EXE}", file=sys.stderr)
        return 2

    lines = ["# hBN A0 end-to-end report", "", f"- solver: `{EXE}`", ""]
    ok_all = True

    # 1) classical complex regression
    w1 = run_case("classical", "input_classical_complex.nml", E2E / "work_classical")
    o1, s1 = load_hhg(w1 / "HHG.dat")
    oc, sc = load_complex_total(w1 / "HHG_complex.dat")
    # align lengths
    n = min(len(s1), len(sc))
    denom = np.maximum(np.abs(s1[:n]), 1e-300)
    err = np.max(np.abs(s1[:n] - sc[:n]) / denom)
    pass1 = err < 1e-10
    ok_all &= pass1
    lines += [
        "## 1. Classical complex FFT regression",
        f"- max relative error HHG.dat vs HHG_complex total: `{err:.3e}`",
        f"- gate `<1e-10`: **{'PASS' if pass1 else 'FAIL'}**",
        "",
    ]

    # 2) coherent GH3
    w2 = run_case("coherent", "input_coherent_gh3.nml", E2E / "work_coherent")
    ics = np.loadtxt(w2 / "HHG_ics_cs.dat", comments="#")
    # cols: order omega ICS CS variance
    ics_s, cs_s, var_s = ics[:, 2], ics[:, 3], ics[:, 4]
    rel_ics_cs = np.max(np.abs(ics_s - cs_s) / np.maximum(ics_s, 1e-300))
    pass2a = np.all(var_s >= -1e-12 * np.maximum(ics_s, 1e-300))
    o_cl, s_cl = load_hhg(w1 / "HHG.dat")

    def nearest(orders, target):
        return int(np.argmin(np.abs(orders - target)))

    ratios = []
    for t in (1.0, 3.0, 5.0):
        i_c = nearest(ics[:, 0], t)
        i_cl = nearest(o_cl, t)
        ratios.append(ics_s[i_c] / max(s_cl[i_cl], 1e-300))
    # Primary classical-limit gate: ICS ≈ classical at shared mean intensity.
    # Residual ICS-CS is expected: residual Husimi vacuum width + nonlinear HHG
    # makes coherent sums cancel even for |α0|=30.
    pass2b = all(abs(r - 1.0) < 0.05 for r in ratios)
    pass2 = pass2a and pass2b and (w2 / "nodes_coherent_gh3.dat").exists()
    ok_all &= pass2
    lines += [
        "## 2. Coherent Gauss-Hermite (α0=30, θ=90°, ⟨I⟩ matched to classical)",
        f"- ICS≥CS (num): **{'PASS' if pass2a else 'FAIL'}**",
        f"- ICS≈classical @H1/H3/H5 ratios: `{[float(f'{r:.4f}') for r in ratios]}` (gate |r-1|<5%: {'PASS' if pass2b else 'FAIL'})",
        f"- ICS vs CS max rel (informational, not hard gate): `{rel_ics_cs:.3e}`",
        f"- nodes manifest exists: **{(w2 / 'nodes_coherent_gh3.dat').exists()}**",
        f"- overall: **{'PASS' if pass2 else 'FAIL'}**",
        "",
    ]

    # 3) SV smoke th=0 / 180
    w3a = run_case("sv0", "input_sv_r0p8_th0.nml", E2E / "work_sv_th0")
    w3b = run_case("sv180", "input_sv_r0p8_th180.nml", E2E / "work_sv_th180")
    ics_a = np.loadtxt(w3a / "HHG_ics_cs.dat", comments="#")
    ics_b = np.loadtxt(w3b / "HHG_ics_cs.dat", comments="#")
    nodes_a = np.loadtxt(w3a / "nodes_sv_r0p8_th0.dat", comments="#")
    nodes_b = np.loadtxt(w3b / "nodes_sv_r0p8_th180.dat", comments="#")
    pass3a = np.all(ics_a[:, 4] >= -1e-12 * np.maximum(ics_a[:, 2], 1e-300))
    pass3b = np.all(ics_b[:, 4] >= -1e-12 * np.maximum(ics_b[:, 2], 1e-300))
    # distributions should differ in phi/I
    dist = np.linalg.norm(nodes_a[:, 4:6] - nodes_b[:, 4:6])  # I, phi cols: 4,5 (0-index: id w re im I phi)
    # columns: id weight re im I phi -> indices 0..5; I=4 phi=5
    pass3c = dist > 1e-6
    pass3 = pass3a and pass3b and pass3c and (w3a / "HHG_nodes_modes.dat").exists()
    ok_all &= pass3
    lines += [
        "## 3. Squeezed-vacuum direction smoke (r=0.8, θs=0° vs 180°)",
        f"- ICS≥CS @θ=0: **{'PASS' if pass3a else 'FAIL'}**",
        f"- ICS≥CS @θ=180: **{'PASS' if pass3b else 'FAIL'}**",
        f"- node (I,φ) distribution L2 distance: `{dist:.3e}` (gate >0: {'PASS' if pass3c else 'FAIL'})",
        f"- HHG_nodes_modes.dat present: **{(w3a / 'HHG_nodes_modes.dat').exists()}**",
        f"- overall: **{'PASS' if pass3 else 'FAIL'}**",
        "",
        "## Notes",
        "- Equal-mean-intensity mapping: not experimental BSV energy calibration.",
        "- Do NOT interpret ICS-CS as quantum squeezing.",
        "",
        f"## OVERALL: **{'PASS' if ok_all else 'FAIL'}**",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.buffer.write((REPORT.read_text(encoding="utf-8") + "\n").encode("utf-8", errors="replace"))
    print(f"OVERALL_EXIT={'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
