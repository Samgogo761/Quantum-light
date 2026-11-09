"""M88 4x4 coherent spot-check. Same 1600 nm protocol as B1.

Do not edit B1. Do not densify M52. Not a BZ-converged claim.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import b1_1600nm_model_compare_20261029 as b1

OUT = ROOT / "results/q_line/m88_1600nm_4x4_20261029"
DOCS = ROOT / "docs/04_BSV量子光"
EXE = b1.EXE
TB_PLUS = b1.TB["m88_plus"]
TB_MINUS = b1.TB["m88_minus"]
NV, NB_END = 58, 88


def nml_text(tb: Path, label: str) -> str:
    tb_s = str(tb).replace("\\", "/")
    return f"""! {label} 1600 nm 25 fs coherent 4x4. Same protocol as B1, nkx=nky=4.
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "{tb_s}"
  wannier_hr_file = ""
  wannier_r_file  = ""
/
&kgrid
  nkx = 4
  nky = 4
/
&bands
  nv_orig = {NV}
  nb_start = 1
  nb_end = {NB_END}
/
&laser
  wvl_nm = 1600.0
  intensity_Wcm2 = 1.0e10
  theta_deg = 0.0
  phi_cep_deg = 0.0
  ncyc = 13.3836
  env_type = 1
  ellipticity = 0.0
  delta_phase_deg = 90.0
/
&laser2
  wvl_nm_2 = 0.0
  intensity_Wcm2_2 = 0.0
/
&external_field
  use_external_A = .false.
  external_A_file = ""
/
&timestep
  dt = 0.35
  n_dt_deph = 5
/
&dephasing
  T2_fs = 1.0e30
  T2_cycles = 0.5
/
&bsv
  bsv_enabled = .false.
/
&method
  gauge_method = 'lg_cov'
/
&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .false.
/
&output
  save_geometry = .false.
  save_occupation = .false.
  save_complex_hhg = .true.
/
"""


def progress(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def analyze_pair(plus: Path, minus: Path) -> dict:
    rec = b1.analyze_pair(plus, minus, "m88")
    rec["kgrid"] = "4x4"
    rec["model"] = "m88"
    return rec


def write_reports(payload: dict) -> None:
    md = DOCS / "线Q_M88_4x4抽检_一页_20261029.md"
    h = payload.get("m88_4x4", {}).get("harmonics", {})
    h2_b1 = payload.get("m88_2x2_b1", {}).get("harmonics", {}).get("2", {})
    h2 = h.get("2", {})
    h4 = h.get("4", {})
    drop = None
    if h2 and h2_b1 and h2_b1.get("odd_amp2d"):
        drop = h2_b1["odd_amp2d"] / max(h2.get("odd_amp2d") or 0.0, 1e-30)
    lines = [
        "# 线 Q：M88 4×4 抽检一页（2026-10-29）",
        "",
        "同一 1600 nm、25 fs、10^10 W/cm^2、lg_cov 相干光。不问量子光。",
        "目的：M52 的 2×2→4×4 振幅降 6.6 倍，在 M88 上是不是同样的 k 采样。",
        "不加密 M52，不把 4×4 叫完整 BZ。",
        "",
        "| 网格 | H2 奇振幅 | 奇/偶 | Jx 相位 | Jy/Jx | 去边窗 | H4 奇振幅 | H4 Δφ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if h2_b1:
        lines.append(
            f"| M88 2×2 (B1) | {h2_b1.get('odd_amp2d', float('nan')):.3e} | "
            f"{h2_b1.get('odd_over_even', float('nan')):.4g} | "
            f"{h2_b1.get('odd_jx_phase_deg', float('nan')):.1f}° | "
            f"{h2_b1.get('odd_jy_over_jx_abs', float('nan')):.2f} | "
            f"{100 * (h2_b1.get('window_rel_odd_amp') or 0):.1f}% | — | — |"
        )
    if h2:
        lines.append(
            f"| M88 4×4 | {h2.get('odd_amp2d', float('nan')):.3e} | "
            f"{h2.get('odd_over_even', float('nan')):.4g} | "
            f"{h2.get('odd_jx_phase_deg', float('nan')):.1f}° | "
            f"{h2.get('odd_jy_over_jx_abs', float('nan')):.2f} | "
            f"{100 * (h2.get('window_rel_odd_amp') or 0):.1f}% | "
            f"{h4.get('odd_amp2d', float('nan')):.3e} | "
            f"{h4.get('window_dphi_jx_odd_deg', float('nan')):.1f}° |"
        )
    lines += ["", "## 判定", ""]
    if drop is not None:
        lines.append(f"- 2×2 / 4×4 振幅比 = **{drop:.2f}**（M52 同协议是 6.6）。")
        dphi = None
        if h2 and h2_b1 and h2.get("odd_jx_phase_deg") is not None and h2_b1.get("odd_jx_phase_deg") is not None:
            raw = float(h2["odd_jx_phase_deg"]) - float(h2_b1["odd_jx_phase_deg"])
            dphi = (raw + 180.0) % 360.0 - 180.0
        if dphi is not None and abs(dphi) > 40.0:
            lines.append(
                f"- **相位被网格改掉**：Δφ = **{dphi:.1f}°**，"
                f"Jy/Jx 从 {h2_b1.get('odd_jy_over_jx_abs', float('nan')):.2f} 到 {h2.get('odd_jy_over_jx_abs', float('nan')):.2f}。"
                "2×2 的 M88 相位/偏振不能当材料定量。不上 8×8；下一步若做材料只做代表 k。"
            )
        elif drop > 4.0:
            lines.append("- 与 M52 同类：当前振幅主要是粗网格采样，不能当材料定量。不要自动上 8×8。")
        elif drop < 1.5:
            lines.append("- 4×4 相对 2×2 稳定得多。M88 比 M52 更适合作为后续材料抽检。")
        else:
            lines.append("- 中等网格敏感。记录下来，仍不升级到 8×8，除非相位/符号也被改掉。")
    if h2:
        lines.append(
            f"- 4×4 相位 {h2.get('odd_jx_phase_deg'):.1f}°，B1 2×2 是 {h2_b1.get('odd_jx_phase_deg'):.1f}°。"
            f"去边窗 {100 * (h2.get('window_rel_odd_amp') or 0):.1f}%；H4 去边 Δφ={h4.get('window_dphi_jx_odd_deg', float('nan')):.1f}°。"
        )
    lines += ["", f"归档：`{OUT.as_posix()}`。", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "m88_4x4.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["one_page"] = str(md)


def main() -> int:
    if not EXE.is_file():
        raise FileNotFoundError(EXE)
    if not TB_PLUS.is_file() or not TB_MINUS.is_file():
        raise FileNotFoundError("M88 tb missing in tb_b1/")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("m88_plusN", OUT / "m88_plusN", TB_PLUS),
        ("m88_minusN", OUT / "m88_minusN", TB_MINUS),
    ]
    recs = {}
    procs = []
    env = b1.sbe_env()
    for name, rundir, tb in jobs:
        rundir.mkdir(parents=True, exist_ok=True)
        (rundir / "input.nml").write_text(nml_text(tb, name), encoding="utf-8")
        jt = rundir / "Jt.dat"
        if jt.is_file() and jt.stat().st_size > 1000:
            recs[name] = {"status": "ok", "skipped": True, "rundir": str(rundir)}
            progress(f"skip existing {name}")
            continue
        log = (rundir / "stdout.txt").open("w", encoding="utf-8")
        t0 = time.perf_counter()
        progress(f"launch SBE {name}")
        p = subprocess.Popen(
            [str(EXE)], cwd=str(rundir), env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        procs.append((name, rundir, p, log, t0))
    for name, rundir, p, log, t0 in procs:
        rc = p.wait()
        log.close()
        wall = time.perf_counter() - t0
        shutil.copyfile(rundir / "stdout.txt", rundir / "run.log")
        jt = rundir / "Jt.dat"
        recs[name] = {
            "status": "ok" if rc == 0 and jt.is_file() else "fail",
            "returncode": rc, "wall_s": wall, "rundir": str(rundir),
            "jt_bytes": jt.stat().st_size if jt.is_file() else 0,
        }
        progress(f"done {name} rc={rc} wall={wall:.1f}s")
        (rundir / "job.json").write_text(json.dumps(recs[name], indent=2), encoding="utf-8")
    payload = {
        "protocol": "1600nm_25fs_1e10_lg_cov_4x4",
        "jobs": recs,
        "did_not_edit_b1": True,
        "did_not_densify_m52": True,
        "not_bz_converged": True,
        "not_quantum_light": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    b1_json = ROOT / "results/q_line/b1_1600nm_model_compare_20261029/b1_compare.json"
    if b1_json.is_file():
        old = json.loads(b1_json.read_text(encoding="utf-8"))
        payload["m88_2x2_b1"] = old.get("models", {}).get("m88")
    if recs.get("m88_plusN", {}).get("status") == "ok" and recs.get("m88_minusN", {}).get("status") == "ok":
        payload["m88_4x4"] = analyze_pair(OUT / "m88_plusN", OUT / "m88_minusN")
    write_reports(payload)
    progress(f"report {payload.get('one_page')}")
    fails = [k for k, v in recs.items() if v.get("status") != "ok"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
