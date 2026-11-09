"""B1: same 1600 nm / 25 fs / 1e10 coherent protocol, 2x2 M52 vs M88 vs full112.

M52 already exists; do not rerun it. Do not copy 0.7-1.2 GB tb files.
Use ASCII hardlinks in tb_b1/; 8.3 Chinese short names still fail in Fortran.
Not a BZ-converged claim.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q_line/b1_1600nm_model_compare_20261029"
DOCS = ROOT / "docs/04_BSV量子光"
EXE = ROOT / "hhg_sbe.exe"
M52_OLD = ROOT / "results/q_line/m52_1600nm_fewk_coherent_20261028"
TB_DIR = ROOT / "tb_b1"
MINGW_BIN = Path(r"C:\msys64\mingw64\bin")

TB = {
    "m88_plus": TB_DIR / "CrI3_M88_tb.dat",
    "m88_minus": TB_DIR / "CrI3_M88_tb_mN_Pconj.dat",
    "full112_plus": TB_DIR / "CrI3_tb.dat",
    "full112_minus": TB_DIR / "CrI3_tb_mN_Pconj.dat",
}
MODELS = {
    "m88": {"nv": 58, "nb_end": 88, "plus": "m88_plus", "minus": "m88_minus"},
    "full112": {"nv": 84, "nb_end": 112, "plus": "full112_plus", "minus": "full112_minus"},
}
AU_TO_FS = 0.024188843265857
FS_TO_AU = 1.0 / AU_TO_FS
TWOPI = 2.0 * math.pi
ORDERS = list(range(1, 6))


def nml_text(tb: Path, nv: int, nb_end: int, label: str) -> str:
    tb_s = str(tb).replace("\\", "/")
    return f"""! {label} 1600 nm 25 fs coherent 2x2. Same protocol as M52 20261028.
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
  nkx = 2
  nky = 2
/
&bands
  nv_orig = {nv}
  nb_start = 1
  nb_end = {nb_end}
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


def sbe_env() -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if MINGW_BIN.is_dir():
        env["PATH"] = str(MINGW_BIN) + os.pathsep + env.get("PATH", "")
    return env


def run_sbe(rundir: Path) -> dict:
    rundir.mkdir(parents=True, exist_ok=True)
    jt = rundir / "Jt.dat"
    if jt.is_file() and jt.stat().st_size > 1000:
        progress(f"skip existing {rundir.name}")
        return {"status": "ok", "skipped": True, "rundir": str(rundir)}
    env = sbe_env()
    log = (rundir / "stdout.txt").open("w", encoding="utf-8")
    t0 = time.perf_counter()
    progress(f"launch SBE {rundir.name}")
    p = subprocess.run(
        [str(EXE)], cwd=str(rundir), env=env,
        stdout=log, stderr=subprocess.STDOUT, check=False,
    )
    log.close()
    wall = time.perf_counter() - t0
    shutil.copyfile(rundir / "stdout.txt", rundir / "run.log")
    rec = {
        "status": "ok" if p.returncode == 0 and jt.is_file() else "fail",
        "returncode": p.returncode, "wall_s": wall, "rundir": str(rundir),
        "jt_bytes": jt.stat().st_size if jt.is_file() else 0,
    }
    progress(f"done {rundir.name} rc={p.returncode} wall={wall:.1f}s")
    (rundir / "job.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def read_jt(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 4:
            continue
        rows.append((float(p[1]), float(p[2]), float(p[3])))
    t = np.array([r[0] for r in rows], dtype=float)
    jx = np.array([r[1] for r in rows], dtype=float)
    jy = np.array([r[2] for r in rows], dtype=float)
    return t, jx, jy


def hann(n: int):
    return 0.5 * (1.0 - np.cos(TWOPI * np.arange(n) / (n - 1)))


def exact_freq_integral(t_fs, jx, jy, omega_au, trim_frac=0.0):
    n = len(t_fs)
    i0 = int(round(trim_frac * n / 2.0))
    i1 = n - i0
    t = t_fs[i0:i1]
    x = jx[i0:i1]
    y = jy[i0:i1]
    w = hann(len(t))
    dt_au = (t[1] - t[0]) * FS_TO_AU
    phase = np.exp(1j * omega_au * t * FS_TO_AU)
    ax = dt_au * np.sum(w * x * phase)
    ay = dt_au * np.sum(w * y * phase)
    return ax, ay


def cinfo(z):
    return {
        "re": float(z.real), "im": float(z.imag), "abs": float(abs(z)),
        "phase_deg": float(math.degrees(np.angle(z))) if abs(z) > 0 else None,
    }


def analyze_pair(plus: Path, minus: Path, model: str) -> dict:
    t_p, jx_p, jy_p = read_jt(plus / "Jt.dat")
    t_m, jx_m, jy_m = read_jt(minus / "Jt.dat")
    if len(t_p) != len(t_m) or np.max(np.abs(t_p - t_m)) > 1e-8:
        raise RuntimeError(f"{model} plus/minus time grids differ")
    dt_fs = float(t_p[1] - t_p[0])
    omega0 = TWOPI / (1600.0 * 1e-7 / 2.99792458e10) * AU_TO_FS
    # 1600 nm cycle 5.337 fs; omega0 in a.u.
    t_cycle_fs = 1600.0e-9 / 2.99792458e8 * 1e15
    omega0 = TWOPI / t_cycle_fs * AU_TO_FS
    out = {
        "model": model, "kgrid": "2x2", "lambda_nm": 1600.0, "I_Wcm2": 1.0e10,
        "omega0_au": float(omega0), "T_cycle_fs": float(t_cycle_fs),
        "t_max_fs": float(t_p[-1]), "dt_fs": dt_fs, "nt": int(len(t_p)),
        "harmonics": {},
    }
    for q in ORDERS:
        omega_q = q * omega0
        variants = {}
        for name, trim in (("full_hann", 0.0), ("trim16", 0.16)):
            axp, ayp = exact_freq_integral(t_p, jx_p, jy_p, omega_q, trim)
            axm, aym = exact_freq_integral(t_m, jx_m, jy_m, omega_q, trim)
            oddx = 0.5 * (axp - axm)
            oddy = 0.5 * (ayp - aym)
            evenx = 0.5 * (axp + axm)
            eveny = 0.5 * (ayp + aym)
            variants[name] = {
                "odd": {
                    "jx": cinfo(oddx), "jy": cinfo(oddy),
                    "amp2d": float(math.hypot(abs(oddx), abs(oddy))),
                },
                "even": {
                    "jx": cinfo(evenx), "jy": cinfo(eveny),
                    "amp2d": float(math.hypot(abs(evenx), abs(eveny))),
                },
            }
        full = variants["full_hann"]
        trim16 = variants["trim16"]
        rel = None
        if full["odd"]["amp2d"] > 0:
            rel = abs(trim16["odd"]["amp2d"] - full["odd"]["amp2d"]) / full["odd"]["amp2d"]
        dphi = None
        a = full["odd"]["jx"]["phase_deg"]
        b = trim16["odd"]["jx"]["phase_deg"]
        if a is not None and b is not None:
            dphi = (b - a + 180.0) % 360.0 - 180.0
        oee = (full["odd"]["amp2d"] / full["even"]["amp2d"]
               if full["even"]["amp2d"] > 0 else None)
        out["harmonics"][str(q)] = {
            "odd_amp2d": full["odd"]["amp2d"],
            "even_amp2d": full["even"]["amp2d"],
            "odd_over_even": oee,
            "odd_jx_phase_deg": full["odd"]["jx"]["phase_deg"],
            "window_rel_odd_amp": rel,
            "window_dphi_jx_odd_deg": dphi,
            "odd_jy_over_jx_abs": (
                full["odd"]["jy"]["abs"] / full["odd"]["jx"]["abs"]
                if full["odd"]["jx"]["abs"] > 0 else None
            ),
        }
    return out


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return float(a / b)


def write_reports(payload: dict) -> None:
    md = DOCS / "线Q_B1_1600nm模型对照_一页_20261029.md"
    rows = payload.get("models", {})
    lines = [
        "# 线 Q：B1 1600 nm 2×2 模型对照一页（2026-10-29）",
        "",
        "同一相干光协议：1600 nm、25 fs、10^10 W/cm^2、lg_cov、T2=0.5 cycle。",
        "M52 复用 10 月 28 日结果，不重跑。不是完整 BZ，不是量子光。",
        "",
        "| 模型 | H2 奇振幅 | H2 奇/偶 | H2 相位 | 去边窗 | H4 奇振幅 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("m52", "m88", "full112"):
        r = rows.get(name)
        if not r:
            lines.append(f"| {name} | 未完成 | | | | |")
            continue
        h2 = r["harmonics"]["2"]
        h4 = r["harmonics"].get("4", {})
        lines.append(
            f"| {name} | {h2['odd_amp2d']:.3e} | {h2['odd_over_even']:.4g} | "
            f"{h2['odd_jx_phase_deg']:.1f}° | {h2['window_rel_odd_amp']:.3f} | "
            f"{h4.get('odd_amp2d', float('nan')):.3e} |"
        )
    m52, m88, f112 = rows.get("m52"), rows.get("m88"), rows.get("full112")
    lines += ["", "## 判定", ""]
    if m52 and m88 and f112:
        r_m88 = ratio(m88["harmonics"]["2"]["odd_amp2d"], m52["harmonics"]["2"]["odd_amp2d"])
        r_112 = ratio(f112["harmonics"]["2"]["odd_amp2d"], m52["harmonics"]["2"]["odd_amp2d"])
        r_88112 = ratio(m88["harmonics"]["2"]["odd_amp2d"], f112["harmonics"]["2"]["odd_amp2d"])
        lines.append(f"- H2 奇振幅比 M88/M52 = {r_m88:.3g}，full112/M52 = {r_112:.3g}，M88/full112 = {r_88112:.3g}。")
        missing = (r_m88 is not None and (r_m88 < 0.4 or r_m88 > 2.5)) or (
            r_112 is not None and (r_112 < 0.4 or r_112 > 2.5))
        if missing:
            lines.append("- **倾向缺通道/轨道空间差异**，不是单纯 2×2 网格噪声。后续材料线用 M88 或 full112，不加密 M52。")
        else:
            lines.append("- 2×2 上三个模型 H2 奇振幅同量级。M52 在该代表网格上未显示决定性通道缺失；定量仍未过关。")
        lines.append("- 两畴约 180° 是 AFM SHG 对称性，不是量子光效应。")
    else:
        lines.append(payload.get("verdict_text", "部分完成，见 JSON。"))
    lines += ["", f"归档：`{OUT.as_posix()}`。", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "b1_compare.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["one_page"] = str(md)


def prepare_runs() -> list[tuple[str, Path]]:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [k for k, p in TB.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"tb missing: { {k: str(TB[k]) for k in missing} }")
    jobs = []
    for model, spec in MODELS.items():
        for side, tbkey in (("plusN", spec["plus"]), ("minusN", spec["minus"])):
            rundir = OUT / f"{model}_{side}"
            rundir.mkdir(parents=True, exist_ok=True)
            (rundir / "input.nml").write_text(
                nml_text(TB[tbkey], spec["nv"], spec["nb_end"], f"{model} {side}"),
                encoding="utf-8",
            )
            jobs.append((f"{model}_{side}", rundir))
    return jobs


def main() -> int:
    if not EXE.is_file():
        raise FileNotFoundError(EXE)
    jobs = prepare_runs()
    recs = {}
    # two at a time: plus/minus of one model
    for i in range(0, len(jobs), 2):
        chunk = jobs[i:i + 2]
        procs = []
        for name, rundir in chunk:
            jt = rundir / "Jt.dat"
            if jt.is_file() and jt.stat().st_size > 1000:
                recs[name] = {"status": "ok", "skipped": True, "rundir": str(rundir)}
                progress(f"skip existing {name}")
                continue
            env = sbe_env()
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
    models = {}
    try:
        models["m52"] = analyze_pair(M52_OLD / "plusN", M52_OLD / "minusN", "m52")
    except Exception as exc:
        progress(f"m52 analyze failed {exc}")
    for model in ("m88", "full112"):
        plus = OUT / f"{model}_plusN"
        minus = OUT / f"{model}_minusN"
        if recs.get(f"{model}_plusN", {}).get("status") == "ok" and recs.get(f"{model}_minusN", {}).get("status") == "ok":
            try:
                models[model] = analyze_pair(plus, minus, model)
            except Exception as exc:
                progress(f"{model} analyze failed {exc}")
        else:
            progress(f"{model} incomplete, skip analyze")
    payload = {
        "protocol": "1600nm_25fs_1e10_lg_cov_2x2",
        "jobs": recs,
        "models": models,
        "tb_paths": {k: str(v) for k, v in TB.items()},
        "did_not_copy_tb": True,
        "did_not_rerun_m52": True,
        "not_bz_converged": True,
        "not_quantum_light": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_reports(payload)
    progress(f"report {payload.get('one_page')}")
    fails = [k for k, v in recs.items() if v.get("status") != "ok"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
