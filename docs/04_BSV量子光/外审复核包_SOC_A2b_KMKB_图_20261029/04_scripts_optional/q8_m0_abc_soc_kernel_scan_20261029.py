"""SOC/Delta scan of the Q6-protocol adiabatic and finite-time odd H2 kernel.

Does not overwrite A1/A2 JSON or markdown. Not chi^(2) certification.
Finite-time reuses A2.run_one with an extended params_ctrl; production
A2 eta1.0 and s0 are reused, not rerun.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import q8_m0_abc_a2_finite_time_kernel_20261029 as a2
import q8_m0_abc_microscopic_kernel_20261029 as adia
import q8_m0_qinput_minimal_20261020 as qmin
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import M0Params, R0, build_geometry, h8

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/abc_soc_kernel_scan_20261029"
FIG = OUT / "figures"
A2_JSON = ROOT / "results/q_line/abc_a2_finite_time_kernel_20261029/a2_finite_time.json"
DOCS = ROOT / "docs/04_BSV量子光"

FT_FRACS = (0.25, 0.50, 0.75)
ADIA_SO_FRACS = tuple(round(x, 4) for x in list(np.linspace(0.0, 1.0, 21)) + [1.25, 1.5, 2.0])
ADIA_DELTA_FRACS = tuple(round(x, 4) for x in np.linspace(0.0, 2.0, 21))
ABS_GATE = 1e-12


def progress(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def params_soc(frac: float) -> M0Params:
    return M0Params(
        t_h=R0.t_h, lam_so=float(frac) * R0.lam_so, delta=R0.delta,
        u=R0.u, v=R0.v, label=f"R0_soc{int(round(100 * frac)):03d}",
    )


def params_delta(frac: float) -> M0Params:
    return M0Params(
        t_h=R0.t_h, lam_so=R0.lam_so, delta=float(frac) * R0.delta,
        u=R0.u, v=R0.v, label=f"R0_d{frac:g}",
    )


def params_ctrl_ext(name: str) -> M0Params:
    if name.startswith("soc") and name[3:].isdigit():
        return params_soc(float(name[3:]) / 100.0)
    return _ORIG_PARAMS_CTRL(name)


def min_gap(geo, p, kcart) -> float:
    gaps = []
    for neel in (1, -1):
        for kk in (kcart, -np.asarray(kcart, float)):
            ev = np.linalg.eigvalsh(h8(geo, p, kk, neel))
            gaps.append(float(ev[4] - ev[3]))
    return float(min(gaps))


def adiabatic_record(geo, p, kcart, ehat, omega, gamma) -> dict:
    rec = adia.odd_even_kernel(
        geo, p, kcart, ehat, np.array([1.0, 0.0]),
        omega + 1j * gamma, omega + 1j * gamma,
    )
    kx = 0.5 * rec["odd"]
    terms = 0.5 * np.asarray(rec["terms_odd"], complex)
    packed = {
        "delta": float(p.delta),
        "lam_so": float(p.lam_so),
        "lam_so_frac": float(p.lam_so / R0.lam_so) if R0.lam_so else 0.0,
        "delta_frac": float(p.delta / R0.delta) if R0.delta else 0.0,
        "label": p.label,
        "min_gap": min_gap(geo, p, kcart),
        "odd_x": a2.cpack(kx),
        "even": a2.cpack(0.5 * rec["even"]),
        "t0": a2.cpack(terms[0]),
        "t1": a2.cpack(terms[1]),
        "t2": a2.cpack(terms[2]),
        "t0_plus_t1": a2.cpack(terms[0] + terms[1]),
        "net_over_t2": float(abs(kx) / max(abs(terms[2]), 1e-30)),
        "gated_off": bool(abs(kx) < ABS_GATE),
    }
    return packed


def load_a2_reuse() -> dict:
    raw = json.loads(A2_JSON.read_text(encoding="utf-8"))
    out = {}
    mapping = {"soc000": "s0", "soc100": "eta1.0"}
    for new, old in mapping.items():
        c = raw["cases"][old]
        out[new] = {
            "reused_from": old,
            "name": new,
            "delta": c["delta"],
            "lam_so": c["lam_so"],
            "lam_so_frac": float(c["lam_so"] / R0.lam_so) if R0.lam_so else 0.0,
            "finite_time_arg_deg": c["finite_time_arg_deg"],
            "adiabatic_arg_deg": c["adiabatic_arg_deg"],
            "odd_H2_x": c["odd_H2_x"],
            "odd_norm2D": c["odd_norm2D"],
            "adiabatic_odd_x": c["adiabatic_odd_x"],
            "finite_minus_adiabatic_deg": c["finite_minus_adiabatic_deg"],
            "gated_off": bool(c["odd_norm2D"] < 1e-10),
            "did_not_rerun": True,
        }
    return out


def write_svg_xy(path: Path, series, title: str, ylabel: str, ymin=None, ymax=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h, x0, y0, bw, bh = 720, 380, 70, 40, 600, 280
    xs, ys = [], []
    for _, pts in series:
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    if ymin is None:
        ymin = min(ys)
    if ymax is None:
        ymax = max(ys)
    if abs(ymax - ymin) < 1e-15:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad

    def xx(x):
        return x0 + bw * (x - xmin) / max(xmax - xmin, 1e-15)

    def yy(y):
        return y0 + bh * (1.0 - (y - ymin) / max(ymax - ymin, 1e-15))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="20" y="24" font-size="15">{title}</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+bh}" stroke="#333"/>',
        f'<line x1="{x0}" y1="{y0+bh}" x2="{x0+bw}" y2="{y0+bh}" stroke="#333"/>',
        f'<text x="16" y="{y0+bh/2:.1f}" font-size="11" transform="rotate(-90 16 {y0+bh/2:.1f})">{ylabel}</text>',
        f'<text x="{x0+bw/2:.1f}" y="{h-12}" text-anchor="middle" font-size="11">lam_so / R0.lam_so</text>',
    ]
    for i, (lab, pts) in enumerate(series):
        col = colors[i % len(colors)]
        if len(pts) >= 2:
            d = " ".join(
                f"{'M' if j == 0 else 'L'}{xx(x):.1f},{yy(y):.1f}" for j, (x, y) in enumerate(pts)
            )
            body.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in pts:
            body.append(f'<circle cx="{xx(x):.1f}" cy="{yy(y):.1f}" r="3.5" fill="{col}"/>')
        body.append(
            f'<text x="{x0+12}" y="{y0+16+16*i}" font-size="12" fill="{col}">{lab}</text>'
        )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def nearest_pi_dev(arg_deg: float) -> float:
    a = ((float(arg_deg) + 180.0) % 360.0) - 180.0
    d0 = abs(a)
    d180 = abs(abs(a) - 180.0)
    return float(min(d0, d180))


def main() -> int:
    global _ORIG_PARAMS_CTRL
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    _ORIG_PARAMS_CTRL = a2.params_ctrl
    a2.params_ctrl = params_ctrl_ext
    a2.progress = progress

    geo = build_geometry()
    kcart = np.asarray(geo.k_cart(sh.K_FRAC), float)
    ehat = qmin.a2_polarization(geo)
    env = __import__("q8_m0_qinput_dynamics_20261026", fromlist=["Envelope"]).Envelope(
        pf.OMEGA0, ramp_cycles=1.0, flat_cycles=4.0
    )
    js = a2.sample_js(a2.N_FULL)
    omega, gamma = pf.OMEGA0, a2.GAMMA_ADIA

    progress("adiabatic SOC scan")
    adia_soc = []
    for frac in ADIA_SO_FRACS:
        p = params_soc(frac)
        rec = adiabatic_record(geo, p, kcart, ehat, omega, gamma)
        rec["scan"] = "lam_so_frac"
        rec["frac"] = float(frac)
        adia_soc.append(rec)
        progress(
            f"adia soc={frac:.3f} abs={rec['odd_x']['abs']:.4e} "
            f"arg={rec['odd_x']['arg_deg']:.2f} gap={rec['min_gap']:.4f}"
        )

    progress("adiabatic Delta scan")
    adia_delta = []
    for frac in ADIA_DELTA_FRACS:
        p = params_delta(frac)
        rec = adiabatic_record(geo, p, kcart, ehat, omega, gamma)
        rec["scan"] = "delta_frac"
        rec["frac"] = float(frac)
        adia_delta.append(rec)

    reused = load_a2_reuse()
    ft_cases = dict(reused)
    for frac in FT_FRACS:
        name = f"soc{int(round(100 * frac)):03d}"
        progress(f"finite-time start {name}")
        rec = a2.run_one(name, geo, kcart, ehat, env, a2.N_FULL, js, a2.EPS)
        rec["lam_so_frac"] = float(frac)
        rec["gated_off"] = bool(rec["odd_norm2D"] < 1e-10)
        rec["did_not_rerun"] = False
        ft_cases[name] = rec
        a2.dump(OUT / f"case_{name}.json", rec)

    live = [c for c in adia_soc if not c["gated_off"]]
    args_live = [c["odd_x"]["arg_deg"] for c in live]
    adia_span = None
    if len(args_live) >= 2:
        diffs = []
        for a in args_live:
            diffs.append(a2.angdiff_deg(a, args_live[0]))
        adia_span = float(max(diffs) - min(diffs))

    ft_live = []
    for key in ("soc025", "soc050", "soc075", "soc100"):
        if key in ft_cases and not ft_cases[key].get("gated_off"):
            ft_live.append((key, ft_cases[key]["finite_time_arg_deg"]))
    ft_span = None
    if len(ft_live) >= 2:
        base = ft_live[-1][1]
        ds = [a2.angdiff_deg(a, base) for _, a in ft_live]
        ft_span = float(max(ds) - min(ds))

    gate = 10.0
    verdict = "kernel_phase_pinned"
    if (adia_span is not None and abs(adia_span) >= gate) or (
        ft_span is not None and abs(ft_span) >= gate
    ):
        verdict = "kernel_phase_moves_ge_10deg"

    rec = {
        "task_tag": "20261029",
        "protocol": "Qin_a2_1_4_1",
        "status": "ok",
        "not_chi2_certification": True,
        "did_not_overwrite_a2": True,
        "R0_lam_so": float(R0.lam_so),
        "R0_delta": float(R0.delta),
        "gamma_adiabatic": gamma,
        "omega0": omega,
        "adiabatic_soc": adia_soc,
        "adiabatic_delta": adia_delta,
        "finite_time": {k: {
            "lam_so": v.get("lam_so"),
            "lam_so_frac": v.get("lam_so_frac"),
            "delta": v.get("delta"),
            "finite_time_arg_deg": v.get("finite_time_arg_deg"),
            "adiabatic_arg_deg": v.get("adiabatic_arg_deg"),
            "odd_norm2D": v.get("odd_norm2D"),
            "odd_H2_x": v.get("odd_H2_x"),
            "adiabatic_odd_x": v.get("adiabatic_odd_x"),
            "finite_minus_adiabatic_deg": v.get("finite_minus_adiabatic_deg"),
            "gated_off": v.get("gated_off"),
            "did_not_rerun": v.get("did_not_rerun"),
            "reused_from": v.get("reused_from"),
            "wall_s": v.get("wall_s"),
            "richardson_vs_fd_rel": v.get("richardson_vs_fd_rel"),
        } for k, v in ft_cases.items()},
        "adiabatic_soc_arg_span_deg": adia_span,
        "finite_time_arg_span_deg": ft_span,
        "move_gate_deg": gate,
        "verdict": verdict,
        "wall_s": time.perf_counter() - t_all,
        "utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Finite-time kernel probes the prescribed-field A^2 / K_B-like channel. "
            "arg(A/B)=arg(K_M/K_B) still requires two-angle GH5."
        ),
    }
    a2.dump(OUT / "soc_kernel_scan.json", rec)

    adia_pts = [(c["frac"], c["odd_x"]["arg_deg"]) for c in adia_soc if not c["gated_off"]]
    ft_pts = []
    for key, v in rec["finite_time"].items():
        if v.get("gated_off"):
            continue
        ft_pts.append((float(v["lam_so_frac"]), float(v["finite_time_arg_deg"])))
    ft_pts.sort()
    write_svg_xy(
        FIG / "arg_vs_soc.svg",
        [("adiabatic arg(K)", adia_pts), ("finite-time arg(B)", ft_pts)],
        "Odd H2 kernel phase vs SOC (Q6 k-pair)",
        "arg (deg)",
    )
    abs_pts = [(c["frac"], c["odd_x"]["abs"]) for c in adia_soc]
    write_svg_xy(
        FIG / "abs_vs_soc.svg",
        [("|K_odd| adiabatic", abs_pts)],
        "Adiabatic odd |K_x| vs SOC",
        "|K|",
        ymin=0.0,
    )

    progress(f"done verdict={verdict} adia_span={adia_span} ft_span={ft_span} wall={rec['wall_s']:.1f}s")
    print(json.dumps({
        "status": rec["status"],
        "verdict": verdict,
        "adiabatic_soc_arg_span_deg": adia_span,
        "finite_time_arg_span_deg": ft_span,
        "ft_args": {k: rec["finite_time"][k]["finite_time_arg_deg"] for k in rec["finite_time"]},
    }, ensure_ascii=False, indent=2))
    return 0


_ORIG_PARAMS_CTRL = a2.params_ctrl


if __name__ == "__main__":
    raise SystemExit(main())
