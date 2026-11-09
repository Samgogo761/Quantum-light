#!/usr/bin/env python3
"""Extract Stage-1E tables and probe6 I(t) figure. Diagnosis only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import propagate_cjj
from q8_m0_classical import p1_a2_protocol
from q8_m0_gsheet import gamma_self
from q8_m0_model import R0, build_geometry
from q8_m0_runaway import k_sets, propagate_seed_self, window_report
from q8_m0_volterra import h2_kernels, volterra_forward

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1e_runaway_20260921"
JSON_PATH = OUT / "stage1e_runaway.json"
MU = -0.009010620372198276
NTPC = 400
EPS = 3.0e-5


def dump_tables() -> dict:
    d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for name, ks in d["k_sets"].items():
        for r in ks["linear"]:
            ib = r["I_bands"]
            ab = r["A_self_bands"]
            rows.append({
                "kset": name,
                "kind": "linear",
                "seed": "H2_Re",
                "sc": r["gamma_scale"],
                "W_el": r["W_el"],
                "W_in": r["W_in"],
                "W_self": r["W_self"],
                "sign": r["W_el_sign"],
                "maxabs_I": r["maxabs_I"],
                "maxabs_I_full": r.get("maxabs_I_full"),
                "dc_I": ib["frac_dc_below_0.25_omega0"],
                "h2_I": ib["frac_H2_1.5_to_2.5_omega0"],
                "peak_over_w0": ib["peak_over_omega0"],
                "dc_A": ab["frac_dc_below_0.25_omega0"],
                "ramp": r["dc_ramp"]["ramp_over_A_self_end"],
                "t_halt": r["t_halt"],
                "early_halt": r["halted_before_end"],
                "Gamma": r["growth"]["Gamma"],
            })
        for r in ks["nonlinear"]:
            ib = r["I_bands"]
            rows.append({
                "kset": name,
                "kind": "nonlinear",
                "seed": r["seed"],
                "sc": r["gamma_scale"],
                "W_el": r["W_el"],
                "W_in": r["W_in"],
                "W_self": r["W_self"],
                "sign": r["W_el_sign"],
                "maxabs_I": r["nl_maxabs_dI"],
                "maxabs_I_full": r["nl_maxabs_dI"],
                "dc_I": ib["frac_dc_below_0.25_omega0"],
                "h2_I": ib["frac_H2_1.5_to_2.5_omega0"],
                "peak_over_w0": ib["peak_over_omega0"],
                "dc_A": r["A_self_bands"]["frac_dc_below_0.25_omega0"],
                "ramp": r["dc_ramp"]["ramp_over_A_self_end"],
                "t_halt": r["t_halt"],
                "early_halt": r["nl_halted"],
                "Gamma": r["growth"]["Gamma"],
            })
    (OUT / "stage1e_table.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return d


def probe6_trajectories():
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    kf = k_sets(geo, R0, MU)["probe6"]["kf"]
    nt = prot.n_cycles * NTPC
    dt = prot.period / NTPC
    times = np.arange(nt + 1) * dt
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    F = hk["F"]
    e_perp = hk["e_perp"]
    print("[extract] CJJ probe6", flush=True)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    x = r0["X_full"]
    w = r0["w_k"]
    mean_d = r0["mean_D_t"]
    i_p1 = np.asarray(r0["mu_per_cell"])
    h_re = np.real(F)[:, None] * e_perp[None, :]
    peak = float(np.max(np.abs(h_re)))
    a_in = (EPS / max(peak, 1e-30)) * h_re
    series = {"times": times, "a_in": a_in, "I_p1": i_p1}
    for sc in (0.0, 0.03, 1.0):
        print(f"[extract] linear sc={sc}", flush=True)
        vol = volterra_forward(x, w, mean_d, dt, sc * gamma, a_L_in=a_in)
        series[f"I_lin_sc{sc}"] = vol["I"]
        series[f"finite_lin_sc{sc}"] = np.array([1.0 if vol["finite"] else 0.0])
        series[f"maxabs_lin_sc{sc}"] = np.array([vol["maxabs_I"]])
    for sc in (0.0, 0.03, 1.0):
        print(f"[extract] nl H2 sc={sc}", flush=True)
        run = propagate_seed_self(
            geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=NTPC,
            seed=a_in, i_ref=i_p1, gamma=sc * gamma, verbose=False,
        )
        series[f"dI_nl_sc{sc}"] = run["dI"]
        series[f"A_self_nl_sc{sc}"] = run["A_self"]
        series[f"halted_nl_sc{sc}"] = np.array([1.0 if run["halted"] else 0.0])
    np.savez(OUT / "probe6_trajectories.npz", **series)
    return series


def make_figure(series: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(series["times"])
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=False)
    ax = axes[0, 0]
    for sc, ls in ((0.0, "-"), (0.03, "--"), (1.0, ":")):
        il = np.linalg.norm(np.asarray(series[f"I_lin_sc{sc}"]), axis=1)
        il = np.where(np.isfinite(il), il, np.nan)
        ax.semilogy(t, np.maximum(il, 1e-30), ls, label=f"linear sc={sc}")
    ax.set_ylabel(r"$||I(t)||$")
    ax.set_title("probe6 linear Volterra")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for sc, ls in ((0.0, "-"), (0.03, "--"), (1.0, ":")):
        di = np.linalg.norm(np.asarray(series[f"dI_nl_sc{sc}"]), axis=1)
        ax.semilogy(t, np.maximum(di, 1e-30), ls, label=f"nl sc={sc}")
    ax.set_title(r"probe6 nonlinear $H[k+A_{\mathrm{cl}}+A_{\mathrm{self}}]$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    il0 = np.linalg.norm(np.asarray(series["I_lin_sc0.0"]), axis=1)
    nl0 = np.linalg.norm(np.asarray(series["dI_nl_sc0.0"]), axis=1)
    il03 = np.linalg.norm(np.asarray(series["I_lin_sc0.03"]), axis=1)
    nl03 = np.linalg.norm(np.asarray(series["dI_nl_sc0.03"]), axis=1)
    ax.plot(t, il0, label="lin sc=0")
    ax.plot(t, nl0, "--", label="nl sc=0")
    ax.plot(t, il03, label="lin sc=0.03")
    ax.plot(t, nl03, "--", label="nl sc=0.03")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$||I||$ or $||\delta I||$")
    ax.set_title("weak coupling: linear vs nonlinear overlay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    as1 = np.linalg.norm(np.asarray(series["A_self_nl_sc1.0"]), axis=1)
    ax.plot(t, as1, label=r"$||A_{\mathrm{self}}||$ nl sc=1")
    ax.axhline(0.8, color="k", ls=":", label="P1 A*=0.8")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$||A_{\mathrm{self}}||$")
    ax.set_title("self-field vs frozen P1 scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Stage-1E probe6: same physics, no T2 / no DC cut / no P1 change",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "stage1e_probe6_trajectories.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dump_tables()
    series = probe6_trajectories()
    make_figure(series)
    print(json.dumps({
        "table": str(OUT / "stage1e_table.json"),
        "traj": str(OUT / "probe6_trajectories.npz"),
        "fig": str(OUT / "stage1e_probe6_trajectories.png"),
        "maxabs_lin_sc1": float(series["maxabs_lin_sc1.0"][0]),
        "finite_lin_sc1": bool(series["finite_lin_sc1.0"][0]),
        "maxabs_nl_sc1": float(np.max(np.abs(series["dI_nl_sc1.0"]))),
        "max_Aself_nl_sc1": float(np.max(np.abs(series["A_self_nl_sc1.0"]))),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
