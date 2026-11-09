#!/usr/bin/env python3
"""Stage-1F: stock ledger/unit/spectrum correction + probe6 ±ε and dt recheck.

Same physics. No T2, no DC cut, no P1 change, no BZ, no Fock expansion.
Old 1E files are left untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_runaway import (  # noqa: E402
    RECHECK_ID,
    a_priori_windows,
    amplitude_compare,
    band_power,
    band_power_unweighted_legacy,
    differential_energy_ledger,
    interval_report,
    k_sets,
    kappa_p1_series,
    propagate_seed_self,
    split_work,
    split_work_interval,
)
from q8_m0_stage1 import stage1_spec  # noqa: E402
from q8_m0_volterra import h2_kernels, volterra_forward  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "results/q_line/stage1e_runaway_20260921"
OUT = REPO / "results/q_line/stage1f_recheck_20260922"
MU = -0.009010620372198276
EPS = 3.0e-5
EPS_SMALL = 3.0e-6
NTPC_LIST = (400, 800)
ZERO_SEED_FAIL = 1.0e-3
ZERO_SEED_TIGHT = 1.0e-6
DT_RECHECK_1600 = 5.0e-2


def _num(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _win_maxabs(a, n: int) -> float:
    a = np.asarray(a)
    sl = a[: n + 1]
    if sl.size == 0:
        return 0.0
    return float(np.max(np.abs(sl)))


def _win_maxmod(a, n: int) -> float:
    a = np.asarray(a, dtype=float)
    sl = a[: n + 1]
    if sl.ndim == 1:
        return float(np.max(np.abs(sl)))
    return float(np.max(np.linalg.norm(sl, axis=1)))


def _rel(num, den) -> float:
    return float(np.max(np.abs(num)) / max(float(np.max(np.abs(den))), 1e-30))


def _cut(a, n: int):
    return np.asarray(a)[: n + 1]


def old_json_residuals() -> dict:
    path = OLD / "stage1e_runaway.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, ks in d["k_sets"].items():
        rows = []
        for r in ks["linear"]:
            wel, win, wsf = r["W_el"], r["W_in"], r["W_self"]
            resid = wel - win - wsf
            rows.append({
                "kind": "linear",
                "sc": r["gamma_scale"],
                "W_el": wel,
                "W_in": win,
                "W_self": wsf,
                "W_residual": resid,
                "abs_resid_over_abs_Wself": abs(resid) / max(abs(wsf), 1e-30),
                "abs_resid_over_abs_Win": abs(resid) / max(abs(win), 1e-30),
                "t_halt_is_diag_window": r["t_halt"],
                "early_halt_field": r.get("halted_before_end"),
            })
        for r in ks["nonlinear"]:
            wel, win, wsf = r["W_el"], r["W_in"], r["W_self"]
            resid = wel - win - wsf
            rows.append({
                "kind": "nonlinear",
                "seed": r["seed"],
                "sc": r["gamma_scale"],
                "W_el": wel,
                "W_in": win,
                "W_self": wsf,
                "W_residual": resid,
                "t_diag_window": r["t_halt"],
                "n_prop_halted": r.get("nl_halted"),
                "note": "t_halt was the diagnostic window; nl_halted is propagation",
            })
        out[name] = rows
    return out


def stock_probe6(geo, prot, gamma: float) -> dict:
    npz_path = OLD / "probe6_trajectories.npz"
    z = np.load(npz_path)
    times = np.asarray(z["times"])
    dt = float(times[1] - times[0])
    a_in = np.asarray(z["a_in"])
    kappa_p1 = kappa_p1_series(times, geo, prot)
    omega0 = prot.omega0
    a_self_nl1 = np.asarray(z["A_self_nl_sc1.0"])
    di_nl1 = np.asarray(z["dI_nl_sc1.0"])
    i_lin0 = np.asarray(z["I_lin_sc0.0"])
    i_lin03 = np.asarray(z["I_lin_sc0.03"])
    i_lin1 = np.asarray(z["I_lin_sc1.0"])
    amp = amplitude_compare(a_self_nl1, kappa_p1, geo, prot.a0_dimless)

    def bands(sig, tag):
        new = band_power(sig, dt, omega0)
        old = band_power_unweighted_legacy(sig, dt, omega0)
        return {
            "tag": tag,
            "new_low": new["frac_low_below_0.25_omega0"],
            "old_low": old["frac_low_below_0.25_omega0"],
            "strict_dc_bin": new["frac_strict_dc_bin"],
            "new_h2": new["frac_H2_1.5_to_2.5_omega0"],
        }

    def ledger(i, ain, g, tag):
        old = split_work(ain, i, dt, g)
        new = split_work_interval(ain, i, dt, g)
        return {
            "tag": tag,
            "legacy_W_el": old["W_el"],
            "legacy_W_in": old["W_in"],
            "legacy_W_self": old["W_self"],
            "legacy_residual": old["W_residual"],
            "new_W_el": new["W_el"],
            "new_W_in": new["W_in"],
            "new_W_self": new["W_self"],
            "new_residual": new["W_residual"],
            "new_Phi": new["Phi_out_minus_in"],
            "new_Phi_plus_W_el": new["Phi_plus_W_el"],
        }

    n190 = int(np.searchsorted(times, 190.066, side="right") - 1)
    return {
        "npz": str(npz_path),
        "dt": dt,
        "units": amp,
        "bands": [
            bands(i_lin0, "I_lin_sc0"),
            bands(i_lin03, "I_lin_sc0.03"),
            bands(np.asarray(z["A_self_nl_sc0.03"]), "A_self_nl_sc0.03"),
            bands(a_self_nl1, "A_self_nl_sc1"),
            bands(di_nl1, "dI_nl_sc1"),
        ],
        "ledgers": [
            ledger(i_lin0, a_in, 0.0, "lin_sc0_full"),
            ledger(i_lin03, a_in, 0.03 * gamma, "lin_sc0.03_full"),
            ledger(i_lin1[: n190 + 1], a_in[: n190 + 1], gamma, "lin_sc1_old_diag_window"),
            ledger(di_nl1, a_in, gamma, "nl_sc1_full"),
            ledger(di_nl1[: n190 + 1], a_in[: n190 + 1], gamma, "nl_sc1_old_diag_window"),
        ],
        "kappa_p1_peak_mod": amp["kappa_p1_peak_mod_ang_inv"],
        "kappa_self_nl1_peak_mod": amp["kappa_self_peak_mod_ang_inv"],
        "ratio_self_over_p1": amp["ratio_kappa_self_mod_over_p1"],
    }


def make_stock_figure(geo, prot, stock: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(OLD / "probe6_trajectories.npz")
    t = np.asarray(z["times"])
    kap_p1 = kappa_p1_series(t, geo, prot)
    a_self = np.asarray(z["A_self_nl_sc1.0"])
    a_nn = float(geo.a_nn_mean_ang)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    ax = axes[0, 0]
    ax.plot(t, np.linalg.norm(kap_p1, axis=1), label=r"$||\kappa_{\mathrm{P1}}||$")
    ax.plot(t, np.linalg.norm(a_self, axis=1), label=r"$||\kappa_{\mathrm{self}}||$")
    ax.set_ylabel(r"$\mathrm{\AA}^{-1}$")
    ax.set_title("same unit: kappa")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, a_nn * np.linalg.norm(kap_p1, axis=1), label=r"$a_{\mathrm{NN}}||\kappa_{\mathrm{P1}}||$")
    ax.plot(t, a_nn * np.linalg.norm(a_self, axis=1), label=r"$a_{\mathrm{NN}}||\kappa_{\mathrm{self}}||$")
    ax.axhline(0.8, color="k", ls=":", label="protocol A*=0.8")
    ax.set_title("same unit: A*")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for sc, ls in ((0.0, "-"), (0.03, "--"), (1.0, ":")):
        il = np.linalg.norm(np.asarray(z[f"I_lin_sc{sc}"]), axis=1)
        ax.semilogy(t, np.maximum(np.where(np.isfinite(il), il, np.nan), 1e-30), ls, label=f"lin sc={sc}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$||I||$")
    ax.set_title("linear (unchanged trajectories)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = [r["tag"] for r in stock["ledgers"]]
    oldr = [abs(r["legacy_residual"]) for r in stock["ledgers"]]
    newr = [abs(r["new_residual"]) for r in stock["ledgers"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, oldr, 0.4, label="legacy residual")
    ax.bar(x + 0.2, newr, 0.4, label="interval residual")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_title(r"$|W_{\mathrm{el}}-W_{\mathrm{in}}-W_{\mathrm{self}}|$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Stage-1F stock correction: units and ledger (no new physics)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stage1f_stock_units_ledger.png", dpi=140)
    plt.close(fig)


def h2_seed(times, prot, geo, eps: float):
    hk = h2_kernels(times, prot.omega0, prot.t_pulse, geo)
    F = hk["F"]
    e_perp = hk["e_perp"]
    h = np.real(F)[:, None] * e_perp[None, :]
    peak = float(np.max(np.abs(h)))
    a_in = (eps / max(peak, 1e-30)) * h
    return a_in, hk


def slim_interval(rep: dict) -> dict:
    out = dict(rep)
    for key in ("I_bands", "A_self_bands"):
        if key in out and isinstance(out[key], dict):
            out[key] = {k: v for k, v in out[key].items() if k != "mean"}
    return out


def window_block(times, period, i, a_self, i_ref=None):
    wins = a_priori_windows(times, period)
    out = {}
    for name, n in wins.items():
        rec = {
            "n": n,
            "t": float(times[n]),
            "maxabs_I": _win_maxabs(i, n),
            "maxmod_A_self": _win_maxmod(a_self, n),
        }
        if i_ref is not None:
            rec["maxabs_dI"] = _win_maxabs(np.asarray(i) - np.asarray(i_ref), n)
        out[name] = rec
    return out


def run_linear(geo, prot, kf, gamma, ntpc, a_in):
    print(f"  CJJ X+D ntpc={ntpc} nk={len(kf)}", flush=True)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=ntpc, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    vol0 = volterra_forward(r0["X_full"], r0["w_k"], r0["mean_D_t"], r0["dt"], 0.0, a_L_in=a_in)
    vol1 = volterra_forward(r0["X_full"], r0["w_k"], r0["mean_D_t"], r0["dt"], gamma, a_L_in=a_in)
    return {
        "times": r0["times"],
        "dt": float(r0["dt"]),
        "I_vv": np.asarray(r0["mu_per_cell"]),
        "I_rho": np.asarray(r0["J_sheet"]) * float(r0["a_cell_ang2"]),
        "vv_rho_maxabs": float(r0["mean_vs_rho_maxabs"]),
        "I_lin0": vol0["I"],
        "I_lin1": vol1["I"],
        "A_self_lin1": vol1["A_self"],
        "lin1_finite": bool(vol1["finite"]),
        "lin1_maxabs_I": vol1["maxabs_I"],
        "cjj_wall": float(r0["wall_seconds"]),
    }


def run_nl(geo, prot, kf, gamma, ntpc, seed, i_ref, tag: str):
    print(f"  nl {tag} ntpc={ntpc}", flush=True)
    return propagate_seed_self(
        geo, R0, prot, kf=kf, mu=MU, neel=+1, nt_per_cycle=ntpc,
        seed=seed, i_ref=i_ref, gamma=gamma, verbose=False,
    )


def one_resolution(geo, prot, kf, gamma, ntpc: int, eps: float, eps_small: float | None):
    nt = prot.n_cycles * ntpc
    dt = prot.period / ntpc
    times = np.arange(nt + 1) * dt
    a_in, _hk = h2_seed(times, prot, geo, eps)
    lin = run_linear(geo, prot, kf, gamma, ntpc, a_in)
    i_vv = lin["I_vv"]
    i0 = run_nl(geo, prot, kf, 0.0, ntpc, np.zeros_like(a_in), i_vv, "I0_gamma0_rho")
    # Same-representation baseline: rho current with seed=0, gamma=0.
    i_ref = i0["I"]
    z = run_nl(geo, prot, kf, gamma, ntpc, np.zeros_like(a_in), i_ref, "seed0_physical")
    p = run_nl(geo, prot, kf, gamma, ntpc, a_in, i_ref, "plus_eps")
    m = run_nl(geo, prot, kf, gamma, ntpc, -a_in, i_ref, "minus_eps")
    rel0 = _rel(z["I"] - i_ref, i_ref)
    kappa_p1 = i0["kappa_p1"]
    amp_p = amplitude_compare(p["A_self"], kappa_p1, geo, prot.a0_dimless)
    led_p = split_work_interval(a_in, p["dI"], dt, gamma)
    eled_p = differential_energy_ledger(
        p["U"], i0["U"], kappa_p1, a_in, p["A_self"], p["I"], i_ref, dt,
    )
    odd = 0.5 * (p["I"] - m["I"])
    even = 0.5 * (p["I"] + m["I"]) - z["I"]
    wins = a_priori_windows(times, prot.period)
    win_cmp = {}
    for name, n in wins.items():
        lin_w = _cut(lin["I_lin1"], n)
        odd_w = _cut(odd, n)
        even_w = _cut(even, n)
        win_cmp[name] = {
            "t": float(times[n]),
            "maxabs_odd": float(np.max(np.abs(odd_w))),
            "maxabs_even": float(np.max(np.abs(even_w))),
            "maxabs_lin1": float(np.max(np.abs(lin_w))) if np.isfinite(lin_w).all() else None,
            "odd_minus_lin_rel": _rel(odd_w - lin_w, odd_w) if np.isfinite(lin_w).all() else None,
            "even_over_odd": float(np.max(np.abs(even_w)) / max(np.max(np.abs(odd_w)), 1e-30)),
            "maxmod_A_self_plus": _win_maxmod(p["A_self"], n),
            "maxabs_dI_plus": _win_maxabs(p["dI"], n),
        }
    small = None
    if eps_small is not None and rel0 <= ZERO_SEED_FAIL:
        a_s, _ = h2_seed(times, prot, geo, eps_small)
        ps = run_nl(geo, prot, kf, gamma, ntpc, a_s, i_ref, "plus_eps_small")
        ms = run_nl(geo, prot, kf, gamma, ntpc, -a_s, i_ref, "minus_eps_small")
        odd_s = 0.5 * (ps["I"] - ms["I"])
        even_s = 0.5 * (ps["I"] + ms["I"]) - z["I"]
        small_cmp = {}
        for name, n in wins.items():
            o = _cut(odd, n) / eps
            os_ = _cut(odd_s, n) / eps_small
            small_cmp[name] = {
                "t": float(times[n]),
                "odd_over_eps_rel": _rel(o - os_, os_),
                "maxabs_odd_small": float(np.max(np.abs(_cut(odd_s, n)))),
                "even_over_odd_small": float(
                    np.max(np.abs(_cut(even_s, n))) / max(np.max(np.abs(_cut(odd_s, n))), 1e-30)
                ),
                "maxmod_A_self_small": _win_maxmod(ps["A_self"], n),
            }
        small = {
            "eps": eps_small,
            "windows": small_cmp,
            "maxabs_dI": ps["maxabs_dI"],
            "maxmod_A_self": ps["maxmod_A_self"],
            "prop_halted": ps["halted"],
            "t_prop_halt": ps["t_prop_halt"],
            "energy": differential_energy_ledger(
                ps["U"], i0["U"], kappa_p1, a_s, ps["A_self"], ps["I"], i_ref, dt,
            ),
            "ledger": slim_interval(interval_report(ps["dI"], a_s, times, dt, gamma, prot.omega0)),
        }
        small_ser = {
            "odd_small": odd_s,
            "even_small": even_s,
            "I_plus_small": ps["I"],
            "A_self_plus_small": ps["A_self"],
        }
    else:
        small_ser = {}

    series = {
        "times": times,
        "a_in": a_in,
        "I0": i_ref,
        "I_vv": i_vv,
        "I_seed0": z["I"],
        "I_plus": p["I"],
        "I_minus": m["I"],
        "A_self_plus": p["A_self"],
        "A_self_minus": m["A_self"],
        "odd": odd,
        "even": even,
        "I_lin0": lin["I_lin0"],
        "I_lin1": lin["I_lin1"],
        "U0": i0["U"],
        "U_plus": p["U"],
        "kappa_p1": kappa_p1,
        **small_ser,
    }
    summary = {
        "nt_per_cycle": ntpc,
        "dt": dt,
        "eps": eps,
        "zero_seed_maxabs_dI_over_I0": rel0,
        "zero_seed_maxabs_dI": float(np.max(np.abs(z["I"] - i_ref))),
        "vv_minus_rho_I0_rel": _rel(i_vv - i_ref, i_ref),
        "cjj_vv_rho_maxabs": lin["vv_rho_maxabs"],
        "zero_seed_pass_tight": bool(rel0 < ZERO_SEED_TIGHT),
        "zero_seed_pass": bool(rel0 <= ZERO_SEED_FAIL),
        "plus_prop_halted": p["halted"],
        "plus_t_prop_halt": p["t_prop_halt"],
        "minus_prop_halted": m["halted"],
        "lin1_finite": lin["lin1_finite"],
        "lin1_maxabs_I": lin["lin1_maxabs_I"],
        "units_plus": amp_p,
        "windows": win_cmp,
        "windows_plus": window_block(times, prot.period, p["I"], p["A_self"], i_ref),
        "ledger_plus": slim_interval(interval_report(p["dI"], a_in, times, dt, gamma, prot.omega0)),
        "energy_plus": eled_p,
        "Phi_plus_W_el": led_p["Phi_plus_W_el"],
        "W_residual_plus": led_p["W_residual"],
        "small": small,
        "cjj_wall": lin["cjj_wall"],
        "nl_wall_plus": p["wall_seconds"],
        "born_status": "reference_diagnostic_not_production_V",
        "gain_claim": "not_evaluated",
    }
    return summary, series


def compare_dt(s400: dict, s800: dict, times400, period: float) -> dict:
    wins = a_priori_windows(times400, period)
    i4 = s400["I_plus"]
    i8 = s800["I_plus"][::2]
    d4 = s400["odd"]
    d8 = s800["odd"][::2]
    a4 = s400["A_self_plus"]
    a8 = s800["A_self_plus"][::2]
    out = {}
    for name, n in wins.items():
        out[name] = {
            "rel_I_plus": _rel(_cut(i4, n) - _cut(i8, n), _cut(i8, n)),
            "rel_odd": _rel(_cut(d4, n) - _cut(d8, n), _cut(d8, n)),
            "rel_A_self": _rel(_cut(a4, n) - _cut(a8, n), _cut(a8, n)),
            "maxabs_odd_800": float(np.max(np.abs(_cut(d8, n)))),
        }
    cycle1_err = out["cycle1"]["rel_odd"]
    return {
        "windows": out,
        "cycle1_odd_rel_err": cycle1_err,
        "need_ntpc_1600": bool(cycle1_err > DT_RECHECK_1600),
        "note": "800 sampled on even nodes to match 400. A priori windows, not amplitude halt.",
    }


def make_reprop_figure(ser400, ser800, period: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t4 = ser400["times"]
    t8 = ser800["times"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    ax = axes[0, 0]
    ax.plot(t4, np.linalg.norm(ser400["odd"], axis=1), label="odd 400")
    ax.plot(t8, np.linalg.norm(ser800["odd"], axis=1), "--", label="odd 800")
    ax.axvline(period, color="k", ls=":", lw=0.8)
    ax.axvline(2 * period, color="k", ls=":", lw=0.8)
    ax.set_title(r"$||[I(+)-I(-)]/2||$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for lab, ser, t in (("400", ser400, t4), ("800", ser800, t8)):
        ax.semilogy(t, np.maximum(np.linalg.norm(ser["I_lin1"], axis=1), 1e-30), label=f"lin {lab}")
        ax.semilogy(t, np.maximum(np.linalg.norm(ser["odd"], axis=1), 1e-30), "--", label=f"odd {lab}")
    ax.set_title("linear vs odd (physical gamma)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t4, np.linalg.norm(ser400["I_seed0"] - ser400["I0"], axis=1), label="seed0-I0 400")
    ax.plot(t8, np.linalg.norm(ser800["I_seed0"] - ser800["I0"], axis=1), "--", label="seed0-I0 800")
    ax.set_xlabel("t")
    ax.set_title("zero-seed residual (same rho baseline)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    a_nn = None
    ax.plot(t4, np.linalg.norm(ser400["A_self_plus"], axis=1), label=r"$||\kappa_{\mathrm{self}}||$ + 400")
    ax.plot(t8, np.linalg.norm(ser800["A_self_plus"], axis=1), "--", label="+ 800")
    ax.plot(t4, np.linalg.norm(ser400["kappa_p1"], axis=1), "k:", label=r"$||\kappa_{\mathrm{P1}}||$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathrm{\AA}^{-1}$")
    ax.set_title("self-field vs P1, same unit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    del a_nn

    fig.suptitle("Stage-1F probe6 physical gamma: 0/+/− eps and dt", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stage1f_probe6_pm_dt.png", dpi=140)
    plt.close(fig)


def maybe_ntpc1600(geo, prot, kf, gamma, dt_cmp, sum400) -> dict | None:
    if not dt_cmp["need_ntpc_1600"]:
        return {"ran": False, "reason": "cycle1_odd_rel_err_within_5pct"}
    if not sum400["zero_seed_pass"]:
        return {"ran": False, "reason": "zero_seed_failed_no_further_dt"}
    print("[stage1f] extra ntpc=1600 because 400 vs 800 cycle1 odd err > 5%", flush=True)
    summary, series = one_resolution(geo, prot, kf, gamma, 1600, EPS, None)
    np.savez(OUT / "probe6_ntpc1600.npz", **series)
    return {"ran": True, "summary": summary}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    gamma = gamma_self(geo)
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True
    kf = k_sets(geo, R0, MU)["probe6"]["kf"]

    print("[stage1f] A stock correction", flush=True)
    stock = stock_probe6(geo, prot, gamma)
    stock["old_json_residuals"] = old_json_residuals()
    make_stock_figure(geo, prot, stock)
    dump(OUT / "stage1f_stock_correction.json", {
        "recheck_id": RECHECK_ID,
        "old_dir_kept": str(OLD),
        "stock": stock,
        "language": (
            "finite_time_linear_self_consistent_runaway_amplification_"
            "not_a_statement_that_the_Volterra_equation_has_no_solution"
        ),
        "gain_claim": "not_evaluated",
        "production_V": "Born_reference_diagnostic_not_qualified",
    })

    print("[stage1f] B probe6 400 and 800", flush=True)
    sum400, ser400 = one_resolution(geo, prot, kf, gamma, 400, EPS, EPS_SMALL)
    np.savez(OUT / "probe6_ntpc400.npz", **ser400)
    sum800, ser800 = one_resolution(geo, prot, kf, gamma, 800, EPS, EPS_SMALL)
    np.savez(OUT / "probe6_ntpc800.npz", **ser800)
    dt_cmp = compare_dt(ser400, ser800, ser400["times"], prot.period)
    make_reprop_figure(ser400, ser800, prot.period)
    extra = maybe_ntpc1600(geo, prot, kf, gamma, dt_cmp, sum400)

    stop = None
    if not sum400["zero_seed_pass"] or not sum800["zero_seed_pass"]:
        stop = "zero_seed_failed_stop_physics_read_fix_implementation"
    elif dt_cmp["need_ntpc_1600"] and extra and extra.get("ran"):
        stop = "dt_not_clear_after_400_800_ran_1600"
    else:
        stop = "batch_complete_early_amplification_status_in_windows"

    payload = {
        "recheck_id": RECHECK_ID,
        "spec": stage1_spec(),
        "frozen": {
            "no_T2": True,
            "no_DC_subtraction": True,
            "P1_local_classical": True,
            "no_full_BZ": True,
            "no_fock_expansion": True,
            "production_V": "Born_reference_diagnostic_not_qualified",
        },
        "resummation_production_blocked": blocked,
        "gain_claim": "not_evaluated",
        "approximation": (
            "strong_classical_P1_plus_weaker_quantum_perturbation_"
            "weakness_not_guaranteed_after_feedback"
        ),
        "ntpc400": sum400,
        "ntpc800": sum800,
        "dt_compare": dt_cmp,
        "ntpc1600": extra,
        "stop_reason": stop,
        "nk96": "not_run",
        "squeezing_claim": "not_evaluated",
        "classical_seed_is_not_quantum_vacuum": True,
    }
    dump(OUT / "stage1f_recheck.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1f_recheck.json"),
        "zero_seed_400": sum400["zero_seed_maxabs_dI_over_I0"],
        "zero_seed_800": sum800["zero_seed_maxabs_dI_over_I0"],
        "cycle1_odd_rel_err": dt_cmp["cycle1_odd_rel_err"],
        "lin1_maxabs_400": sum400["lin1_maxabs_I"],
        "plus_ratio_self_p1_400": sum400["units_plus"]["ratio_kappa_self_mod_over_p1"],
        "W_residual_plus_400": sum400["W_residual_plus"],
        "energy_resid_400": sum400["energy_plus"]["residual_dU_minus_work"],
        "cycle1_even_over_odd_400": sum400["windows"]["cycle1"]["even_over_odd"],
        "small_cycle1_odd_rel_400": (
            None if sum400["small"] is None else sum400["small"]["windows"]["cycle1"]["odd_over_eps_rel"]
        ),
        "stop_reason": stop,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
