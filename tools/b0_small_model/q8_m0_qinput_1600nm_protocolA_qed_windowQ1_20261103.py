#!/usr/bin/env python3
"""Q1: independent three-window BAC fits on existing 1600 nm QED (and GH5) series.

No propagator. Does not rewrite fourangle_fit.json. Front/back coefficients
are fitted in those windows; they are not the frozen main-window A/B/C.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import q8_m0_qinput_1600nm_protocolA_qed_postprocess_20261102 as pp
from qinput_spectrum_reanalysis_20261028 import packed, spectrum

ROOT = pp.ROOT
QED = pp.QED_OUT
GH5 = pp.GH5_OUT
FIT = pp.FIT_PATH
OUT = QED / "analysis_windowQ1_20261103"
FROZEN_SHA = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"
ESYM = 0.02
HOLD = 0.25 * np.pi
CHANNELS = (
    ("h2_odd_x", 2, "odd", "x"),
    ("h2_odd_y", 2, "odd", "y"),
    ("h3_odd_x", 3, "odd", "x"),
    ("h2_even_x", 2, "even", "x"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cz(z) -> complex:
    if isinstance(z, dict):
        return complex(float(z["re"]), float(z["im"]))
    return complex(z)


def load_series(folder: Path, tag: str):
    with np.load(folder / f"series_{tag}.npz", allow_pickle=False) as z:
        t = np.asarray(z["t"], float)
        j = np.column_stack([np.asarray(z["Jx"]), np.asarray(z["Jy"])]).astype(complex, copy=False)
    return t, j


def assert_frozen() -> str:
    got = sha256_file(FIT)
    if got != FROZEN_SHA:
        raise RuntimeError(f"frozen fit SHA changed: {got}")
    return got


def pair_currents(folder: Path, ptag: str, mtag: str):
    tp, jp = load_series(folder, ptag)
    tm, jm = load_series(folder, mtag)
    if tp.shape != tm.shape or not np.allclose(tp, tm, atol=0.0, rtol=0.0):
        raise RuntimeError(f"time grid mismatch {ptag}/{mtag}")
    return tp, jp, jm


def z_channel(t, jp, jm, window: str, q: int, parity: str, axis: str, method="linear_exact"):
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    j = jo if parity == "odd" else je
    a, b = pp.WINDOWS[window]
    spec = spectrum(t, j, a * pp.PERIOD, b * pp.PERIOD, np.arange(1, 7, dtype=float), method=method, omega=pp.OMEGA0)
    iq = int(q) - 1
    ia = 0 if axis == "x" else 1
    return complex(spec[iq, ia])


def holdout_metrics(z: complex, pred: complex, A: complex) -> dict:
    delta = z - pred
    denom = max(abs(z), abs(pred))
    esym = None if denom == 0.0 else float(abs(delta) / denom)
    ea = None if abs(A) == 0.0 else float(abs(delta) / abs(A))
    dphi = float(np.degrees(np.angle(z * np.conj(pred)))) if abs(z) and abs(pred) else None
    return {
        "z": packed(z),
        "z_pred": packed(pred),
        "delta": packed(delta),
        "abs_delta": float(abs(delta)),
        "Esym": esym,
        "EA": ea,
        "phase_error_deg": dphi,
        "Esym_pass_0p02": None if esym is None else bool(esym <= ESYM),
    }


def amp2d(t, jp, jm, window: str, method="linear_exact"):
    z0x = z_channel(t[0], jp[0], jm[0], window, 2, "odd", "x", method)
    z0y = z_channel(t[0], jp[0], jm[0], window, 2, "odd", "y", method)
    zpx = z_channel(t[2], jp[2], jm[2], window, 2, "odd", "x", method)
    zpy = z_channel(t[2], jp[2], jm[2], window, 2, "odd", "y", method)
    n0 = float(np.hypot(abs(z0x), abs(z0y)))
    npi = float(np.hypot(abs(zpx), abs(zpy)))
    return None if n0 == 0.0 else npi / n0


def analyze_folder(folder: Path, label: str) -> dict:
    train_t, train_p, train_m = [], [], []
    for _, ptag, mtag in pp.ANGLE_PAIRS:
        t, jp, jm = pair_currents(folder, ptag, mtag)
        train_t.append(t)
        train_p.append(jp)
        train_m.append(jm)
    ht, hp, hm = pair_currents(folder, "p_q", "m_q")
    leak = spectrum(train_t[0], np.ones((train_t[0].size, 2)), 2 * pp.PERIOD, 4 * pp.PERIOD,
                    np.array([2.0]), method="linear_exact", omega=pp.OMEGA0)
    leak_abs = float(np.max(np.abs(leak)))
    windows = {}
    for wname in ("front", "main", "back"):
        channels = {}
        for cname, q, parity, axis in CHANNELS:
            zs = [z_channel(train_t[i], train_p[i], train_m[i], wname, q, parity, axis) for i in range(4)]
            fit = pp.fit_from_z(zs, 1e-18)
            B, A, C = fit["coeff"]
            z_hold = z_channel(ht, hp, hm, wname, q, parity, axis)
            met = holdout_metrics(z_hold, fit["holdout_prediction"], A)
            rho = fit["rho"]
            cancel = None if rho is None or abs(1.0 - rho) < 1e-18 else float((1.0 + rho) / abs(1.0 - rho))
            channels[cname] = {
                "q": q, "parity": parity, "axis": axis,
                "B": packed(B), "A": packed(A), "C": packed(C),
                "observations": [packed(z) for z in zs],
                "relative_L2_residual": fit["relative_L2_residual"],
                "rho": rho,
                "arg_A_over_B_deg": fit["arg_AB_deg"],
                "arg_distance_to_pi_deg": fit["arg_distance_to_pi_deg"],
                "abs_C_over_A": fit["abs_C_over_A"],
                "x_amp_ratio_pi_over_0": fit["x_amp_ratio_pi_over_0"],
                "antiphase_cancellation_ratio": cancel,
                "holdout": met,
                "fitted_in_this_window": True,
                "did_not_reuse_main_coefficients": wname != "main",
            }
        t0 = [train_t[0], train_t[2]]
        p0 = [train_p[0], train_p[2]]
        m0 = [train_m[0], train_m[2]]
        channels["h2_odd_x"]["amp2d_ratio_pi_over_0"] = amp2d(
            [train_t[0], None, train_t[2]],
            [train_p[0], None, train_p[2]],
            [train_m[0], None, train_m[2]],
            wname,
        )
        windows[wname] = {
            "lo_over_T": pp.WINDOWS[wname][0],
            "hi_over_T": pp.WINDOWS[wname][1],
            "channels": channels,
        }
    primary = {w: windows[w]["channels"]["h2_odd_x"] for w in ("front", "main", "back")}
    return {
        "label": label,
        "folder": str(folder.relative_to(ROOT)),
        "constant_leak_abs_max": leak_abs,
        "windows": windows,
        "primary_h2_odd_x": primary,
        "near_antiphase_all_windows": all(
            (primary[w]["arg_distance_to_pi_deg"] is not None and primary[w]["arg_distance_to_pi_deg"] < 10.0)
            for w in primary
        ),
        "holdout_pass_all_windows": all(bool(primary[w]["holdout"]["Esym_pass_0p02"]) for w in primary),
    }


def make_figure(qed: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    names = ["front", "main", "back"]
    rho = [qed["primary_h2_odd_x"][w]["rho"] for w in names]
    dist = [qed["primary_h2_odd_x"][w]["arg_distance_to_pi_deg"] for w in names]
    esym = [100.0 * qed["primary_h2_odd_x"][w]["holdout"]["Esym"] for w in names]
    ca = [100.0 * qed["primary_h2_odd_x"][w]["abs_C_over_A"] for w in names]
    fig, ax = plt.subplots(2, 2, figsize=(8.2, 6.4), dpi=140)
    ax[0, 0].plot(names, rho, "o-", color="#1f4e79")
    ax[0, 0].axhline(pp.IDEAL_R, color="#c47b16", ls="--", lw=1, label="ideal R")
    ax[0, 0].set_ylabel(r"$\rho=|A|/|B|$")
    ax[0, 0].legend(frameon=False)
    ax[0, 1].plot(names, dist, "o-", color="#1f4e79")
    ax[0, 1].set_ylabel(r"$|\arg(A/B)-\pi|$ (deg)")
    ax[1, 0].bar(names, esym, color="#7aa2c4")
    ax[1, 0].axhline(2.0, color="#b23a3a", ls="--", lw=1)
    ax[1, 0].set_ylabel(r"$E_{\mathrm{sym}}$ (%) this-window fit")
    ax[1, 1].plot(names, ca, "o-", color="#1f4e79")
    ax[1, 1].set_ylabel(r"$|C/A|$ (%)")
    for a in ax.ravel():
        a.set_xlabel("window")
    fig.suptitle("Q1: each window fitted independently; not main-window prediction on neighbors")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    sha_before = assert_frozen()
    qed = analyze_folder(QED, "exact_QED")
    gh5 = analyze_folder(GH5, "GH5_corrected_integrator_not_MTEF")
    payload = {
        "schema": "1600nm_protocolA_qed_windowQ1_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_sha256_before": sha_before,
        "did_not_rewrite_fourangle_fit": True,
        "did_not_propagate": True,
        "n8_p_q_fockaudit_not_used": True,
        "neighbor_15pct_was_main_prediction_on_other_windows": True,
        "this_round_fits_each_window_separately": True,
        "ideal_R": pp.IDEAL_R,
        "qed": qed,
        "gh5": gh5,
        "not_steady_state_chi": True,
        "not_nmax_convergence": True,
        "fitted_BAC_are_not_microscopic_kernels": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json(OUT / "windowQ1.json", payload)
    make_figure(qed, OUT / "figures" / "fig_windowQ1.png")
    sha_after = assert_frozen()
    payload["fit_sha256_after"] = sha_after
    dump_json(OUT / "windowQ1.json", payload)
    print("Q1_OK", sha_after)
    for w in ("front", "main", "back"):
        row = qed["primary_h2_odd_x"][w]
        print(w, "rho", row["rho"], "dpi", row["arg_distance_to_pi_deg"],
              "Esym", row["holdout"]["Esym"], "pass", row["holdout"]["Esym_pass_0p02"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
