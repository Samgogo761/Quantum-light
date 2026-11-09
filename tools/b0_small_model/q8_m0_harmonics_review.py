#!/usr/bin/env python3
"""Stock P1 A*=0.8 H1–H9 complex-amplitude review. No new propagation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import compare_complex_amplitudes, p1_protocol, project_harmonic  # noqa: E402
from q8_m0_h2_qualify import ABS_REPORT_RESOLUTION, classify_channel, dump, z_of  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NK96 = REPO / "results/q_line/m0_p0p1_20260914/p1_A0.8_nk96_nt400/analysis.json"
QUAL = REPO / "results/q_line/p1_h2_qualification_20260915"
OUT = REPO / "results/q_line/p1_harmonics_review_20260916"
JOBS = {
    "nk96": NK96,
    "nk192": QUAL / "p1_A0.8_nk192_nt400/analysis.json",
    "nk192_shift": QUAL / "p1_A0.8_nk192_nt400_shift/analysis.json",
    "nk192_dt800": QUAL / "p1_A0.8_nk192_nt800/analysis.json",
}
NAMES = ("J_plus", "J_minus", "J_e", "J_o")
ORDERS = list(range(1, 10))
NPZ192 = QUAL / "p1_A0.8_nk192_nt400"


def load_tab(path: Path, window: str = "sin2") -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    return rec["analysis"]["windows"][window]


def pack(z: complex) -> dict:
    return {"re": z.real, "im": z.imag, "abs": abs(z), "phase_deg": float(np.angle(z, deg=True)) if abs(z) else None}


def review() -> dict:
    tabs = {key: load_tab(path) for key, path in JOBS.items()}
    ref = tabs["nk192"]
    rows = {}
    for name in NAMES:
        for n in ORDERS:
            for comp in ("x", "y"):
                z_ref = z_of(ref, name, n, comp)
                z96 = z_of(tabs["nk96"], name, n, comp)
                z_s = z_of(tabs["nk192_shift"], name, n, comp)
                z_dt = z_of(tabs["nk192_dt800"], name, n, comp)
                verdict = classify_channel(z_ref, z_k=z96, z_ind=z_s, z_dt=z_dt)
                cmp_k = compare_complex_amplitudes(z96, z_ref)
                cmp_s = compare_complex_amplitudes(z_s, z_ref)
                cmp_dt = compare_complex_amplitudes(z_dt, z_ref)
                label = f"{name}_{comp}_H{n}"
                rows[label] = {
                    "name": name,
                    "n": n,
                    "comp": comp,
                    "hbar_Omega_over_th": float(n * 0.05),
                    "z_ref_nk192": pack(z_ref),
                    "z_nk96": pack(z96),
                    "z_shift": pack(z_s),
                    "z_dt800": pack(z_dt),
                    "rel_k": cmp_k["rel_complex_vs_b"],
                    "rel_ind": cmp_s["rel_complex_vs_b"],
                    "rel_dt": cmp_dt["rel_complex_vs_b"],
                    "d_k": cmp_k["complex_abs_diff"],
                    "d_ind": cmp_s["complex_abs_diff"],
                    "d_dt": cmp_dt["complex_abs_diff"],
                    "phase_diff_deg": verdict["phase_diff_deg"],
                    "status": verdict["status"],
                    "reasons": verdict["reasons"],
                    "window": "sin2",
                    "denominator": "|z_nk192_Gamma_nt400|",
                }
    summary = {
        "cjj_definition": "classical_P1_q0_percell_greater_connected_v1",
        "definition_review": "user_accepted_limited_validation_20260916",
        "approval_scope": "small_scale_classical_pump_electronic_source_validation",
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
        "absolute_reporting_resolution_mean_current": ABS_REPORT_RESOLUTION,
        "classification_rule": "20260915_unresolved_v2",
        "note": (
            "5% is an engineering preliminary line for this campaign, not a paper error bar. "
            "H3 qualification does not transfer to H4 or to CJJ. "
            "boxcar is diagnostic only and is not the primary observable."
        ),
        "qualified_preliminary": sorted(k for k, v in rows.items() if v["status"] == "qualified_preliminary"),
        "trend_not_qualified": sorted(k for k, v in rows.items() if v["status"] == "trend_not_qualified"),
        "unresolved": sorted(k for k, v in rows.items() if v["status"] == "unresolved"),
        "below_reporting_resolution": sorted(k for k, v in rows.items() if v["status"] == "below_reporting_resolution"),
        "channels": rows,
    }
    return summary, tabs


def boxcar_diag(tabs: dict) -> dict:
    out = {}
    for key, tab in tabs.items():
        try:
            btab = load_tab(JOBS[key], "boxcar") if key != "nk96" else None
        except Exception:
            btab = None
        out[key] = "see analysis.json windows.boxcar; not used for qualification"
        del btab
    rec = json.loads(JOBS["nk192"].read_text(encoding="utf-8"))
    if "boxcar" in rec["analysis"]["windows"]:
        b = rec["analysis"]["windows"]["boxcar"]
        out["nk192_Je_x_H3_boxcar"] = pack(z_of(b, "J_e", 3, "x"))
        out["nk192_Je_x_H3_sin2"] = pack(z_of(rec["analysis"]["windows"]["sin2"], "J_e", 3, "x"))
        out["note"] = "boxcar vs sin2 is a change of observable, not a dt/k error of the same z"
    return out


def spectra_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plus = np.load(NPZ192 / "run_p1_A0.8_nk192_nt400_N+1_A+1.npz")
    minus = np.load(NPZ192 / "run_p1_A0.8_nk192_nt400_N-1_A+1.npz")
    times = np.asarray(plus["times"], dtype=float)
    je = 0.5 * (np.asarray(plus["J_sheet"]) + np.asarray(minus["J_sheet"]))
    jo = 0.5 * (np.asarray(plus["J_sheet"]) - np.asarray(minus["J_sheet"]))
    prot = p1_protocol(0.8)
    ws = np.linspace(0.0, 10.0, 501)
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for ax, sig, title in ((axs[0], je, r"$J_e$"), (axs[1], jo, r"$J_o$")):
        vals = np.array([project_harmonic(times, sig, w * prot.omega0, "sin2") for w in ws])
        ax.semilogy(ws, np.abs(vals[:, 0]) + 1e-40, label="x")
        ax.semilogy(ws, np.abs(vals[:, 1]) + 1e-40, label="y")
        for n in range(1, 10):
            ax.axvline(n, color="k", lw=0.4, alpha=0.35)
        ax.set_ylabel(r"$|z(\Omega)|$")
        ax.set_title(title + r"  sin² projection (display sampling)")
        ax.legend(fontsize=8)
    axs[1].set_xlabel(r"$n=\Omega/\omega_1$   (also $\hbar\Omega/t_h=0.05\,n$)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, tabs = review()
    summary["boxcar_diagnostic"] = boxcar_diag(tabs)
    dump(OUT / "h1_h9_qualification.json", summary)
    slim = {k: summary[k] for k in (
        "qualified_preliminary", "trend_not_qualified", "unresolved",
        "below_reporting_resolution", "quantum_input", "photon_output",
    )}
    dump(OUT / "summary_slim.json", slim)
    try:
        spectra_figure(OUT / "fig_Je_Jo_spectrum_n_and_energy.png")
    except Exception as exc:
        dump(OUT / "figure_error.json", {"error": str(exc)})
    print(json.dumps(slim, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
