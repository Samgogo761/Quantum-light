#!/usr/bin/env python3
"""Stage-1Q: p-h covariance -> Pauli depletion / commutator defect.

Raw N_B is reported, not used as pass/fail. Micro conversion is N_B/w_k
from the frozen intensive I_cell vertex. N_cell is unset and cancelled.
No G_self emergence, no BZ, no production V.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_io_audit import audit_grid  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_phgauss import (  # noqa: E402
    PAULI_NORM,
    PHGAUSS_ID,
    h2_two_port_packets,
    pauli_from_phi,
    propagate_phi,
    slim_pauli,
)
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1q_pauli_20260923"
MU = -0.009010620372198276
NTPC = 32
G1Q_ID = "phgauss_pauli_depletion_20260923"


def log(msg: str) -> None:
    print(f"[stage1q] {msg}", flush=True)


def _plot(times, ev, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    ax[0].plot(times, ev["N_B_max_t"], color="C0")
    ax[0].set_ylabel("max N_B")
    ax[1].plot(times, ev["Ncell_times_eps_Pauli_t"], color="C1")
    ax[1].set_ylabel("N_cell * eps_Pauli")
    ax[2].plot(times, ev["Ncell_times_C_minus_I_t"], color="C2")
    ax[2].set_ylabel("N_cell * ||C-I||")
    ax[2].set_xlabel("t (code)")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    pref = eta2_ncell(geo)
    eta = float(np.sqrt(pref))
    e_par, e_perp = parallel_perp_basis(geo)
    t0 = time.perf_counter()
    kf = np.array([[0.0, 0.0], [0.17, 0.41]])
    log(f"CJJ nk=2 ntpc={NTPC}")
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    x, w, md = rec["X_full"], rec["w_k"], rec["mean_D_t"]
    dt = float(rec["dt"])
    times = rec["times"]
    log(f"nt={x.shape[0]} w={np.asarray(w).tolist()}")

    omega, dw = audit_grid(prot, harm_max=8.0)
    flist, labs = h2_two_port_packets(times, prot.omega0, prot.t_pulse, e_perp, dw, omega)

    ev0 = propagate_phi(x, w, md, [], dt, eta, save_pauli=True)
    p0 = ev0["pauli_final"]
    log(
        f"empty-photon max N_B={p0['N_B_max']:.3e} "
        f"Ncell*eps={p0['Ncell_times_eps_Pauli']:.3e} "
        f"Ncell*|C-I|={p0['Ncell_times_C_minus_I']:.3e}"
    )

    ev = propagate_phi(x, w, md, flist, dt, eta, save_pauli=True)
    pf = ev["pauli_final"]
    imax = int(np.argmax(ev["Ncell_times_eps_Pauli_t"]))
    peak_phi = ev["Phi"]
    # peak is along the saved series of the *final* Phi only if max is at the end.
    # Recompute Phi-at-peak by a second pass only if needed: use the time series
    # extrema and the final snapshot; also rebuild peak snapshot via stride store.
    log(
        f"2-packet symplectic {ev['symplectic_maxabs']:.3e} "
        f"N_B_max={pf['N_B_max']:.6g} "
        f"Ncell*f_micro_max={pf['Ncell_times_f_micro_max']:.6g} "
        f"Ncell*eps={pf['Ncell_times_eps_Pauli']:.6g} "
        f"Ncell*|C-I|={pf['Ncell_times_C_minus_I']:.6g} "
        f"Ncell*eps_t_max={float(np.max(ev['Ncell_times_eps_Pauli_t'])):.6g} at n={imax}"
    )

    # If the Pauli max is not at the last step, re-run once storing Phi at that index.
    peak = None
    if imax == x.shape[0] - 1:
        peak = pf
        peak_n = imax
    else:
        # second propagate, capture Phi at imax
        from q8_m0_phgauss import (
            cayley_step,
            is_hamiltonian,
            linear_K,
            pack_waveforms,
            sqrt_w,
            symplectic_J,
        )

        sw = sqrt_w(w)
        fr, fi = pack_waveforms(flist)
        n_ph, n_b, n = ev["n_ph"], ev["n_B"], ev["n"]
        j = symplectic_J(n)
        phi = np.eye(2 * n, dtype=float)
        for nstep in range(imax):
            xmid = 0.5 * (x[nstep] + x[nstep + 1])
            dmid = 0.5 * (md[nstep] + md[nstep + 1])
            frm = 0.5 * (fr[:, nstep] + fr[:, nstep + 1])
            fim = 0.5 * (fi[:, nstep] + fi[:, nstep + 1])
            phi = cayley_step(phi, linear_K(n_ph, sw, xmid, frm, fim, dmid, eta), dt)
        peak = pauli_from_phi(phi, n_ph, n_b, w)
        peak_n = imax
        _ = is_hamiltonian, j, peak_phi

    _plot(times, ev, OUT / "stage1q_pauli_vs_t.png")
    payload = {
        "g1q_id": G1Q_ID,
        "phgauss_id": PHGAUSS_ID,
        "spec": stage1_spec(),
        "frozen": {
            "photon_basis": "H2_L_plus_H2_R_not_continuum",
            "raw_N_B_is_not_a_gate": True,
            "pauli_norm": PAULI_NORM,
            "N_cell": "unset_not_equal_to_nk",
            "S_needed_for_Pauli_gate": True,
            "no_G_self_emergence": True,
            "no_large_BZ": True,
            "no_production_V": True,
        },
        "grid": {
            "n_k": int(kf.shape[0]),
            "n_t": int(x.shape[0]),
            "ntpc": NTPC,
            "kf": kf.tolist(),
            "w_k": np.asarray(w).tolist(),
            "photon_labels": labs,
        },
        "empty_photon": slim_pauli(p0),
        "final": slim_pauli(pf),
        "peak": slim_pauli(peak),
        "peak_index": int(peak_n),
        "peak_time": float(times[peak_n]),
        "series": {
            "N_B_max_t": ev["N_B_max_t"].tolist(),
            "Ncell_times_eps_Pauli_t": ev["Ncell_times_eps_Pauli_t"].tolist(),
            "Ncell_times_C_minus_I_t": ev["Ncell_times_C_minus_I_t"].tolist(),
            "N_B_max_of_t": float(np.max(ev["N_B_max_t"])),
            "Ncell_times_eps_Pauli_max_of_t": float(np.max(ev["Ncell_times_eps_Pauli_t"])),
            "Ncell_times_C_minus_I_max_of_t": float(np.max(ev["Ncell_times_C_minus_I_t"])),
        },
        "evolution": {
            "n_ph": ev["n_ph"],
            "n_B": ev["n_B"],
            "symplectic_maxabs": ev["symplectic_maxabs"],
            "hamiltonian_K_maxabs": ev["hamiltonian_K_maxabs"],
            "eta": ev["eta"],
        },
        "human_gate": True,
        "verdict_left_to_human": True,
        "next_if_human_passes": "G_self_emergence_on_small_k_not_BZ",
        "cjj_wall_s": rec["wall_seconds"],
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1q_pauli.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1q_pauli.json"),
        "empty_Ncell_eps": p0["Ncell_times_eps_Pauli"],
        "N_B_max_t": payload["series"]["N_B_max_of_t"],
        "Ncell_times_eps_Pauli_max_t": payload["series"]["Ncell_times_eps_Pauli_max_of_t"],
        "Ncell_times_C_minus_I_max_t": payload["series"]["Ncell_times_C_minus_I_max_of_t"],
        "Ncell_times_f_micro_max_peak": peak["Ncell_times_f_micro_max"],
        "w_k": np.asarray(w).tolist(),
        "N_cell": "unset_not_equal_to_nk",
        "wall_s": payload["wall_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
