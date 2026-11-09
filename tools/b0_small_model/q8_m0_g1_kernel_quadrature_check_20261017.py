"""P2: Independent check of the production detector kernel quadrature.

Compares the production kernel_at_times (fixed uniform grid dw=(2*pi/Tp)/64,
positive frequencies only, upper cutoff 12*omega0=0.6, trapezoid) against
high-resolution reference quadratures of the SAME frozen mathematical object

    K(t) = i * eta * ∫_0^inf phi(omega)/sqrt(omega) * exp(i omega t) d omega,

where phi is renormalized on each quadrature's own grid. The reference is
refined until self-converged (R1 vs R2); the production kernel is then
compared on a common time grid covering 0..20 carrier cycles.

This checks the DETECTOR-kernel frequency quadrature only. It is a different
integral from the input-channel frequency quadrature (which was refined to
1e-10 earlier); do not mix the two.

Outputs JSON + npz into results/q_line/g1_kernel_quadrature_20261017/.
Read-only with respect to production caches. No propagation is run.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from q8_m0_classical import p1_a2_protocol
from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
from q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 import kernel_at_times

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_kernel_quadrature_20261017"
OUTJSON = OUTDIR / "kernel_quadrature_check.json"
OUTNPZ = OUTDIR / "kernel_quadrature_check.npz"

GAMMA = 0.0010807892763646022  # frozen, see handoff sec.2
ETA_FROZEN = 0.013115370973561481


def ref_grids():
    """Two independent high-resolution positive-frequency grids."""
    g1 = np.unique(np.concatenate((
        np.geomspace(1.0e-8, 0.02, 4000),
        np.linspace(0.02, 0.6, 8000))))
    g2 = np.unique(np.concatenate((
        np.geomspace(1.0e-9, 0.02, 12000),
        np.linspace(0.02, 0.6, 24000))))
    return g1, g2


def kernel_on_grid(om, center, tp, eta, times, batch=32):
    """K(t) on an arbitrary positive-frequency grid; phi renormalized here."""
    raw = phi_unnormalized(om, center, tp)
    phi, nrm = normalize_phi(om, raw)
    weight = phi / np.sqrt(om)
    k = np.empty(times.size, complex)
    for s in range(0, times.size, batch):
        sl = slice(s, min(s + batch, times.size))
        phase = np.exp(1j * om[:, None] * times[sl][None, :])
        k[sl] = 1j * eta * np.trapezoid(weight[:, None] * phase, om, axis=0)
    return k, float(nrm)


def time_grid(period):
    fine = np.arange(0.0, 9.0, 1.0 / 128.0) * period          # 0..9 cycles, 128/cycle
    tail = (9.0 + np.arange(0.0, 11.0, 1.0 / 32.0)) * period  # 9..20 cycles, 32/cycle
    marks = np.array([9.0, 20.0, 40.0, 80.0]) * period        # reported checkpoints
    return np.unique(np.concatenate((fine, tail, marks)))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    prot = p1_a2_protocol(0.8)
    tp = float(prot.t_pulse)
    period = float(prot.period)
    center = 2.0 * float(prot.omega0)
    eta = eta_of(GAMMA)
    times = time_grid(period)
    print(f"eta={eta!r} frozen={ETA_FROZEN!r} match={abs(eta-ETA_FROZEN) < 1e-14}", flush=True)
    print(f"tp={tp} period={period} center={center} n_times={times.size}", flush=True)

    # Production kernel on its own native grid, evaluated at the same times.
    k_prod, nrm_prod = kernel_at_times(GAMMA, prot, times)
    om_prod = omega_grid(prot.omega0, tp)
    om_prod_pos = om_prod[om_prod > 0.0]
    prod_dw = float(om_prod[1] - om_prod[0])
    print(f"prod grid: first_pos={om_prod_pos[0]:.6e} dw={prod_dw:.6e} "
          f"max={om_prod[-1]:.6f} n={om_prod_pos.size}", flush=True)

    g1, g2 = ref_grids()
    k_r1, nrm_r1 = kernel_on_grid(g1, center, tp, eta, times)
    print("ref R1 done", flush=True)
    k_r2, nrm_r2 = kernel_on_grid(g2, center, tp, eta, times)
    print("ref R2 done", flush=True)

    d_ref = k_r1 - k_r2  # reference self-consistency
    d_prod = k_prod - k_r2  # production vs refined reference

    def stats(d, name):
        per = {
            "name": name,
            "max_abs": float(np.max(np.abs(d))),
            "max_abs_re": float(np.max(np.abs(np.real(d)))),
            "max_abs_im": float(np.max(np.abs(np.imag(d)))),
            "t_at_max_abs": float(times[int(np.argmax(np.abs(d)))]),
        }
        # per-cycle max |d|
        cyc = times / period
        by = {}
        for lo in range(0, 20):
            m = (cyc >= lo) & (cyc < lo + 1)
            if np.any(m):
                by[f"{lo}..{lo+1}"] = float(np.max(np.abs(d[m])))
        per["max_abs_by_cycle"] = by
        return per

    report = {
        "task": "P2 detector-kernel quadrature check",
        "date": "2026-10-17",
        "frozen": {"gamma": GAMMA, "eta_from_gamma": eta, "eta_frozen": ETA_FROZEN,
                    "omega0": float(prot.omega0), "t_pulse": tp, "period": period,
                    "kernel": "K(t)=i*eta*∫ phi(ω)/√ω e^{iωt} dω, ω>0",
                    "phi": "pulse-matched sin^2 packet centered at 2*omega0"},
        "production_grid": {
            "rule": "uniform arange, dw=(2π/Tp)/64, ω>0, trapezoid",
            "first_positive_omega": float(om_prod_pos[0]),
            "dw": prod_dw,
            "omega_max": float(om_prod[-1]),
            "n_points": int(om_prod_pos.size),
            "phi_normalizer_on_prod_grid": nrm_prod,
        },
        "reference_grids": {
            "R1": "geomspace(1e-8,0.02,4000) ∪ linspace(0.02,0.6,8000)",
            "R2": "geomspace(1e-9,0.02,12000) ∪ linspace(0.02,0.6,24000)",
            "phi_renormalized_per_grid": True,
            "nrm_R1": nrm_r1, "nrm_R2": nrm_r2,
        },
        "reference_self_consistency_R1_minus_R2": stats(d_ref, "R1-R2"),
        "production_minus_R2": stats(d_prod, "prod-R2"),
        "checkpoint_abs_K": {
            f"{c}_cycles": {
                "prod": float(np.abs(k_prod[np.argmin(np.abs(times - c * period))])),
                "R2": float(np.abs(k_r2[np.argmin(np.abs(times - c * period))])),
            } for c in (9.0, 20.0, 40.0, 80.0)
        },
        "notes": [
            "Detector-kernel quadrature is distinct from the input-channel frequency quadrature (earlier refined to 1e-10).",
            "Near-zero crossings absolute error is the meaningful metric.",
            "No frozen physics changed; phi, eta, gamma untouched.",
        ],
    }

    import json
    with open(OUTJSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    np.savez_compressed(
        OUTNPZ, times=times, k_prod=k_prod, k_r1=k_r1, k_r2=k_r2,
        om_prod=om_prod_pos, g1=g1, g2=g2)
    print("saved", OUTJSON, flush=True)


if __name__ == "__main__":
    main()
