#!/usr/bin/env python3
"""Independent, prescribed-current check of the free-space photon self-kernel.

This is a numerical representation audit, not an electron propagation or a
quantum-output calculation.  The positive-frequency continuum is integrated
without a zero-frequency oscillator.  No hand-made self-field is inserted into
the photon calculation; Maxwell is used only as the comparison target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_gself_emerge import emerge_A_from_I, long_window_bins  # noqa: E402
from q8_m0_gsheet import gamma_self  # noqa: E402
from q8_m0_model import build_geometry  # noqa: E402
from q8_m0_s1_kernel import parallel_perp_basis  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1s_continuum_20260923"
NTPC = 32
T_OBS_OVER_TP = 8
HARM_MAX = 8.0


def photon_retarded_kernel(lags, omega, weights, gamma):
    """K(t-s)=(2 gamma/pi) int_{omega>0} sin(omega*(t-s))/omega d omega.

    The two radiation ports are included in the factor 2 gamma/pi.  Every
    quadrature node is strictly positive; the integrand's removable limit at
    omega=0 is not represented by a photon oscillator.
    """
    lags = np.asarray(lags, dtype=float)
    omega = np.asarray(omega, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(omega <= 0) or np.any(weights <= 0):
        raise ValueError("positive frequencies and weights required")
    return (2.0 * gamma / np.pi) * (
        np.sin(lags[:, None] * omega[None, :]) @ (weights / omega)
    )


def box_grid(t_obs, omega_max):
    dw = 2.0 * np.pi / float(t_obs)
    n = int(round(float(omega_max) / dw))
    return dw * np.arange(1, n + 1), np.full(n, dw)


def open_grid(omega_max, n):
    """Independent nonuniform Gauss quadrature on (0, omega_max)."""
    x, w = leggauss(int(n))
    return 0.5 * omega_max * (x + 1.0), 0.5 * omega_max * w


def field_from_kernel(current, kernel, dt):
    current = np.asarray(current, dtype=float)
    return float(dt) * np.stack(
        [np.convolve(current[:, a], kernel, mode="full")[:len(current)]
         for a in range(current.shape[1])], axis=1,
    )


def compare(field, target, current, dt, gamma):
    den_a = max(float(np.max(np.abs(target))), 1e-30)
    # The target left-step derivative on interval n-1 -> n is gamma I[n-1].
    den_d = max(float(np.max(np.abs(gamma * current[:-1]))), 1e-30)
    derivative = np.diff(field, axis=0) / float(dt)
    # Source-strength-normalized metric: useful when the source is oscillatory
    # and the Maxwell target has near-zero amplitude at some times.
    source_norm = max(float(np.linalg.norm(gamma * current[:-1]) * np.sqrt(dt)), 1e-30)
    return {
        "A_rel_max": float(np.max(np.abs(field - target)) / den_a),
        "A_rel_l2": float(np.linalg.norm(field - target) /
                          max(np.linalg.norm(target), 1e-30)),
        "A_source_norm_l2": float(np.linalg.norm(field - target) / source_norm),
        "dA_rel_max": float(np.max(np.abs(derivative - gamma * current[:-1])) /
                            den_d),
        "dA_source_norm_l2": float(
            np.linalg.norm(derivative - gamma * current[:-1]) * np.sqrt(dt) / source_norm
        ),
        "A_final_over_Maxwell_final": float(
            np.linalg.norm(field[-1]) / max(np.linalg.norm(target[-1]), 1e-30)
        ),
    }


def main():
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    _, e_perp = parallel_perp_basis(geo)
    data = long_window_bins(prot, e_perp, t_obs_over_tp=T_OBS_OVER_TP,
                            ntpc=NTPC, harm_max=HARM_MAX)
    n = data["n_pulse"]
    current = data["i_cell"][:n]
    dt = float(data["dt"])
    lags = dt * np.arange(n)
    gamma = gamma_self(geo)
    target = field_from_kernel(current, np.r_[0.0, np.full(n - 1, gamma)], dt)
    omega_max = HARM_MAX * prot.omega0

    wbox, qbox = box_grid(T_OBS_OVER_TP * prot.t_pulse, omega_max)
    kbox = photon_retarded_kernel(lags, wbox, qbox, gamma)
    abox = field_from_kernel(current, kbox, dt)

    # Exact infinite-UV periodic-box identity for 0 < lag < T_obs:
    # dw sum_{j=1}^infty sin(j*dw*lag)/(j*dw)
    #     = pi/2 - pi*lag/T_obs.
    # Thus K_box,inf(lag)=gamma*(1-2*lag/T_obs).
    t_obs = T_OBS_OVER_TP * prot.t_pulse
    kbox_inf = np.zeros(n)
    kbox_inf[1:] = gamma * (1.0 - 2.0 * lags[1:] / t_obs)
    abox_inf = field_from_kernel(current, kbox_inf, dt)

    open_rows = []
    open_fields = {}
    direct_eom_check = None
    for nquad in (8, 16, 32, 96, 192, 384):
        wo, qo = open_grid(omega_max, nquad)
        ko = photon_retarded_kernel(lags, wo, qo, gamma)
        ao = field_from_kernel(current, ko, dt)
        open_rows.append({"nquad": nquad, "omega_min": float(wo[0]),
                          "omega_max": float(wo[-1]),
                          **compare(ao, target, current, dt, gamma)})
        open_fields[nquad] = ao
        if nquad == 192:
            phase = np.exp(1j * lags[:, None] * wo[None, :])
            phase *= np.sqrt(qo / wo)[None, :]
            f_list = []
            for j in range(nquad):
                f = phase[:, j, None] * e_perp[None, :]
                f_list.extend((f, f))
            eom = emerge_A_from_I(f_list, current, dt,
                                  np.sqrt(gamma / (2.0 * np.pi)),
                                  np.sqrt(gamma / (2.0 * np.pi)))
            direct_eom_check = float(
                np.max(np.abs(eom - ao)) /
                max(np.max(np.abs(target)), 1e-30)
            )

    payload = {
        "status": "prescribed_current_kernel_diagnostic_only",
        "model": "same_two_port_q0_sheet; no electron propagation",
        "T_obs_over_Tp": T_OBS_OVER_TP,
        "n_pulse": n,
        "dt": dt,
        "omega_cut": omega_max,
        "gamma": gamma,
        "no_omega_zero_oscillator": True,
        "box": {"n_modes_per_port": len(wbox),
                 **compare(abox, target, current, dt, gamma)},
        "box_infinite_UV_identity": compare(abox_inf, target, current, dt, gamma),
        "finite_UV_vs_periodic_identity_rel_max": float(
            np.max(np.abs(abox - abox_inf)) /
            max(np.max(np.abs(target)), 1e-30)),
        "open_positive_quadrature": open_rows,
        "open_192_to_384_rel_max": float(
            np.max(np.abs(open_fields[192] - open_fields[384])) /
            max(np.max(np.abs(target)), 1e-30)),
        "open_192_explicit_photon_EOM_vs_retarded_kernel_rel_max":
            direct_eom_check,
        "interpretation": "finite periodic box has gamma*(1-2*tau/Tobs) "
                          "retarded kernel; open positive continuum removes "
                          "that periodic-image term, while finite UV remains",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1s_continuum.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
