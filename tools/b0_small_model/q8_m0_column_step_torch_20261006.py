#!/usr/bin/env python3
"""Optional CUDA midpoint for Γ12 column blocks. float64, no autograd.

Fails closed: import this module without torch and gpu_ok() is False.
Does not change D, gamma, P1, or the CPU midpoint algebra.
"""
from __future__ import annotations

import os
import time

import numpy as np

from q8_m0_continuum_columns_20261004 import ColumnState, current, midpoint_step


def gpu_probe():
    rec = {
        "torch_import": False,
        "cuda_is_available": False,
        "device": None,
        "torch_version": None,
        "cuda_version": None,
        "device_name": None,
    }
    try:
        import torch
    except Exception as exc:
        rec["error"] = type(exc).__name__ + ":" + str(exc)[:200]
        return rec
    rec["torch_import"] = True
    rec["torch_version"] = str(torch.__version__)
    rec["cuda_version"] = str(torch.version.cuda)
    rec["cuda_is_available"] = bool(torch.cuda.is_available())
    if rec["cuda_is_available"]:
        rec["device"] = "cuda:0"
        rec["device_name"] = torch.cuda.get_device_name(0)
        rec["device_count"] = int(torch.cuda.device_count())
    return rec


def _as_c128(t, x):
    import torch
    return torch.as_tensor(x, dtype=torch.complex128, device=t.device)


def midpoint_step_torch(state_t, u, d, dt, gamma, fa, torch_mod):
    """Same 2x2 Schur midpoint as CPU midpoint_step. State stays on device."""
    torch = torch_mod
    u_t = _as_c128(state_t["b"], u)
    d_t = torch.as_tensor(d, dtype=torch.complex128, device=state_t["b"].device)
    fa_t = _as_c128(state_t["b"], fa)
    b, bd, a = state_t["b"], state_t["bdag"], state_t["A"]
    difference = u_t.conj() @ u_t.T - u_t @ u_t.conj().T
    eye = torch.eye(2, dtype=torch.complex128, device=b.device)
    lhs = eye + (dt * gamma / 2.0) * d_t - (1j * dt * dt * gamma / 4.0) * difference
    target = a + fa_t / 2.0 + (dt * gamma / 2.0) * (u_t.conj() @ b + u_t @ bd)
    amid = torch.linalg.solve(lhs, target)
    return {
        "b": b + dt * (1j * (u_t.T @ amid)),
        "bdag": bd + dt * (-1j * (u_t.conj().T @ amid)),
        "A": 2.0 * amid - a,
    }


def current_torch(state_t, u, d, torch_mod):
    u_t = _as_c128(state_t["b"], u)
    d_t = torch_mod.as_tensor(d, dtype=torch_mod.complex128, device=state_t["b"].device)
    return u_t.conj() @ state_t["b"] + u_t @ state_t["bdag"] - d_t @ state_t["A"]


def to_device(state, torch_mod, device):
    return {
        "b": torch_mod.as_tensor(np.asarray(state.b, np.complex128), dtype=torch_mod.complex128, device=device),
        "bdag": torch_mod.as_tensor(np.asarray(state.bdag, np.complex128), dtype=torch_mod.complex128, device=device),
        "A": torch_mod.as_tensor(np.asarray(state.A, np.complex128), dtype=torch_mod.complex128, device=device),
    }


def to_host(state_t):
    return ColumnState(
        state_t["b"].detach().cpu().numpy(),
        state_t["bdag"].detach().cpu().numpy(),
        state_t["A"].detach().cpu().numpy(),
    )


def accumulate_torch(photon, gamma, initial, n_point, k_scalar, vertices, nstep, h, pol, free_A_increment, torch_mod, device, t0=0.0):
    """Equivalent midpoint, device-resident state, end-to-end timing.

    Input vertices may be numpy arrays/memmaps or device tensors. No physical
    approximations, no autograd, no replacement of bdag by conjugate(b).
    """
    torch = torch_mod
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    with torch.no_grad():
        st = to_device(initial, torch, device)
        pol_t = torch.as_tensor(pol, dtype=torch.complex128, device=device)
        k_t = torch.as_tensor(k_scalar, dtype=torch.complex128, device=device)
        # Only the small 2 x n_el vertices are shared across columns.
        # Stack the requested window once instead of uploading twice per step.
        u_t = torch.as_tensor(np.asarray([vertices[j][0] for j in range(nstep)]), dtype=torch.complex128, device=device)
        d_t = torch.as_tensor(np.asarray([vertices[j][1] for j in range(nstep)]), dtype=torch.complex128, device=device)
        omega_t = torch.as_tensor(np.asarray(photon.omega), dtype=torch.float64, device=device)
        afree_t = torch.as_tensor(photon.free_A0, dtype=torch.complex128, device=device)
        phase_delta = torch.expm1(-1j * omega_t * h)[None, :]
        ncol = int(initial.b.shape[1])
        nph = int(omega_t.numel())
        if nph + int(n_point) != ncol:
            raise ValueError("Photon/dark column count mismatch")
        acc_u = torch.zeros((2, ncol), dtype=torch.complex128, device=device)
        acc_v = torch.zeros_like(acc_u)
        fa = torch.zeros_like(acc_u)
        if cuda:
            torch.cuda.synchronize()
        loop_start = time.perf_counter()
        ii = torch.zeros_like(acc_u)
        for step in range(nstep):
            time_now = float(t0) + step * h
            fa[:, :nph] = afree_t * torch.exp(-1j * omega_t * time_now)[None, :] * phase_delta
            prev = st
            st = midpoint_step_torch(st, u_t[step], d_t[step], h, gamma, fa, torch)
            mid = {name: 0.5 * (prev[name] + st[name]) for name in ("b", "bdag", "A")}
            ii = current_torch(mid, u_t[step], d_t[step], torch)
            src = pol_t @ ii
            acc_u.add_(h * k_t[step] * src)
            acc_v.add_(h * k_t[step] * src.conj())
        if cuda:
            torch.cuda.synchronize()
        loop_end = time.perf_counter()
        host = to_host(st)
        result = {
            "state": host,
            "delta_u": acc_u.cpu().numpy(),
            "v": acc_v.cpu().numpy(),
            "final_I": ii.cpu().numpy(),
            "device": str(device),
        }
        if cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        result.update({
            "wall_s": end - total_start,
            "h2d_setup_s": loop_start - total_start,
            "loop_s": loop_end - loop_start,
            "d2h_s": end - loop_end,
            "wall_includes_transfers": True,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if cuda else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if cuda else None,
            "float_precision": "complex128",
        })
        return result


def cpu_accumulate(photon, gamma, initial, n_point, k_scalar, vertices, nstep, h, pol, free_A_increment, t0=0.0):
    state = initial.copy()
    ncol = int(state.b.shape[1])
    acc_u = np.zeros((2, ncol), complex)
    acc_v = np.zeros((2, ncol), complex)
    t0w = time.perf_counter()
    ii = np.zeros((2, ncol), complex)
    t_base = float(t0)
    for step in range(nstep):
        u, d = vertices[step]
        t = t_base + step * h
        fa = free_A_increment(photon, t, t + h, n_point)
        prev = state
        state = midpoint_step(state, u, d, h, gamma, free_A_increment=fa)
        mid = ColumnState(0.5 * (prev.b + state.b), 0.5 * (prev.bdag + state.bdag), 0.5 * (prev.A + state.A))
        ii = current(mid, u, d)
        src = pol @ ii
        acc_u += h * k_scalar[step] * src
        acc_v += h * k_scalar[step] * np.conj(src)
    return {
        "state": state,
        "delta_u": acc_u,
        "v": acc_v,
        "final_I": ii,
        "wall_s": time.perf_counter() - t0w,
        "device": "cpu",
    }
