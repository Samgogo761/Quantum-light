#!/usr/bin/env python3
"""Build a Hilbert-completion WINDOW144 TB: production 112 WFs plus Q_32.

This is **not** a chemical 144-WF model (not I-d).  The extra 32 functions
span the outer-window complement Q = I_W - Π of full112.  The original 112
columns are the production Wannier gauge U_prod = U_dis @ U_inner from the
existing .chk, so a later k20 HHG contrast asks only “what did the missing
space do?”, not “what if we re-Wannierized everything?”.

Do not commit the resulting tb.dat (GB-scale).  Do not sbatch GH7 / k40 / k50.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from audit_pi_rq_coupling import (
    DEFAULT_WANNIER_DIRS,
    berry_connection_ang,
    complement_basis,
    iterate_mmn_blocks,
    load_eig,
    mv_shell_weights,
    neighbor_bvecs,
    open_udis_store,
    skip_lines_binary,
    subspace_energy_basis,
    unpack_udis,
)
from audit_wannier_chk_subspace import open_u_matrix_store
from construct_p_conjugate_tb import parse_lattice_ang, write_tb_file

SCHEMA = "window144_hilbert_tb_v1"
N_WAN = 112
N_Q = 32
N_W = N_WAN + N_Q
N_OCC = 84


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def recip_from_lattice_ang(lattice: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def polar_right_unitary(overlap: np.ndarray) -> np.ndarray:
    """Unitary U maximising Re Tr(overlap @ U)."""
    w, _s, vh = np.linalg.svd(overlap, full_matrices=True)
    return vh.conj().T @ w.conj().T


def reorth_complement(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = v - u @ (u.conj().T @ v)
    q, r = np.linalg.qr(v, mode="reduced")
    diag = np.diag(r)
    scale = np.where(np.abs(diag) < 1.0e-14, 1.0 + 0.0j, diag / np.abs(diag))
    return q * scale


def fourier_to_R(kfrac: np.ndarray, irvec: np.ndarray, values: np.ndarray) -> np.ndarray:
    """values: (nk, ...) → (nrpts, ...).  H(R) = 1/Nk Σ_k e^{-ik·R} H(k)."""
    nk = kfrac.shape[0]
    phase = np.exp(-2j * np.pi * (kfrac @ irvec.T)) / nk
    return np.tensordot(phase, values, axes=(0, 0))


def hop_frobenius_by_shell(irvec: np.ndarray, h_r: np.ndarray) -> dict[str, float]:
    rnorm = np.linalg.norm(irvec.astype(float), axis=1)
    weights = np.array([float(np.linalg.norm(h_r[ir])) for ir in range(h_r.shape[0])])
    total = float(weights.sum())
    far = rnorm >= 8.0
    very_far = rnorm >= 12.0
    return {
        "sum_all": total,
        "sum_R0": float(weights[rnorm < 0.5].sum()) if total else 0.0,
        "frac_Rge8": float(weights[far].sum() / total) if total else 0.0,
        "frac_Rge12": float(weights[very_far].sum() / total) if total else 0.0,
        "max_abs_Rge12": float(weights[very_far].max()) if np.any(very_far) else 0.0,
    }


def read_tb_rlist(path: Path) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Return irvec, ndegen, nwann, and text header lines (comment through ndegen)."""
    header: list[str] = []
    with path.open("rb") as raw:
        comment = raw.readline()
        require(comment, f"empty tb: {path}")
        header.append(comment.decode("utf-8", errors="replace").rstrip("\n"))
        line1 = raw.readline()
        require(line1, "truncated tb header")
        text1 = line1.decode("utf-8", errors="replace")
        header.append(text1.rstrip("\n"))
        if b"." in line1:
            for _ in range(2):
                extra = raw.readline()
                header.append(extra.decode("utf-8", errors="replace").rstrip("\n"))
            nw_line = raw.readline()
            header.append(nw_line.decode("utf-8", errors="replace").rstrip("\n"))
            nwann = int(nw_line.split()[0])
        else:
            nwann = int(text1.split()[0])
        nr_line = raw.readline()
        header.append(nr_line.decode("utf-8", errors="replace").rstrip("\n"))
        nrpts = int(nr_line.split()[0])
        ndegen: list[int] = []
        while len(ndegen) < nrpts:
            chunk = raw.readline()
            require(chunk, "truncated ndegen")
            header.append(chunk.decode("utf-8", errors="replace").rstrip("\n"))
            ndegen.extend(int(x) for x in chunk.split())
        ndegen_arr = np.array(ndegen[:nrpts], dtype=float)
        skip_per = nwann * nwann
        irvec = np.zeros((nrpts, 3), dtype=int)
        for ir in range(nrpts):
            raw.readline()
            rline = raw.readline()
            require(rline, f"truncated R header at ir={ir}")
            parts = rline.split()
            irvec[ir] = [int(parts[0]), int(parts[1]), int(parts[2])]
            skip_lines_binary(raw, skip_per, label="tb-H", quiet=True)
    return irvec, ndegen_arr, nwann, header


def content_sha256(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def find_wannier_file(name: str, explicit: Path | None) -> Path:
    if explicit is not None:
        require(explicit.is_file(), f"missing {explicit}")
        return explicit
    for folder in DEFAULT_WANNIER_DIRS:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"cannot find {name} in DEFAULT_WANNIER_DIRS")


def build_bases(chk_path: Path, eig: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    index, udis_raw = open_udis_store(chk_path)
    require(index.num_wann == N_WAN, f"chk num_wann={index.num_wann}, expected {N_WAN}")
    require(index.u_matrix_offset > 0, "chk has no u_matrix; need a postwann checkpoint")
    um_raw = open_u_matrix_store(index)
    nk = index.num_kpts
    nb = index.num_bands
    u_prod = np.zeros((nk, nb, N_WAN), dtype=np.complex128)
    v_q = np.zeros((nk, nb, N_Q), dtype=np.complex128)
    orth_u = np.empty(nk)
    orth_v = np.empty(nk)
    uv_max = np.empty(nk)
    occ_mismatch = np.empty(nk)
    t0 = time.time()
    for ik in range(nk):
        rec = unpack_udis(index, udis_raw, ik + 1)
        u_dis = rec["u"]
        u_inner = np.asarray(um_raw[ik]).T.copy()
        u = u_dis @ u_inner
        gram = u.conj().T @ u
        orth_u[ik] = float(np.max(np.abs(gram - np.eye(N_WAN, dtype=complex))))
        comp = complement_basis(u, rec["lwindow"])
        _u_e, v_e, _eps_pi, _eps_q = subspace_energy_basis(u, comp["v"], eig[ik])
        require(v_e.shape[1] == N_Q, f"Q dim {v_e.shape[1]} at k={ik + 1}")
        v_e = reorth_complement(u, v_e)
        u_prod[ik] = u
        v_q[ik] = v_e
        orth_v[ik] = float(np.max(np.abs(v_e.conj().T @ v_e - np.eye(N_Q, dtype=complex))))
        uv_max[ik] = float(np.max(np.abs(u.conj().T @ v_e)))
        window_e = np.sort(eig[ik][rec["lwindow"]].real)
        h144 = np.concatenate([u, v_e], axis=1)
        eps_144 = np.sort(np.linalg.eigvalsh(h144.conj().T @ (eig[ik][:, None] * h144)).real)
        require(eps_144.size == window_e.size, "window vs 144 spectrum size")
        occ_mismatch[ik] = float(np.max(np.abs(eps_144 - window_e)))
        if (ik + 1) % 96 == 0:
            print(f"  bases: k={ik + 1}/{nk} in {time.time() - t0:.1f}s", flush=True)
    del udis_raw, um_raw
    meta = {
        "num_kpts": int(nk),
        "num_bands": int(nb),
        "nntot": int(index.nntot),
        "chk_header": index.header,
        "kpoints": np.asarray(index.kpoints, dtype=float),
        "lwindow": np.asarray(index.lwindow, dtype=bool),
        "max_Uprod_gram_error": float(orth_u.max()),
        "max_Vq_gram_error": float(orth_v.max()),
        "max_U_V_overlap": float(uv_max.max()),
        "max_window144_eig_mismatch_eV": float(occ_mismatch.max()),
        "index": index,
    }
    print(
        f"bases: U_gram={orth_u.max():.2e} V_gram={orth_v.max():.2e} "
        f"U·V={uv_max.max():.2e} window144={occ_mismatch.max():.3e} eV "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return u_prod, v_q, meta


def collect_q_overlaps(
    mmn_path: Path,
    u_prod: np.ndarray,
    v_q: np.ndarray,
    kpoints: np.ndarray,
    recip: np.ndarray,
) -> tuple[list[list[np.ndarray]], list[list[int]], list[list[tuple[int, int, int, int, int]]]]:
    nk = u_prod.shape[0]
    overlaps: list[list[np.ndarray]] = [[] for _ in range(nk)]
    neighbors: list[list[int]] = [[] for _ in range(nk)]
    headers_all: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(nk)]
    t0 = time.time()
    n_done = 0
    for ik_1, headers, mats in iterate_mmn_blocks(mmn_path, needed=None):
        ik = ik_1 - 1
        v_k = v_q[ik]
        for header, mat in zip(headers, mats):
            jk = header[1] - 1
            overlaps[ik].append(v_k.conj().T @ mat @ v_q[jk])
            neighbors[ik].append(jk)
            headers_all[ik].append(header)
        n_done += 1
        if n_done % 96 == 0:
            print(f"  mmn pass1 (Q overlap): {n_done}/{nk} in {time.time() - t0:.1f}s", flush=True)
    print(f"mmn pass1 done in {time.time() - t0:.1f}s", flush=True)
    _ = recip, kpoints, u_prod
    return overlaps, neighbors, headers_all


def parallel_transport_q(
    v_q: np.ndarray,
    u_prod: np.ndarray,
    overlaps: list[list[np.ndarray]],
    neighbors: list[list[int]],
) -> dict[str, float]:
    nk, _nb, nq = v_q.shape
    g_list = [np.eye(nq, dtype=np.complex128) for _ in range(nk)]
    visited = np.zeros(nk, dtype=bool)
    visited[0] = True
    queue: deque[int] = deque([0])
    n_edges = 0
    while queue:
        ik = queue.popleft()
        for inn, jk in enumerate(neighbors[ik]):
            if visited[jk]:
                continue
            ov = g_list[ik].conj().T @ overlaps[ik][inn]
            g_list[jk] = polar_right_unitary(ov)
            visited[jk] = True
            queue.append(jk)
            n_edges += 1
    n_unvisited = int(np.count_nonzero(~visited))
    require(n_unvisited == 0, f"Q parallel transport missed {n_unvisited} k-points")
    uv_max = 0.0
    for ik in range(nk):
        v_new = reorth_complement(u_prod[ik], v_q[ik] @ g_list[ik])
        v_q[ik] = v_new
        uv_max = max(uv_max, float(np.max(np.abs(u_prod[ik].conj().T @ v_new))))
    return {"pt_tree_edges": int(n_edges), "max_U_V_overlap_after_pt": float(uv_max)}


def assemble_h_and_r(
    mmn_path: Path,
    u_prod: np.ndarray,
    v_q: np.ndarray,
    eig: np.ndarray,
    kpoints: np.ndarray,
    recip: np.ndarray,
    headers_all: list[list[tuple[int, int, int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    nk = u_prod.shape[0]
    h_k = np.zeros((nk, N_W, N_W), dtype=np.complex128)
    r_k = np.zeros((nk, N_W, N_W, 3), dtype=np.complex128)
    t0 = time.time()
    u_all = np.concatenate([u_prod, v_q], axis=2)
    for ik in range(nk):
        u = u_all[ik]
        h_k[ik] = u.conj().T @ (eig[ik][:, None] * u)
        h_k[ik] = 0.5 * (h_k[ik] + h_k[ik].conj().T)
    n_done = 0
    for ik_1, headers, mats in iterate_mmn_blocks(mmn_path, needed=None):
        ik = ik_1 - 1
        u_k = u_all[ik]
        m_w: list[np.ndarray] = []
        for header, mat in zip(headers, mats):
            jk = header[1] - 1
            m_w.append(u_k.conj().T @ mat @ u_all[jk])
        bvecs = neighbor_bvecs(kpoints[ik], np.array(headers), kpoints, recip)
        weights = mv_shell_weights(bvecs)
        r_ang = berry_connection_ang(m_w, bvecs, weights)
        for alpha in range(3):
            r_k[ik, :, :, alpha] = r_ang[alpha]
        n_done += 1
        if n_done % 96 == 0:
            print(f"  mmn pass2 (r): {n_done}/{nk} in {time.time() - t0:.1f}s", flush=True)
        _ = headers_all
    print(f"H(k)+r(k) assembled in {time.time() - t0:.1f}s", flush=True)
    return h_k, r_k


def interpolate_h(irvec: np.ndarray, ndegen: np.ndarray, h_r: np.ndarray, kfrac: np.ndarray) -> np.ndarray:
    phase = np.exp(2j * np.pi * (irvec @ np.asarray(kfrac, dtype=float))) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    return 0.5 * (hk + hk.conj().T)


def run_gates(
    h_k: np.ndarray,
    h_r: np.ndarray,
    r_r: np.ndarray,
    irvec: np.ndarray,
    ndegen: np.ndarray,
    kpoints: np.ndarray,
    eig: np.ndarray,
    lwindow: np.ndarray,
    u_prod: np.ndarray,
    v_q: np.ndarray,
    ref_h_r: np.ndarray | None,
) -> dict[str, Any]:
    nk = h_k.shape[0]
    herm = max(float(np.max(np.abs(h_k[ik] - h_k[ik].conj().T))) for ik in range(min(nk, 8)))
    mesh_err = []
    window_err = []
    pi_block_err = []
    probe = [0, min(61, nk - 1), min(457, nk - 1), nk // 2]
    for ik in probe:
        hk_i = interpolate_h(irvec, ndegen, h_r, kpoints[ik])
        mesh_err.append(float(np.max(np.abs(hk_i - h_k[ik]))))
        eps_i = np.sort(np.linalg.eigvalsh(hk_i).real)
        window_e = np.sort(eig[ik][np.asarray(lwindow[ik], dtype=bool)].real)
        window_err.append(float(np.max(np.abs(eps_i - window_e))))
        u = u_prod[ik]
        h_pi = u.conj().T @ (eig[ik][:, None] * u)
        pi_block_err.append(float(np.max(np.abs(h_k[ik][:N_WAN, :N_WAN] - h_pi))))
    decay_all = hop_frobenius_by_shell(irvec, h_r)
    decay_q = hop_frobenius_by_shell(irvec, h_r[:, N_WAN:, N_WAN:])
    decay_coup = hop_frobenius_by_shell(irvec, h_r[:, :N_WAN, N_WAN:])
    r0 = next(i for i in range(irvec.shape[0]) if tuple(irvec[i]) == (0, 0, 0))
    r_diag = np.linalg.norm(np.diagonal(r_r[r0], axis1=0, axis2=1).real, axis=0)
    ref_block = None
    if ref_h_r is not None:
        diffs = []
        for ir in range(min(h_r.shape[0], ref_h_r.shape[0])):
            diffs.append(float(np.max(np.abs(h_r[ir, :N_WAN, :N_WAN] - ref_h_r[ir]))))
        ref_block = {"max_abs_HR_112_block_vs_prod": float(max(diffs))}
    uv = max(float(np.max(np.abs(u_prod[ik].conj().T @ v_q[ik]))) for ik in probe)
    return {
        "hermiticity_maxabs_probe": herm,
        "mesh_roundtrip_maxabs": float(max(mesh_err)),
        "window144_vs_QE_maxabs_eV": float(max(window_err)),
        "pi_block_vs_Uprod_maxabs_eV": float(max(pi_block_err)),
        "U_V_overlap_probe": uv,
        "hop_all": decay_all,
        "hop_QQ": decay_q,
        "hop_PiQ": decay_coup,
        "r0_center_norm_ang_mean": float(np.mean(r_diag)),
        "prod_112_block": ref_block,
        "probe_k_1based": [ik + 1 for ik in probe],
    }


def decide_gates(gates: dict[str, Any]) -> dict[str, str]:
    status = {}
    status["HERMITICITY"] = "PASS" if gates["hermiticity_maxabs_probe"] < 1.0e-8 else "FAIL"
    status["MESH_ROUNDTRIP"] = "PASS" if gates["mesh_roundtrip_maxabs"] < 5.0e-4 else "FAIL"
    status["WINDOW144_VS_QE"] = "PASS" if gates["window144_vs_QE_maxabs_eV"] < 5.0e-3 else "FAIL"
    status["PI_BLOCK"] = "PASS" if gates["pi_block_vs_Uprod_maxabs_eV"] < 1.0e-6 else "FAIL"
    status["U_ORTH_Q"] = "PASS" if gates["U_V_overlap_probe"] < 1.0e-8 else "FAIL"
    qq_far = gates["hop_QQ"]["frac_Rge12"]
    if qq_far < 0.05:
        status["Q_LOCALIZATION"] = "PASS"
    elif qq_far < 0.20:
        status["Q_LOCALIZATION"] = "WARN"
    else:
        status["Q_LOCALIZATION"] = "FAIL"
    hard = {"HERMITICITY", "MESH_ROUNDTRIP", "WINDOW144_VS_QE", "PI_BLOCK", "U_ORTH_Q"}
    status["OVERALL"] = "FAIL" if any(status[k] == "FAIL" for k in hard) else "PASS"
    if status["OVERALL"] == "PASS" and status["Q_LOCALIZATION"] == "FAIL":
        status["OVERALL"] = "WARN"
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chk", type=Path, default=None)
    ap.add_argument("--eig", type=Path, default=None)
    ap.add_argument("--mmn", type=Path, default=None)
    ap.add_argument("--win", type=Path, default=None)
    ap.add_argument("--ref-tb", type=Path, default=None, help="production 112 tb.dat (R-list + optional 112-block gate)")
    ap.add_argument("--out-tb", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--write-tb", action="store_true")
    ap.add_argument("--skip-pt", action="store_true", help="keep energy-ordered Q; no mmn parallel transport")
    ap.add_argument("--skip-r", action="store_true", help="Fourier H only; r(R)=0 (debug)")
    args = ap.parse_args()

    chk = find_wannier_file("CrI3.chk", args.chk)
    eig_path = find_wannier_file("CrI3.eig", args.eig)
    mmn = find_wannier_file("CrI3.mmn", args.mmn)
    win = find_wannier_file("CrI3.win", args.win)
    ref_tb = find_wannier_file("CrI3_tb.dat", args.ref_tb)
    out_json = args.out_json or (
        Path(__file__).resolve().parents[2] / "results" / "window144_hilbert_20260901" / "build_gate.json"
    )
    out_tb = args.out_tb or Path(r"C:\Users\26507\Documents\量子光研究\01_输入与模型\wannier\window144_hilbert\CrI3_window144_tb.dat")

    print("WINDOW144 Hilbert completion builder", flush=True)
    print(f"  chk={chk}", flush=True)
    print(f"  mmn={mmn}", flush=True)
    eig = load_eig(eig_path)
    u_prod, v_q, meta = build_bases(chk, eig)
    lattice = parse_lattice_ang(win)
    recip = recip_from_lattice_ang(lattice)
    kpoints = meta["kpoints"]

    if args.skip_pt:
        pt_info = {"skipped": True}
        print("parallel transport skipped (--skip-pt)", flush=True)
        headers_all: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(u_prod.shape[0])]
    else:
        overlaps, neighbors, headers_all = collect_q_overlaps(mmn, u_prod, v_q, kpoints, recip)
        pt_info = parallel_transport_q(v_q, u_prod, overlaps, neighbors)
        print(f"parallel transport: {pt_info}", flush=True)
        del overlaps

    if args.skip_r:
        h_k = np.zeros((u_prod.shape[0], N_W, N_W), dtype=np.complex128)
        r_k = np.zeros((u_prod.shape[0], N_W, N_W, 3), dtype=np.complex128)
        u_all = np.concatenate([u_prod, v_q], axis=2)
        for ik in range(u_prod.shape[0]):
            u = u_all[ik]
            h_k[ik] = 0.5 * (
                (u.conj().T @ (eig[ik][:, None] * u))
                + (u.conj().T @ (eig[ik][:, None] * u)).conj().T
            )
    else:
        h_k, r_k = assemble_h_and_r(mmn, u_prod, v_q, eig, kpoints, recip, headers_all)

    print(f"reading R-list from {ref_tb} ...", flush=True)
    t_ref = time.time()
    irvec, ndegen, nw_ref, header_lines = read_tb_rlist(ref_tb)
    require(nw_ref == N_WAN, f"ref tb nwann={nw_ref}")
    print(f"  nrpts={irvec.shape[0]} in {time.time() - t_ref:.1f}s", flush=True)

    h_r = fourier_to_R(kpoints, irvec, h_k)
    r_r = fourier_to_R(kpoints, irvec, r_k)
    gates = run_gates(h_k, h_r, r_r, irvec, ndegen, kpoints, eig, meta["lwindow"], u_prod, v_q, None)
    status = decide_gates(gates)
    print("GATES:", json.dumps(status, sort_keys=True), flush=True)
    print("  mesh_roundtrip", gates["mesh_roundtrip_maxabs"], "window144", gates["window144_vs_QE_maxabs_eV"], flush=True)
    print("  hop_QQ", gates["hop_QQ"], flush=True)

    wrote = False
    if args.write_tb and status["OVERALL"] != "FAIL":
        out_tb.parent.mkdir(parents=True, exist_ok=True)
        comment = (
            "WINDOW144 Hilbert completion Pi_112(+)Q_32; NOT a chemical 144-WF model; "
            "frozen production U_inner gauge"
        )
        new_header = list(header_lines)
        for i, line in enumerate(new_header):
            if line.strip().isdigit() and int(line.strip()) == N_WAN:
                new_header[i] = f"{N_W:12d}"
                break
        else:
            raise RuntimeError("could not replace nwann in tb header")
        print(f"writing {out_tb} ...", flush=True)
        t_w = time.time()
        write_tb_file(out_tb, new_header, irvec, h_r, r_r, comment=comment)
        wrote = True
        print(f"  wrote in {time.time() - t_w:.1f}s size={out_tb.stat().st_size}", flush=True)
    elif args.write_tb:
        print("refusing to write tb.dat because OVERALL=FAIL", flush=True)

    report = {
        "schema": SCHEMA,
        "label": "WINDOW144_HILBERT_COMPLETION_NOT_CHEMICAL",
        "n_wan_frozen": N_WAN,
        "n_q": N_Q,
        "n_w": N_W,
        "n_occ": N_OCC,
        "chk": str(chk),
        "mmn": str(mmn),
        "ref_tb": str(ref_tb),
        "out_tb": str(out_tb) if wrote else None,
        "wrote_tb": wrote,
        "parallel_transport": pt_info,
        "basis_meta": {
            "max_Uprod_gram_error": meta["max_Uprod_gram_error"],
            "max_Vq_gram_error": meta["max_Vq_gram_error"],
            "max_U_V_overlap_before_pt": meta["max_U_V_overlap"],
            "max_window144_eig_mismatch_eV": meta["max_window144_eig_mismatch_eV"],
            "chk_header": meta["chk_header"],
        },
        "gates": gates,
        "status": status,
        "IMPACT_ON_HHG": "UNRESOLVED",
        "forbidden": {
            "GH7": True,
            "k40": True,
            "k50": True,
            "chemical_144_as_Id": True,
        },
    }
    dump = {k: v for k, v in report.items() if k != "content_sha256_without_self"}
    report["content_sha256_without_self"] = content_sha256(dump)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    return 0 if status["OVERALL"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
