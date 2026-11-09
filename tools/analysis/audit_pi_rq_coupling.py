#!/usr/bin/env python3
"""Line-M Hilbert diagnostic: Π r Q, Π v Q, and η_pq at predetermined hotspots.

Π is the 112-dimensional selected subspace projector.  Q = I_W - Π inside the
outer window.  Do not call this PrQ: P is reserved for spatial inversion.

Hotspot rules are fixed (not chosen by inspection):
  set1: k of max edge occupation (bands 105-112) among archived k20 audits
  set2: k of max in-Π interband |e_x · v|^2 on the same k20 grid (TB dH/dk)

SBE never dumped J(k, ω), so set2 is an operator-weight rule, not an H2
Fourier map.  True Π uses U_dis from a .chk when present; otherwise the
QE169-200 coordinate proxy is used and labelled as a proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "full112_pi_rq_coupling_v1"
FORBIDDEN_KEYS = {"PrQ", "PrQ_norm", "pvq"}

NK_SBE = 20
N_OCC = 84
N_WAN = 112
MP_GRID = (24, 24, 1)
WINDOW_FIRST = 57
WINDOW_LAST = 200
PROXY_FIRST = 169
PROXY_LAST = 200
EDGE_LO, EDGE_HI = 105, 112
WVL_NM = 3200.0
POL_CART = np.array([1.0, 0.0, 0.0])  # E ∥ x, solver theta=0

WCM2_TO_AU = 1.0 / 3.50944758e16
EV_TO_HA = 1.0 / 27.211386245988
HA_TO_EV = 27.211386245988
ANG_TO_BOHR = 1.0 / 0.529177210903
BOHR_TO_ANG = 0.529177210903
NM_TO_BOHR = 10.0 * ANG_TO_BOHR
C_AU = 137.035999084
TWOPI = 2.0 * math.pi
OMEGA0_AU = TWOPI * C_AU / (WVL_NM * NM_TO_BOHR)

I_ID3 = 10925781516.425518
I_BASE = 2.0e11
I_HIGH = 1.0e12
FIELD_CASES = (
    ("id3_1p09e10", I_ID3),
    ("baseline_2e11", I_BASE),
    ("tail_1e12", I_HIGH),
)
ETA_DENOM_FLOOR_EV = 0.05
ENERGY_SCREEN_LO = 3.5
ENERGY_SCREEN_HI = 8.0

DEFAULT_WANNIER_DIRS = (
    Path(r"C:\Users\26507\Documents\量子光研究\01_输入与模型\wannier"),
    Path(r"C:\Users\26507\Documents\量子光研究\wannier"),
)
OCC_RELATIVE = (
    Path("docs") / "04_BSV量子光" / "GH5_node_occ_29458_bracket.json",
    Path("docs") / "04_BSV量子光" / "Isafe_occ_29452_bracket.json",
    Path("docs") / "04_BSV量子光" / "GH7_tail_k20_29403_edge_band_audit.json",
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dumps_strict(payload: dict) -> str:
    return json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            require(key not in FORBIDDEN_KEYS, f"forbidden key {key}")
            out[str(key)] = to_jsonable(val)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not math.isfinite(x):
            raise ValueError(f"non-finite float {x}")
        return x
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"not jsonable: {type(obj)}")


def e0_au(intensity_wcm2: float) -> float:
    return math.sqrt(float(intensity_wcm2) * WCM2_TO_AU)


def a0_au(intensity_wcm2: float) -> float:
    return e0_au(intensity_wcm2) / OMEGA0_AU


def sbe_frac(ikx: int, iky: int, nk: int = NK_SBE) -> tuple[float, float, float]:
    return ((ikx - 1) / nk, (iky - 1) / nk, 0.0)


def wrap_frac_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 0.5) % 1.0 - 0.5


def regular_mp_kpoints(mp_grid: tuple[int, int, int] = MP_GRID) -> np.ndarray:
    nkx, nky, nkz = mp_grid
    pts = []
    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                pts.append((ikx / nkx, iky / nky, ikz / nkz))
    return np.asarray(pts, dtype=float)


def nearest_k(frac: tuple[float, float, float], kpoints: np.ndarray) -> dict:
    target = np.asarray(frac, dtype=float)
    delta = wrap_frac_delta(kpoints, target[None, :])
    dist = np.linalg.norm(delta, axis=1)
    ik0 = int(np.argmin(dist))
    return {
        "wannier_k_index_1based": ik0 + 1,
        "wannier_k_frac": [float(x) for x in kpoints[ik0]],
        "sbe_k_frac": [float(x) for x in target],
        "frac_distance": float(dist[ik0]),
        "exact_on_wannier_mesh": bool(dist[ik0] < 1.0e-12),
    }


def resolve_repo(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().parents[2]


def find_existing(names: tuple[str, ...], extra_dirs: tuple[Path, ...] = ()) -> dict[str, Path]:
    found: dict[str, Path] = {}
    dirs = tuple(p for p in extra_dirs + DEFAULT_WANNIER_DIRS if p is not None)
    for directory in dirs:
        if not directory.is_dir():
            continue
        for name in names:
            if name in found:
                continue
            path = directory / name
            if path.is_file():
                found[name] = path
    return found


def walk_edge_max(obj: Any, source: str, acc: list[dict], ctx: dict | None = None) -> None:
    ctx = dict(ctx or {})
    if isinstance(obj, dict):
        for key in (
            "dir",
            "case",
            "kind",
            "id",
            "theta_deg",
            "I_over_Imax",
            "I_Wcm2",
            "phi_cep_deg",
            "slurm_job_id",
            "job",
            "cep_family",
        ):
            if key in obj and obj[key] is not None:
                ctx[key] = obj[key]
        gem = obj.get("global_edge_max")
        if isinstance(gem, dict) and "occ" in gem and "ikx" in gem and "iky" in gem:
            acc.append(
                {
                    "source": source,
                    "dir": ctx.get("dir"),
                    "case": ctx.get("case"),
                    "kind": ctx.get("kind"),
                    "probe_id": ctx.get("id"),
                    "theta_deg": ctx.get("theta_deg"),
                    "I_over_Imax": ctx.get("I_over_Imax"),
                    "job": ctx.get("slurm_job_id") or ctx.get("job"),
                    "I_Wcm2": ctx.get("I_Wcm2"),
                    "phi_cep_deg": ctx.get("phi_cep_deg"),
                    "cep_family": ctx.get("cep_family"),
                    "ikx": int(gem["ikx"]),
                    "iky": int(gem["iky"]),
                    "band": gem.get("band"),
                    "occ": float(gem["occ"]),
                    "t_fs": gem.get("t_fs"),
                    "it": gem.get("it"),
                    "nk": NK_SBE,
                }
            )
        for val in obj.values():
            walk_edge_max(val, source, acc, ctx)
    elif isinstance(obj, list):
        for item in obj:
            walk_edge_max(item, source, acc, ctx)


def register_set1_edge_occupation(repo: Path) -> dict:
    rows: list[dict] = []
    sha: dict[str, str] = {}
    for rel in OCC_RELATIVE:
        path = repo / rel
        require(path.is_file(), f"missing occupation audit {path}")
        sha[path.name] = sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        walk_edge_max(payload, path.name, rows)
    require(rows, "no global_edge_max records")
    winner = max(rows, key=lambda r: r["occ"])
    frac = sbe_frac(winner["ikx"], winner["iky"], winner["nk"])
    mapped = nearest_k(frac, regular_mp_kpoints())
    return {
        "rule": (
            "argmax of global_edge_max.occ over archived k20 occupation audits "
            "29458/29452/29403; not a high-symmetry point chosen by inspection"
        ),
        "n_candidates": len(rows),
        "winner": {**winner, **mapped, "frac_k": list(frac)},
        "occupation_audit_sha256": sha,
    }


def parse_nnkp(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: dict[str, list[str]] = {}
    name = None
    buf: list[str] = []
    for raw in text:
        line = raw.strip()
        if line.startswith("begin "):
            name = line.split(None, 1)[1]
            buf = []
            continue
        if line.startswith("end ") and name is not None:
            blocks[name] = buf
            name = None
            buf = []
            continue
        if name is not None:
            buf.append(line)
    real_l = np.array([list(map(float, row.split()[:3])) for row in blocks["real_lattice"]])
    recip = np.array([list(map(float, row.split()[:3])) for row in blocks["recip_lattice"]])
    kp_lines = blocks["kpoints"]
    nk = int(kp_lines[0].split()[0])
    kpoints = np.array([list(map(float, row.split()[:3])) for row in kp_lines[1 : 1 + nk]])
    nn_lines = blocks["nnkpts"]
    nntot = int(nn_lines[0].split()[0])
    triples = []
    for row in nn_lines[1 : 1 + nk * nntot]:
        ik, jk, g1, g2, g3 = (int(x) for x in row.split()[:5])
        triples.append((ik, jk, g1, g2, g3))
    nnkpts = np.asarray(triples, dtype=int).reshape(nk, nntot, 5)
    return {
        "real_lattice_ang": real_l,
        "recip_lattice": recip,
        "kpoints": kpoints,
        "nntot": nntot,
        "nnkpts": nnkpts,
        "num_kpts": nk,
    }


def load_eig(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    nb = int(data[:, 0].max())
    nk = int(data[:, 1].max())
    eig = np.zeros((nk, nb))
    eig[data[:, 1].astype(int) - 1, data[:, 0].astype(int) - 1] = data[:, 2]
    return eig


def _analysis_dir() -> Path:
    return Path(__file__).resolve().parent


def open_udis_store(chk_path: Path):
    analysis = str(_analysis_dir())
    if analysis not in sys.path:
        sys.path.insert(0, analysis)
    from audit_wannier_chk_subspace import index_checkpoint

    index = index_checkpoint(chk_path)
    dtype = np.dtype(index.endian + "c16")
    udis_raw = np.memmap(
        index.path,
        mode="r",
        dtype=dtype,
        offset=index.udis_offset,
        shape=(index.num_kpts, index.num_wann, index.num_bands),
        order="C",
    )
    return index, udis_raw


def unpack_udis(index, udis_raw, k_1based: int) -> dict:
    require(1 <= k_1based <= index.num_kpts, f"k={k_1based} out of range")
    ik = k_1based - 1
    u_stored = np.asarray(udis_raw[ik]).T.copy()
    nwin = int(index.ndimwin[ik])
    window_rows = np.flatnonzero(index.lwindow[ik])
    require(int(window_rows.size) == nwin, "window row count mismatch")
    compact = u_stored[:nwin, :]
    u = np.zeros((index.num_bands, index.num_wann), dtype=complex)
    u[window_rows, :] = compact
    gram = compact.conj().T @ compact
    orth = float(np.max(np.abs(gram - np.eye(index.num_wann, dtype=complex))))
    return {
        "u": u,
        "lwindow": np.asarray(index.lwindow[ik], dtype=bool).copy(),
        "ndimwin": nwin,
        "num_wann": int(index.num_wann),
        "num_bands": int(index.num_bands),
        "k_frac": [float(x) for x in index.kpoints[ik]],
        "orth_error": orth,
        "chk_header": index.header,
        "chk_num_kpts": int(index.num_kpts),
        "chk_nntot": int(index.nntot),
    }


def load_udis_at_k(chk_path: Path, k_1based: int) -> dict:
    index, udis_raw = open_udis_store(chk_path)
    rec = unpack_udis(index, udis_raw, k_1based)
    del udis_raw
    return rec


def complement_basis(u: np.ndarray, lwindow: np.ndarray) -> dict:
    nb, nw = u.shape
    window = np.flatnonzero(lwindow)
    nwin = int(window.size)
    nq = nwin - nw
    require(nq > 0, "empty complement")
    u_win = u[window, :]
    q_win = np.eye(nwin, dtype=complex) - u_win @ u_win.conj().T
    q_win = 0.5 * (q_win + q_win.conj().T)
    evals, evecs = np.linalg.eigh(q_win)
    order = np.argsort(evals.real)[::-1]
    evals = evals[order].real
    evecs = evecs[:, order]
    n_pos = int(np.sum(evals > 0.5))
    require(n_pos == nq, f"Q rank {n_pos} != {nq}; head={evals[: min(6, len(evals))]}")
    v = np.zeros((nb, nq), dtype=complex)
    v[window, :] = evecs[:, :nq]
    pi = u @ u.conj().T
    q = v @ v.conj().T
    eye_w = np.zeros((nb, nb), dtype=complex)
    eye_w[np.ix_(window, window)] = np.eye(nwin, dtype=complex)
    return {
        "v": v,
        "pi": pi,
        "q": q,
        "i_window": eye_w,
        "window": window,
        "nq": nq,
        "pi_trace": float(np.trace(pi).real),
        "q_trace": float(np.trace(q).real),
        "u_v_overlap_maxabs": float(np.max(np.abs(u.conj().T @ v))),
        "v_gram_error": float(np.max(np.abs(v.conj().T @ v - np.eye(nq, dtype=complex)))),
        "q_minus_iw_minus_pi_maxabs": float(np.max(np.abs(q - (eye_w - pi)))),
        "q_projector_eig_min": float(evals[nq - 1]),
        "q_projector_eig_max": float(evals[0]),
    }


def subspace_energy_basis(u: np.ndarray, v: np.ndarray, eig_k: np.ndarray):
    h_pi = u.conj().T @ (eig_k[:, None] * u)
    h_q = v.conj().T @ (eig_k[:, None] * v)
    h_pi = 0.5 * (h_pi + h_pi.conj().T)
    h_q = 0.5 * (h_q + h_q.conj().T)
    eps_pi, rot_pi = np.linalg.eigh(h_pi)
    eps_q, rot_q = np.linalg.eigh(h_q)
    return u @ rot_pi, v @ rot_q, np.asarray(eps_pi.real), np.asarray(eps_q.real)


def frobenius_rect(r_cart: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[dict, np.ndarray]:
    out = {}
    for name, axis in (("x", 0), ("y", 1), ("z", 2)):
        block = left.conj().T @ r_cart[axis] @ right
        out[f"pi_r{name}_q_frobenius_ang"] = float(np.linalg.norm(block))
        out[f"pi_r{name}_q_maxabs_ang"] = float(np.max(np.abs(block))) if block.size else 0.0
    r_pol = np.tensordot(POL_CART, r_cart, axes=(0, 0))
    block_pol = left.conj().T @ r_pol @ right
    out["pi_rpol_q_frobenius_ang"] = float(np.linalg.norm(block_pol))
    out["pi_rpol_q_maxabs_ang"] = float(np.max(np.abs(block_pol))) if block_pol.size else 0.0
    return out, block_pol


def relabel_velocity_frobenius(norms: dict) -> dict:
    return {
        "pi_vpol_q_frobenius_au": norms["pi_rpol_q_frobenius_ang"],
        "pi_vpol_q_maxabs_au": norms["pi_rpol_q_maxabs_ang"],
        "pi_vx_q_frobenius_au": norms["pi_rx_q_frobenius_ang"],
        "pi_vy_q_frobenius_au": norms["pi_ry_q_frobenius_ang"],
        "pi_vz_q_frobenius_au": norms["pi_rz_q_frobenius_ang"],
    }


def skip_lines_binary(raw, n_lines: int, bufsize: int = 8 << 20, label: str = "mmn", quiet: bool = False) -> None:
    remaining = int(n_lines)
    if remaining <= 0:
        return
    total = remaining
    t0 = time.time()
    last_print = 0
    while remaining > 0:
        chunk = raw.read(bufsize)
        if not chunk:
            raise RuntimeError(f"unexpected EOF skipping {label}, remaining {remaining} lines")
        count = chunk.count(b"\n")
        if count <= remaining:
            remaining -= count
            done = total - remaining
            if (not quiet) and (done - last_print >= 8_000_000 or remaining == 0):
                dt = time.time() - t0
                print(f"  {label}: skipped {done}/{total} lines in {dt:.1f}s", flush=True)
                last_print = done
            continue
        pos = -1
        for _ in range(remaining):
            pos = chunk.find(b"\n", pos + 1)
            if pos < 0:
                raise RuntimeError(f"{label} newline count mismatch during overshoot")
        extra = len(chunk) - (pos + 1)
        if extra:
            raw.seek(-extra, os.SEEK_CUR)
        remaining = 0
        if not quiet:
            dt = time.time() - t0
            print(f"  {label}: skipped {total}/{total} lines in {dt:.1f}s", flush=True)


def read_mmn_matrix_binary(raw, nb: int) -> np.ndarray:
    n = nb * nb
    reim = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        line = raw.readline()
        if not line:
            raise RuntimeError("truncated mmn matrix")
        a, b = line.split()[:2]
        reim[i, 0] = float(a)
        reim[i, 1] = float(b)
    return (reim[:, 0] + 1j * reim[:, 1]).reshape((nb, nb), order="F")


def mv_shell_weights(bvecs: np.ndarray) -> np.ndarray:
    mag2 = np.sum(bvecs * bvecs, axis=1)
    keys: list[float] = []
    shell = np.empty(len(bvecs), dtype=int)
    for i, mag in enumerate(mag2):
        matched = None
        for j, key in enumerate(keys):
            scale = max(key, 1.0e-16)
            if abs(mag - key) / scale < 1.0e-6:
                matched = j
                break
        if matched is None:
            matched = len(keys)
            keys.append(float(mag))
        shell[i] = matched
    nshell = len(keys)
    matrix = np.zeros((6, nshell))
    for s in range(nshell):
        bs = bvecs[shell == s]
        matrix[0, s] = np.sum(bs[:, 0] ** 2)
        matrix[1, s] = np.sum(bs[:, 1] ** 2)
        matrix[2, s] = np.sum(bs[:, 2] ** 2)
        matrix[3, s] = np.sum(bs[:, 0] * bs[:, 1])
        matrix[4, s] = np.sum(bs[:, 0] * bs[:, 2])
        matrix[5, s] = np.sum(bs[:, 1] * bs[:, 2])
    rhs = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    active = np.linalg.norm(matrix, axis=1) > 1.0e-10
    weights_shell, *_ = np.linalg.lstsq(matrix[active], rhs[active], rcond=None)
    return weights_shell[shell]


def neighbor_bvecs(kfrac: np.ndarray, nn_rows: np.ndarray, kpoints: np.ndarray, recip: np.ndarray) -> np.ndarray:
    bvecs = np.zeros((len(nn_rows), 3))
    for i, row in enumerate(nn_rows):
        _ik, jk, g1, g2, g3 = (int(x) for x in row)
        g = np.array([g1, g2, g3], dtype=float)
        dk_frac = kpoints[jk - 1] + g - kfrac
        bvecs[i] = dk_frac @ recip
    return bvecs


def berry_connection_ang(m_list: list[np.ndarray], bvecs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    nb = m_list[0].shape[0]
    r_ang = np.zeros((3, nb, nb), dtype=complex)
    eye = np.eye(nb, dtype=complex)
    for mat, bvec, weight in zip(m_list, bvecs, weights):
        delta = mat - eye
        for alpha in range(3):
            r_ang[alpha] += 1j * weight * bvec[alpha] * delta
    for alpha in range(3):
        r_ang[alpha] = 0.5 * (r_ang[alpha] + r_ang[alpha].conj().T)
    return r_ang


def read_mmn_for_k(path: Path, k_1based: int) -> tuple[int, int, int, list[tuple[int, int, int, int, int]], list[np.ndarray]]:
    t0 = time.time()
    with path.open("rb") as raw:
        raw.readline()
        counts = raw.readline().split()
        nb, nk, nntot = (int(x) for x in counts[:3])
        require(1 <= k_1based <= nk, f"mmn k={k_1based} out of range nk={nk}")
        n_skip = (k_1based - 1) * nntot * (1 + nb * nb)
        print(
            f"mmn: nb={nb} nk={nk} nntot={nntot}; skipping {n_skip} lines to k={k_1based}",
            flush=True,
        )
        skip_lines_binary(raw, n_skip, label="mmn")
        headers: list[tuple[int, int, int, int, int]] = []
        mats: list[np.ndarray] = []
        for _ in range(nntot):
            line = raw.readline()
            if not line:
                raise RuntimeError("truncated mmn neighbour header")
            parts = line.split()
            if len(parts) < 5:
                raise RuntimeError("truncated mmn neighbour header fields")
            ik = int(parts[0])
            require(ik == k_1based, f"mmn expected ik={k_1based}, got {ik}")
            header = (ik, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            mats.append(read_mmn_matrix_binary(raw, nb))
            headers.append(header)
        require(len(mats) == nntot, f"mmn neighbours for k={k_1based}: {len(mats)} != {nntot}")
        print(f"mmn: extracted {nntot} overlaps at k={k_1based} in {time.time() - t0:.1f}s", flush=True)
        return nb, nk, nntot, headers, mats


def iterate_mmn_blocks(path: Path, needed: set[int] | None = None):
    """Yield ``(k_1based, headers, mats)`` in file order.

    ``needed=None`` streams every k.  Otherwise one sequential pass that skips
    unrequested blocks, same as :func:`iterate_mmn_selected`.
    """
    remaining = None if needed is None else set(int(k) for k in needed)
    if remaining is not None:
        require(remaining, "empty mmn k set")
    with path.open("rb") as raw:
        raw.readline()
        counts = raw.readline().split()
        nb, nk, nntot = (int(x) for x in counts[:3])
        skip_per = nntot * (1 + nb * nb)
        for ik in range(1, nk + 1):
            if remaining is not None and ik not in remaining:
                skip_lines_binary(raw, skip_per, label="mmn", quiet=True)
                continue
            headers: list[tuple[int, int, int, int, int]] = []
            mats: list[np.ndarray] = []
            for _ in range(nntot):
                line = raw.readline()
                if not line:
                    raise RuntimeError("truncated mmn neighbour header")
                parts = line.split()
                if len(parts) < 5:
                    raise RuntimeError("truncated mmn neighbour header fields")
                header_ik = int(parts[0])
                require(header_ik == ik, f"mmn expected ik={ik}, got {header_ik}")
                header = (
                    header_ik,
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                )
                mats.append(read_mmn_matrix_binary(raw, nb))
                headers.append(header)
            if remaining is not None:
                remaining.remove(ik)
            yield ik, headers, mats
            if remaining is not None and not remaining:
                return
    if remaining:
        require(not remaining, f"mmn missing k={sorted(remaining)[:8]}")


def iterate_mmn_selected(path: Path, needed: set[int]):
    """Yield (k_1based, mats) for requested k in file order. One sequential pass."""
    remaining = set(int(k) for k in needed)
    require(remaining, "empty mmn k set")
    with path.open("rb") as raw:
        raw.readline()
        counts = raw.readline().split()
        nb, nk, nntot = (int(x) for x in counts[:3])
        skip_per = nntot * (1 + nb * nb)
        print(
            f"mmn scan: nb={nb} nk={nk} nntot={nntot}; extracting {len(remaining)} k-points",
            flush=True,
        )
        for ik in range(1, nk + 1):
            if ik not in remaining:
                skip_lines_binary(raw, skip_per, label="mmn", quiet=True)
                continue
            mats: list[np.ndarray] = []
            for _ in range(nntot):
                line = raw.readline()
                if not line:
                    raise RuntimeError("truncated mmn neighbour header")
                parts = line.split()
                header_ik = int(parts[0])
                require(header_ik == ik, f"mmn expected ik={ik}, got {header_ik}")
                mats.append(read_mmn_matrix_binary(raw, nb))
            remaining.remove(ik)
            yield ik, mats
            if not remaining:
                return
    require(not remaining, f"mmn missing k={sorted(remaining)[:8]}")


def proxy_index_sets(nb: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.arange(WINDOW_FIRST - 1, min(WINDOW_LAST, nb))
    proxy_q = np.arange(PROXY_FIRST - 1, min(PROXY_LAST, nb))
    proxy_q = proxy_q[(proxy_q >= 0) & (proxy_q < nb)]
    pi_idx = np.setdiff1d(window, proxy_q)
    return window, pi_idx, proxy_q


def frobenius_pi_r_q(r_cart: np.ndarray, pi_idx: np.ndarray, q_idx: np.ndarray) -> dict:
    out = {}
    for name, axis in (("x", 0), ("y", 1), ("z", 2)):
        block = r_cart[axis][np.ix_(pi_idx, q_idx)]
        out[f"pi_r{name}_q_frobenius_ang"] = float(np.linalg.norm(block))
        out[f"pi_r{name}_q_maxabs_ang"] = float(np.max(np.abs(block))) if block.size else 0.0
    pol = np.tensordot(POL_CART, r_cart, axes=(0, 0))
    block = pol[np.ix_(pi_idx, q_idx)]
    out["pi_rpol_q_frobenius_ang"] = float(np.linalg.norm(block))
    out["pi_rpol_q_maxabs_ang"] = float(np.max(np.abs(block))) if block.size else 0.0
    return out


def pair_table_from_rpol(r_pol_ang: np.ndarray, eps_p: np.ndarray, eps_q: np.ndarray) -> list[dict]:
    rows = []
    r_bohr = np.asarray(r_pol_ang) * ANG_TO_BOHR
    n_p, n_q = r_bohr.shape
    for i in range(n_p):
        for j in range(n_q):
            den_ev = abs(float(eps_q[j] - eps_p[i]))
            if den_ev < ETA_DENOM_FLOOR_EV:
                continue
            den_ha = den_ev * EV_TO_HA
            rabs = float(abs(r_bohr[i, j]))
            vabs = rabs * den_ha  # |v| a.u. = |Δε| |r| with ħ = 1
            rec = {
                "p_index_1based": i + 1,
                "q_index_1based": j + 1,
                "eps_p_eV": float(eps_p[i]),
                "eps_q_eV": float(eps_q[j]),
                "delta_e_eV": float(eps_q[j] - eps_p[i]),
                "r_pol_abs_ang": float(abs(r_pol_ang[i, j])),
                "r_pol_abs_bohr": rabs,
            }
            for label, intensity in FIELD_CASES:
                e0 = e0_au(intensity)
                a0 = a0_au(intensity)
                rec[f"eta_r_{label}"] = e0 * rabs / den_ha
                rec[f"eta_v_{label}"] = a0 * vabs / den_ha
            rows.append(rec)
    rows.sort(key=lambda r: r["eta_r_baseline_2e11"], reverse=True)
    return rows


def pair_table(r_cart: np.ndarray, eig_k: np.ndarray, pi_idx: np.ndarray, q_idx: np.ndarray) -> list[dict]:
    r_pol = np.tensordot(POL_CART, r_cart, axes=(0, 0))
    rows = pair_table_from_rpol(
        r_pol[np.ix_(pi_idx, q_idx)],
        np.asarray(eig_k)[pi_idx],
        np.asarray(eig_k)[q_idx],
    )
    for rec in rows:
        rec["p_qe_1based"] = int(pi_idx[rec["p_index_1based"] - 1] + 1)
        rec["q_qe_1based"] = int(q_idx[rec["q_index_1based"] - 1] + 1)
    return rows


def summarize_eta(rows: list[dict]) -> dict:
    if not rows:
        return {"n_pairs_denom_ge_floor": 0}
    summary = {"n_pairs_denom_ge_floor": len(rows), "delta_e_floor_eV": ETA_DENOM_FLOOR_EV}
    for label, intensity in FIELD_CASES:
        vals_r = np.array([r[f"eta_r_{label}"] for r in rows])
        vals_v = np.array([r[f"eta_v_{label}"] for r in rows])
        summary[label] = {
            "I_Wcm2": intensity,
            "E0_au": e0_au(intensity),
            "A0_au": a0_au(intensity),
            "eta_r_max": float(vals_r.max()),
            "eta_r_median": float(np.median(vals_r)),
            "n_eta_r_gt_0p1": int(np.sum(vals_r > 0.1)),
            "n_eta_r_gt_0p3": int(np.sum(vals_r > 0.3)),
            "n_eta_r_gt_1": int(np.sum(vals_r > 1.0)),
            "eta_v_max": float(vals_v.max()),
            "interpretation": (
                "eta << 1: complement mostly virtual; eta ~ 1: field mixes Π/Q"
            ),
        }
    return summary


def eta_r_stats_from_rpol(r_pol_ang: np.ndarray, eps_p: np.ndarray, eps_q: np.ndarray) -> dict:
    den_ev = np.abs(np.asarray(eps_q)[None, :] - np.asarray(eps_p)[:, None])
    r_bohr = np.abs(np.asarray(r_pol_ang)) * ANG_TO_BOHR
    ok = den_ev >= ETA_DENOM_FLOOR_EV
    den_ha = np.where(ok, den_ev * EV_TO_HA, np.inf)
    summary = {"n_pairs_denom_ge_floor": int(np.sum(ok)), "delta_e_floor_eV": ETA_DENOM_FLOOR_EV}
    for label, intensity in FIELD_CASES:
        e0 = e0_au(intensity)
        a0 = a0_au(intensity)
        eta_r = np.where(ok, e0 * r_bohr / den_ha, 0.0)
        eta_v = np.where(ok, a0 * r_bohr, 0.0)
        vals_r = eta_r[ok] if np.any(ok) else np.array([0.0])
        summary[label] = {
            "I_Wcm2": intensity,
            "E0_au": e0,
            "A0_au": a0,
            "eta_r_max": float(eta_r.max()) if np.any(ok) else 0.0,
            "eta_r_median": float(np.median(vals_r)),
            "n_eta_r_gt_0p1": int(np.sum(eta_r > 0.1)),
            "n_eta_r_gt_0p3": int(np.sum(eta_r > 0.3)),
            "n_eta_r_gt_1": int(np.sum(eta_r > 1.0)),
            "eta_v_max": float(eta_v.max()) if np.any(ok) else 0.0,
            "interpretation": (
                "eta << 1: complement mostly virtual; eta ~ 1: field mixes Π/Q"
            ),
        }
    return summary


def map_k20_to_wannier(nk: int = NK_SBE) -> list[dict]:
    mesh = regular_mp_kpoints()
    rows = []
    for iky in range(1, nk + 1):
        for ikx in range(1, nk + 1):
            frac = sbe_frac(ikx, iky, nk)
            mapped = nearest_k(frac, mesh)
            rows.append({"ikx": ikx, "iky": iky, "frac_k": list(frac), **mapped})
    return rows


def eta_max_true_pi_from_mats(nnkp: dict, eig_k: np.ndarray, k_1based: int, mats: list[np.ndarray], udis: dict) -> dict:
    kfrac = nnkp["kpoints"][k_1based - 1]
    nn_rows = nnkp["nnkpts"][k_1based - 1]
    bvecs = neighbor_bvecs(kfrac, nn_rows, nnkp["kpoints"], nnkp["recip_lattice"])
    weights = mv_shell_weights(bvecs)
    r_ang = berry_connection_ang(mats, bvecs, weights)
    require(udis["num_bands"] == r_ang.shape[1], "chk/mmn num_bands mismatch")
    comp = complement_basis(udis["u"], udis["lwindow"])
    u_eig, v_eig, eps_pi, eps_q = subspace_energy_basis(udis["u"], comp["v"], eig_k)
    norms, r_pol_block = frobenius_rect(r_ang, u_eig, v_eig)
    eta = eta_r_stats_from_rpol(r_pol_block, eps_pi, eps_q)
    gap = float(eps_q.min() - eps_pi.max()) if eps_pi.size and eps_q.size else None
    compact_eta = {}
    for label, _intensity in FIELD_CASES:
        block = eta[label]
        compact_eta[label] = {
            "eta_r_max": block["eta_r_max"],
            "eta_r_median": block["eta_r_median"],
            "n_eta_r_gt_0p1": block["n_eta_r_gt_0p1"],
            "n_eta_r_gt_0p3": block["n_eta_r_gt_0p3"],
            "n_eta_r_gt_1": block["n_eta_r_gt_1"],
            "eta_v_max": block["eta_v_max"],
        }
    return {
        "k_index_1based": k_1based,
        "k_frac": [float(x) for x in kfrac],
        "pi_rpol_q_frobenius_ang": norms["pi_rpol_q_frobenius_ang"],
        "pi_rpol_q_maxabs_ang": norms["pi_rpol_q_maxabs_ang"],
        "gap_pi_q_eV": gap,
        "eps_pi_max_eV": float(eps_pi.max()),
        "eps_q_min_eV": float(eps_q.min()),
        "n_pairs_denom_ge_floor": eta["n_pairs_denom_ge_floor"],
        "eta": compact_eta,
        "udis_orth_error": udis["orth_error"],
        "pi_trace": comp["pi_trace"],
        "q_trace": comp["q_trace"],
    }


def _field_grid_summary(rows: list[dict], label: str) -> dict:
    vals = np.array([r["eta"][label]["eta_r_max"] for r in rows], dtype=float)
    imax = int(np.argmax(vals))
    win = rows[imax]
    return {
        "eta_r_max_over_k20": float(vals.max()),
        "eta_r_min_over_k20": float(vals.min()),
        "eta_r_median_over_k20": float(np.median(vals)),
        "n_k_eta_r_gt_0p1": int(np.sum(vals > 0.1)),
        "n_k_eta_r_gt_0p3": int(np.sum(vals > 0.3)),
        "n_k_eta_r_gt_1": int(np.sum(vals > 1.0)),
        "argmax": {
            "ikx": win["ikx"],
            "iky": win["iky"],
            "frac_k": win["frac_k"],
            "wannier_k_index_1based": win["wannier_k_index_1based"],
            "frac_distance": win["frac_distance"],
            "eta_r_max": float(vals[imax]),
        },
    }


def scan_k20_eta_true_pi(nnkp: dict, eig: np.ndarray, mmn_path: Path, chk_path: Path) -> dict:
    maps = map_k20_to_wannier()
    needed = {int(r["wannier_k_index_1based"]) for r in maps}
    print(
        f"k20 eta scan: {len(maps)} SBE k -> {len(needed)} unique Wannier k",
        flush=True,
    )
    index, udis_raw = open_udis_store(chk_path)
    by_wannier: dict[int, dict] = {}
    t0 = time.time()
    for ik, mats in iterate_mmn_selected(mmn_path, needed):
        udis = unpack_udis(index, udis_raw, ik)
        rec = eta_max_true_pi_from_mats(nnkp, eig[ik - 1], ik, mats, udis)
        by_wannier[ik] = rec
        n_done = len(by_wannier)
        if n_done % 20 == 0 or n_done == len(needed):
            print(
                f"  eta: {n_done}/{len(needed)} Wannier k in {time.time() - t0:.1f}s",
                flush=True,
            )
    del udis_raw
    require(len(by_wannier) == len(needed), "incomplete Wannier eta map")
    k20_rows = []
    for item in maps:
        w = by_wannier[int(item["wannier_k_index_1based"])]
        k20_rows.append(
            {
                "ikx": item["ikx"],
                "iky": item["iky"],
                "frac_k": item["frac_k"],
                "wannier_k_index_1based": item["wannier_k_index_1based"],
                "wannier_k_frac": item["wannier_k_frac"],
                "frac_distance": item["frac_distance"],
                "exact_on_wannier_mesh": item["exact_on_wannier_mesh"],
                "pi_rpol_q_frobenius_ang": w["pi_rpol_q_frobenius_ang"],
                "gap_pi_q_eV": w["gap_pi_q_eV"],
                "eta": w["eta"],
            }
        )
    ranked = sorted(
        k20_rows, key=lambda r: r["eta"]["baseline_2e11"]["eta_r_max"], reverse=True
    )
    grid = {label: _field_grid_summary(k20_rows, label) for label, _intensity in FIELD_CASES}
    all_max = max(grid[label]["eta_r_max_over_k20"] for label, _ in FIELD_CASES)
    return {
        "status": "K20_ETA_SCAN_TRUE_UDIS",
        "rule": (
            "For each k20 point, map to nearest 24x24 Wannier k and compute "
            "eta_r^max of true Pi r Q (U_dis from chk). Not in-Pi |v_vc|^2."
        ),
        "n_k20": len(k20_rows),
        "n_unique_wannier_k": len(needed),
        "n_exact_on_wannier_mesh": int(sum(1 for r in k20_rows if r["exact_on_wannier_mesh"])),
        "max_frac_distance": float(max(r["frac_distance"] for r in k20_rows)),
        "projector": "TRUE_UDIS_FROM_CHK",
        "eta_r_max_any_field_over_k20": float(all_max),
        "all_k20_eta_r_max_lt_0p3": bool(
            all(grid[label]["n_k_eta_r_gt_0p3"] == 0 for label, _ in FIELD_CASES)
        ),
        "grid": grid,
        "top5_by_eta_r_baseline": ranked[:5],
        "set1_k458": next(
            r for r in k20_rows if r["wannier_k_index_1based"] == 458 and r["ikx"] == 17 and r["iky"] == 2
        ),
        "set2_gamma": next(
            r for r in k20_rows if r["ikx"] == 1 and r["iky"] == 1
        ),
        "per_k20": k20_rows,
    }


def stream_amn_at_k(path: Path, ik0: int, n_cr_d: int = 40) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.readline()
        nb, nk, nw = (int(x) for x in handle.readline().split()[:3])
        p_d = np.zeros(nb)
        p_p = np.zeros(nb)
        target = ik0 + 1
        for line in handle:
            parts = line.split()
            if int(parts[2]) != target:
                continue
            n = int(parts[0]) - 1
            m = int(parts[1]) - 1
            mag2 = float(parts[3]) ** 2 + float(parts[4]) ** 2
            if m < n_cr_d:
                p_d[n] += mag2
            else:
                p_p[n] += mag2
    return p_d, p_p


def screen_existing_channels(eig_k: np.ndarray, p_d: np.ndarray, p_p: np.ndarray) -> dict:
    mask = (eig_k >= ENERGY_SCREEN_LO) & (eig_k <= ENERGY_SCREEN_HI)
    idx = np.flatnonzero(mask)
    p_tot = p_d + p_p
    bands = []
    for i in idx:
        bands.append(
            {
                "qe_band_1based": int(i + 1),
                "energy_eV": float(eig_k[i]),
                "P_cr_d_trial": float(p_d[i]),
                "P_i_p_trial": float(p_p[i]),
                "P_tot_trial": float(p_tot[i]),
                "note": "trial-A overlap, not a normalized orbital percent; no I-d channel",
            }
        )
    return {
        "window_eV": [ENERGY_SCREEN_LO, ENERGY_SCREEN_HI],
        "n_states": int(idx.size),
        "P_tot_mean": float(p_tot[mask].mean()) if idx.size else 0.0,
        "P_cr_d_mean": float(p_d[mask].mean()) if idx.size else 0.0,
        "P_i_p_mean": float(p_p[mask].mean()) if idx.size else 0.0,
        "cannot_identify_I_d": True,
        "bands": bands,
    }


def velocity_from_hr(irvec: np.ndarray, ndegen: np.ndarray, h_r: np.ndarray, kfrac, real_lattice_ang: np.ndarray) -> np.ndarray:
    k = np.asarray(kfrac, dtype=float)
    phase = np.exp(2j * np.pi * (irvec @ k)) / ndegen
    r_cart = irvec.astype(float) @ real_lattice_ang
    nw = h_r.shape[1]
    vel = np.zeros((3, nw, nw), dtype=complex)
    for alpha in range(3):
        vel[alpha] = np.tensordot(1j * r_cart[:, alpha] * phase, h_r, axes=(0, 0))
        vel[alpha] = 0.5 * (vel[alpha] + vel[alpha].conj().T)
    return vel


def current_weight_at_k(irvec, ndegen, h_r, kfrac, real_lattice_ang, n_occ: int = N_OCC):
    k = np.asarray(kfrac, dtype=float)
    phase = np.exp(2j * np.pi * (irvec @ k)) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    hk = 0.5 * (hk + hk.conj().T)
    _evals, evecs = np.linalg.eigh(hk)
    r_cart = irvec.astype(float) @ real_lattice_ang
    vx = np.tensordot(1j * r_cart[:, 0] * phase, h_r, axes=(0, 0))
    vx = 0.5 * (vx + vx.conj().T)
    vx_band = evecs.conj().T @ vx @ evecs
    block = vx_band[:n_occ, n_occ:]
    return float(np.sum(np.abs(block) ** 2))


def register_set2_current_weight(
    tb_path: Path,
    real_lattice_ang: np.ndarray,
    nk: int = NK_SBE,
    n_occ: int = N_OCC,
    set1_sbe: dict | None = None,
) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from construct_p_conjugate_tb import read_tb_blocks

    print(f"set2: reading Hamiltonian block {tb_path}", flush=True)
    t_read = time.time()
    irvec, ndegen, h_r, _r = read_tb_blocks(tb_path, read_position=False)
    print(
        f"set2: tb nrpts={irvec.shape[0]} nwann={h_r.shape[1]} in {time.time() - t_read:.1f}s",
        flush=True,
    )
    require(h_r.shape[1] == N_WAN, f"TB nwann={h_r.shape[1]} != {N_WAN}")
    rows: list[dict] = []
    n_tot = nk * nk
    t0 = time.time()
    done = 0
    for iky in range(1, nk + 1):
        for ikx in range(1, nk + 1):
            frac = sbe_frac(ikx, iky, nk)
            weight = current_weight_at_k(irvec, ndegen, h_r, frac, real_lattice_ang, n_occ)
            rows.append(
                {"ikx": ikx, "iky": iky, "frac_k": list(frac), "weight_ex_v_interband": weight}
            )
            done += 1
            if done % 40 == 0 or done == n_tot:
                print(f"  set2: {done}/{n_tot} k-points in {time.time() - t0:.1f}s", flush=True)
    ranked = sorted(rows, key=lambda r: r["weight_ex_v_interband"], reverse=True)
    best = ranked[0]
    mapped = nearest_k(tuple(best["frac_k"]), regular_mp_kpoints())
    set1_on_grid = None
    if set1_sbe is not None:
        ikx1 = int(set1_sbe["ikx"])
        iky1 = int(set1_sbe["iky"])
        hit = next(r for r in rows if r["ikx"] == ikx1 and r["iky"] == iky1)
        rank = 1 + next(i for i, r in enumerate(ranked) if r["ikx"] == ikx1 and r["iky"] == iky1)
        set1_on_grid = {
            **hit,
            "rank_among_k20": rank,
            "n_k20": n_tot,
            "weight_ratio_to_set2_winner": float(
                hit["weight_ex_v_interband"] / best["weight_ex_v_interband"]
            ),
        }
    return {
        "rule": (
            "argmax_{k20} sum_{n<=84, m>84} |e_x · v_nm(k)|^2 from full112 TB dH/dk; "
            "not SBE H2-per-k (that file was never written)"
        ),
        "tb_name": tb_path.name,
        "tb_sha256": sha256_file(tb_path),
        "n_k20": n_tot,
        "n_occ": n_occ,
        "n_wan": N_WAN,
        "winner": {**best, **mapped},
        "top5": ranked[:5],
        "set1_sbe_k_on_same_metric": set1_on_grid,
        "status": "COMPUTED",
    }


def build_hotspots_only(repo: Path) -> dict:
    set1 = register_set1_edge_occupation(repo)
    report = {
        "schema": SCHEMA,
        "status": "HOTSPOTS_REGISTERED_OPERATORS_NOT_RUN",
        "definitions": {
            "Pi": "112-dimensional selected subspace projector; not inversion P",
            "Q": "I_window - Pi",
            "eta_r": "e E0 |e·r_pq| / |ε_q-ε_p| in atomic units",
            "eta_v": "e A0 |e·v_pq| / |ε_q-ε_p| in atomic units",
            "set1": "max edge occupation k",
            "set2": "max in-Π interband |e_x v|^2 on k20",
        },
        "laser": {
            "wvl_nm": WVL_NM,
            "omega0_au": OMEGA0_AU,
            "polarization": "E parallel Cartesian x (theta=0)",
            "fields_Wcm2": {name: intensity for name, intensity in FIELD_CASES},
        },
        "set1_edge_occupation": set1,
        "set2_current_weight": {
            "status": "NOT_RUN",
            "rule": (
                "argmax_{k20} sum_{n<=84, m>84} |e_x · v_nm(k)|^2 from TB dH/dk; "
                "requires CrI3_tb.dat Hamiltonian block"
            ),
        },
        "not_done": [
            "no new Slurm",
            "no Wannier expansion",
            "no GH7",
            "operators Pi r Q / eta need mmn at the registered k",
        ],
    }
    return report


def _energy_summary(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min_eV": float(arr.min()) if arr.size else None,
        "median_eV": float(np.median(arr)) if arr.size else None,
        "max_eV": float(arr.max()) if arr.size else None,
    }


def _proxy_operator_block(r_ang: np.ndarray, eig_k: np.ndarray) -> dict:
    _window, pi_idx, q_idx = proxy_index_sets(r_ang.shape[1])
    norms = frobenius_pi_r_q(r_ang, pi_idx, q_idx)
    pairs = pair_table(r_ang, eig_k, pi_idx, q_idx)
    r_bohr = r_ang * ANG_TO_BOHR
    de_ha = (eig_k[:, None] - eig_k[None, :]) * EV_TO_HA
    v_au = 1j * de_ha * r_bohr
    v_norms = frobenius_pi_r_q(v_au, pi_idx, q_idx)
    return {
        "projector": "QE169-200_COORDINATE_PROXY_FOR_Q",
        "proxy_note": (
            "Coordinate subspace QE169-200, not the true complement. "
            "Trace overlap with true Q is ~0.987 (2026-08-27 chk audit)."
        ),
        "n_pi": int(pi_idx.size),
        "n_q": int(q_idx.size),
        "pi_r_q": norms,
        "pi_v_q_from_commutator_au": {
            "pi_vpol_q_frobenius_au": v_norms["pi_rpol_q_frobenius_ang"],
            "pi_vx_q_frobenius_au": v_norms["pi_rx_q_frobenius_ang"],
            "pi_vy_q_frobenius_au": v_norms["pi_ry_q_frobenius_ang"],
        },
        "eta": summarize_eta(pairs),
        "top_pairs_by_eta_r_baseline": pairs[:15],
    }


def audit_operators_at_k(
    *,
    nnkp: dict,
    mmn_path: Path,
    eig: np.ndarray,
    k_1based: int,
    amn_path: Path | None = None,
    chk_path: Path | None = None,
) -> dict:
    kfrac = nnkp["kpoints"][k_1based - 1]
    nn_rows = nnkp["nnkpts"][k_1based - 1]
    bvecs = neighbor_bvecs(kfrac, nn_rows, nnkp["kpoints"], nnkp["recip_lattice"])
    weights = mv_shell_weights(bvecs)
    _nb, _nk, _nntot, headers, mats = read_mmn_for_k(mmn_path, k_1based)
    r_ang = berry_connection_ang(mats, bvecs, weights)
    eig_k = eig[k_1based - 1]
    require(eig_k.shape[0] == r_ang.shape[1], "eig/mmn band count mismatch")
    proxy = _proxy_operator_block(r_ang, eig_k)
    screen = None
    if amn_path is not None:
        print(f"amn: streaming trial overlaps at k={k_1based}", flush=True)
        p_d, p_p = stream_amn_at_k(amn_path, k_1based - 1)
        screen = screen_existing_channels(eig_k, p_d, p_p)

    out: dict[str, Any] = {
        "k_index_1based": k_1based,
        "k_frac_nnkp": [float(x) for x in kfrac],
        "mmn_neighbour_headers": [list(h) for h in headers],
        "mv_weight_sum": float(np.sum(weights)),
        "bvec_max_ang_inv": float(np.max(np.linalg.norm(bvecs, axis=1))),
        "coordinate_proxy": proxy,
        "existing_channel_screen_3p5_8eV": screen,
    }

    if chk_path is None:
        out["status"] = "OPERATORS_PROXY_ONLY_NO_CHK"
        out["projector"] = "QE169-200_COORDINATE_PROXY_FOR_Q"
        out["pi_r_q"] = proxy["pi_r_q"]
        out["eta"] = proxy["eta"]
        out["top_pairs_by_eta_r_baseline"] = proxy["top_pairs_by_eta_r_baseline"]
        out["pi_v_q_from_commutator_au"] = proxy["pi_v_q_from_commutator_au"]
        out["n_pi"] = proxy["n_pi"]
        out["n_q"] = proxy["n_q"]
        return out

    print(f"chk: loading U_dis at k={k_1based} from {chk_path}", flush=True)
    udis = load_udis_at_k(chk_path, k_1based)
    require(udis["num_bands"] == r_ang.shape[1], "chk/mmn num_bands mismatch")
    u = udis["u"]
    comp = complement_basis(u, udis["lwindow"])
    u_eig, v_eig, eps_pi, eps_q = subspace_energy_basis(u, comp["v"], eig_k)
    norms, r_pol_block = frobenius_rect(r_ang, u_eig, v_eig)
    r_bohr = r_ang * ANG_TO_BOHR
    de_ha = (eig_k[:, None] - eig_k[None, :]) * EV_TO_HA
    v_au = 1j * de_ha * r_bohr
    v_norms, _ = frobenius_rect(v_au, u_eig, v_eig)
    pairs = pair_table_from_rpol(r_pol_block, eps_pi, eps_q)
    for rec in pairs:
        rec["p_kind"] = "Pi_H_eigen"
        rec["q_kind"] = "Q_H_eigen"
    eta = summarize_eta(pairs)
    out.update(
        {
            "status": "OPERATORS_COMPUTED_TRUE_UDIS",
            "projector": "TRUE_UDIS_FROM_CHK",
            "proxy_note": (
                "Primary numbers use Pi = U_dis U_dis^dagger and "
                "Q = I_W - Pi. coordinate_proxy is a companion only."
            ),
            "n_pi": int(u.shape[1]),
            "n_q": int(comp["nq"]),
            "pi_r_q": norms,
            "pi_v_q_from_commutator_au": relabel_velocity_frobenius(v_norms),
            "eta": eta,
            "top_pairs_by_eta_r_baseline": pairs[:15],
            "eps_pi_eV": _energy_summary(eps_pi),
            "eps_q_eV": _energy_summary(eps_q),
            "udis_checks": {
                "chk_header": udis["chk_header"],
                "k_frac_chk": udis["k_frac"],
                "k_frac_nnkp_minus_chk_maxabs": float(
                    np.max(np.abs(np.asarray(kfrac) - np.asarray(udis["k_frac"])))
                ),
                "orth_error": udis["orth_error"],
                "ndimwin": udis["ndimwin"],
                "pi_trace": comp["pi_trace"],
                "q_trace": comp["q_trace"],
                "u_v_overlap_maxabs": comp["u_v_overlap_maxabs"],
                "v_gram_error": comp["v_gram_error"],
                "q_minus_iw_minus_pi_maxabs": comp["q_minus_iw_minus_pi_maxabs"],
                "q_projector_eig_min": comp["q_projector_eig_min"],
                "q_projector_eig_max": comp["q_projector_eig_max"],
            },
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hotspots-only", action="store_true")
    parser.add_argument("--mmn", type=Path, default=None)
    parser.add_argument("--nnkp", type=Path, default=None)
    parser.add_argument("--eig", type=Path, default=None)
    parser.add_argument("--amn", type=Path, default=None)
    parser.add_argument("--chk", type=Path, default=None)
    parser.add_argument("--tb", type=Path, default=None)
    parser.add_argument("--k", type=int, default=None, help="1-based Wannier k (default: set1 winner)")
    parser.add_argument("--set2", action="store_true", help="also rank TB current weight (reads Hamiltonian block)")
    parser.add_argument(
        "--set2-only",
        action="store_true",
        help="rank TB current weight; skip set1 mmn extract (use after set1 operators already exist)",
    )
    parser.add_argument(
        "--scan-k20-eta",
        action="store_true",
        help="scan eta_r^max(Pi,Q) on all k20 (true U_dis; one mmn pass)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = resolve_repo(args.repo)
    report = build_hotspots_only(repo)
    files = find_existing(("CrI3.mmn", "CrI3.nnkp", "CrI3.eig", "CrI3.amn", "CrI3.chk", "CrI3_tb.dat"))
    report["resolved_inputs"] = {name: str(path) for name, path in files.items()}
    report["chk_present"] = "CrI3.chk" in files or (args.chk is not None and args.chk.is_file())
    if args.hotspots_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps_strict(report), encoding="utf-8")
        print(f"wrote {args.out}")
        win = report["set1_edge_occupation"]["winner"]
        print(
            "set1",
            win["source"],
            "ikx",
            win["ikx"],
            "iky",
            win["iky"],
            "occ",
            win["occ"],
            "wannier_k",
            win["wannier_k_index_1based"],
        )
        return 0

    nnkp_path = args.nnkp or files.get("CrI3.nnkp")
    mmn_path = args.mmn or files.get("CrI3.mmn")
    eig_path = args.eig or files.get("CrI3.eig")
    amn_path = args.amn or files.get("CrI3.amn")
    require(nnkp_path is not None and nnkp_path.is_file(), "need CrI3.nnkp")
    require(mmn_path is not None and mmn_path.is_file(), "need CrI3.mmn")
    require(eig_path is not None and eig_path.is_file(), "need CrI3.eig")
    nnkp = parse_nnkp(nnkp_path)
    eig = load_eig(eig_path)
    chk_path = args.chk or files.get("CrI3.chk")
    if args.scan_k20_eta:
        require(chk_path is not None and chk_path.is_file(), "need CrI3.chk for --scan-k20-eta")
        report["k20_eta_scan"] = scan_k20_eta_true_pi(nnkp, eig, mmn_path, chk_path)
        report["status"] = "K20_ETA_SCAN_TRUE_UDIS"
        report["not_done"] = [
            "no new Slurm",
            "no Wannier expansion",
            "no GH7",
            "no expanded-parent HHG contrast",
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        body = dumps_strict(report)
        report["content_sha256_without_self"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        args.out.write_text(dumps_strict(report), encoding="utf-8")
        print(f"wrote {args.out}")
        scan = report["k20_eta_scan"]
        print("unique_wannier", scan["n_unique_wannier_k"], "all_lt_0p3", scan["all_k20_eta_r_max_lt_0p3"])
        for label, _intensity in FIELD_CASES:
            g = scan["grid"][label]
            arg = g["argmax"]
            print(
                label,
                "eta_r_max",
                g["eta_r_max_over_k20"],
                "n_k_gt_0p3",
                g["n_k_eta_r_gt_0p3"],
                "argmax",
                arg["ikx"],
                arg["iky"],
                "wannier",
                arg["wannier_k_index_1based"],
            )
        s1 = scan["set1_k458"]["eta"]["baseline_2e11"]["eta_r_max"]
        s2 = scan["set2_gamma"]["eta"]["baseline_2e11"]["eta_r_max"]
        print("check set1 k458 eta_r_baseline", s1)
        print("check set2 gamma eta_r_baseline", s2)
        return 0

    k1 = int(args.k or report["set1_edge_occupation"]["winner"]["wannier_k_index_1based"])
    run_set2 = bool(args.set2 or args.set2_only)
    skip_set1_ops = bool(args.set2_only)
    if skip_set1_ops:
        print(f"set2-only: skip set1 mmn at k={k1}", flush=True)
        report["set1_operators"] = {
            "status": "SKIPPED_REUSE_PRIOR_JSON",
            "prior_json": "results/full112_pi_rq_20260831/operators_set1_k458.json",
            "k_index_1based": k1,
        }
        report["status"] = "SET2_PENDING_OR_RUNNING"
    else:
        print(f"operators at Wannier k={k1}; chk={'yes' if chk_path else 'no'}", flush=True)
        report["set1_operators"] = audit_operators_at_k(
            nnkp=nnkp,
            mmn_path=mmn_path,
            eig=eig,
            k_1based=k1,
            amn_path=amn_path,
            chk_path=chk_path,
        )
        report["status"] = "SET1_OPERATORS_COMPUTED_TRUE_UDIS"
    report["not_done"] = [
        "no new Slurm",
        "no Wannier expansion",
        "no GH7",
        "set2 current-weight k not run" if not run_set2 else "set2 ran in this report",
    ]
    if run_set2:
        tb_path = args.tb or files.get("CrI3_tb.dat")
        require(tb_path is not None and tb_path.is_file(), "need CrI3_tb.dat for --set2")
        report["set2_current_weight"] = register_set2_current_weight(
            tb_path,
            nnkp["real_lattice_ang"],
            set1_sbe=report["set1_edge_occupation"]["winner"],
        )
        k2 = int(report["set2_current_weight"]["winner"]["wannier_k_index_1based"])
        print(
            "set2 winner ikx",
            report["set2_current_weight"]["winner"]["ikx"],
            "iky",
            report["set2_current_weight"]["winner"]["iky"],
            "wannier_k",
            k2,
            "weight",
            report["set2_current_weight"]["winner"]["weight_ex_v_interband"],
            flush=True,
        )
        if k2 != k1:
            print(f"set2 k={k2} != set1 k={k1}; computing Pi r Q / eta", flush=True)
            report["set2_operators"] = audit_operators_at_k(
                nnkp=nnkp,
                mmn_path=mmn_path,
                eig=eig,
                k_1based=k2,
                amn_path=amn_path,
                chk_path=chk_path,
            )
            report["status"] = "SET2_OPERATORS_COMPUTED_TRUE_UDIS"
        else:
            report["set2_operators"] = {"same_k_as_set1": True, "k_index_1based": k1}
            report["status"] = "SET2_SAME_K_AS_SET1"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = dumps_strict(report)
    report["content_sha256_without_self"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    args.out.write_text(dumps_strict(report), encoding="utf-8")
    print(f"wrote {args.out}")
    ops = report.get("set2_operators") if skip_set1_ops else report.get("set1_operators")
    if isinstance(ops, dict) and ops.get("eta"):
        print("status", report.get("status"), "projector", ops.get("projector"))
        eta = ops.get("eta") or {}
        for label, _intensity in FIELD_CASES:
            block = eta.get(label)
            if isinstance(block, dict):
                print(
                    label,
                    "eta_r_max",
                    block.get("eta_r_max"),
                    "n_gt_0.3",
                    block.get("n_eta_r_gt_0p3"),
                    "n_gt_1",
                    block.get("n_eta_r_gt_1"),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
