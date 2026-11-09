#!/usr/bin/env python3
"""Synthetic gates for ``audit_wannier_chk_subspace.py``."""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from audit_wannier_chk_subspace import (
    CheckpointFormatError,
    audit_subspace,
    index_checkpoint,
    open_u_matrix_store,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _record(payload: bytes) -> bytes:
    marker = struct.pack("<I", len(payload))
    return marker + payload + marker


def _i4(values) -> bytes:
    return np.asarray(values, dtype="<i4").tobytes()


def _f8(values) -> bytes:
    return np.asarray(values, dtype="<f8").tobytes()


def _write_synthetic(path: Path) -> None:
    nb, nw, nk = 7, 3, 2
    # QE bands 2--6 are in the outer window; rows are stored compactly in
    # u_matrix_opt(1:5,:,:), as in Wannier90 3.1.
    lwindow = np.zeros((nk, nb), dtype=int)
    lwindow[:, 1:6] = 1
    parts = [
        _record(b"synthetic checkpoint".ljust(33)),
        _record(_i4([nb])),
        _record(_i4([0])),
        _record(b""),
        _record(_f8(np.eye(3).reshape(-1, order="F"))),
        _record(_f8(np.eye(3).reshape(-1, order="F"))),
        _record(_i4([nk])),
        _record(_i4([2, 1, 1])),
        _record(_f8([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])),
        _record(_i4([1])),
        _record(_i4([nw])),
        _record(b"postwann".ljust(20)),
        _record(_i4([1])),
        _record(_f8([0.125])),
        _record(_i4(lwindow)),
        _record(_i4([5, 5])),
    ]
    # Compact rows 1--3 map to QE bands 2--4.  QE bands 5--6 are exactly Q.
    u = np.zeros((nk, nw, nb), dtype="<c16")
    for ik in range(nk):
        u[ik, 0, 0] = 1.0
        u[ik, 1, 1] = 1.0
        u[ik, 2, 2] = 1.0
    parts.append(_record(u.tobytes(order="C")))
    path.write_bytes(b"".join(parts))


def test_exact_proxy_complement() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.chk"
        _write_synthetic(path)
        index = index_checkpoint(path)
        require(index.num_bands == 7, "num_bands")
        require(index.num_wann == 3, "num_wann")
        require(index.num_kpts == 2, "num_kpts")
        report = audit_subspace(index, 5, 6, include_per_k=True)
        frac = report["proxy_subspace"]["q_fraction_of_complement"]
        require(abs(frac["min"] - 1.0) < 1e-14, f"proxy fraction {frac}")
        eigmin = report["proxy_subspace"]["principal_overlap_eigenvalue_min"]
        require(abs(eigmin["min"] - 1.0) < 1e-14, f"principal overlap {eigmin}")
        orth = report["numerical_validation"]["udis_orthonormality_max_abs"]
        require(orth["max"] < 1e-14, f"orthogonality {orth}")
        trace = report["numerical_validation"]["projector_trace_in_window"]
        require(abs(trace["min"] - 3.0) < 1e-14, f"projector trace {trace}")
        require(report["window"]["first_qe_band_1based"]["min"] == 2.0, "window first")
        require(report["window"]["last_qe_band_1based"]["max"] == 6.0, "window last")
        require(report["window"]["constant_contiguous_band_range"], "window range")
        require(
            abs(report["proxy_subspace"]["off_proxy_q_trace"]["max"]) < 1e-14,
            "off-proxy Q trace",
        )
        require(len(report["per_k"]) == 2, "per-k rows")


def test_u_matrix_after_m_matrix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic_u.chk"
        nb, nw, nk, nntot = 7, 3, 2, 1
        lwindow = np.zeros((nk, nb), dtype=int)
        lwindow[:, 1:6] = 1
        parts = [
            _record(b"synthetic checkpoint".ljust(33)),
            _record(_i4([nb])),
            _record(_i4([0])),
            _record(b""),
            _record(_f8(np.eye(3).reshape(-1, order="F"))),
            _record(_f8(np.eye(3).reshape(-1, order="F"))),
            _record(_i4([nk])),
            _record(_i4([2, 1, 1])),
            _record(_f8([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])),
            _record(_i4([nntot])),
            _record(_i4([nw])),
            _record(b"postwann".ljust(20)),
            _record(_i4([1])),
            _record(_f8([0.125])),
            _record(_i4(lwindow)),
            _record(_i4([5, 5])),
        ]
        u_opt = np.zeros((nk, nw, nb), dtype="<c16")
        for ik in range(nk):
            u_opt[ik, 0, 0] = 1.0
            u_opt[ik, 1, 1] = 1.0
            u_opt[ik, 2, 2] = 1.0
        parts.append(_record(u_opt.tobytes(order="C")))
        u_inner = np.zeros((nk, nw, nw), dtype="<c16")
        for ik in range(nk):
            u_inner[ik] = np.eye(nw, dtype=np.complex128)
            # C layout is (j, i) with i fastest = Fortran u(i,j).
            u_inner[ik, 1, 0] = 0.2
            u_inner[ik, 0, 1] = -0.2
        parts.append(_record(u_inner.tobytes(order="C")))
        m_matrix = np.zeros((nk, nntot, nw, nw), dtype="<c16")
        parts.append(_record(m_matrix.tobytes(order="C")))
        path.write_bytes(b"".join(parts))
        index = index_checkpoint(path)
        require(index.u_matrix_offset > 0, "u_matrix_offset")
        store = open_u_matrix_store(index)
        got = np.asarray(store[0]).T.copy()
        del store
        require(abs(got[0, 1] - 0.2) < 1e-14, f"U_inner[0,1]={got[0, 1]}")
        require(abs(got[1, 0] + 0.2) < 1e-14, f"U_inner[1,0]={got[1, 0]}")
        require(abs(got[2, 2] - 1.0) < 1e-14, "U_inner diagonal")


def test_corrupt_marker_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.chk"
        _write_synthetic(path)
        payload = bytearray(path.read_bytes())
        payload[37:41] = struct.pack("<I", 32)  # wrong closing marker for header
        path.write_bytes(payload)
        try:
            index_checkpoint(path)
        except CheckpointFormatError:
            return
        raise RuntimeError("corrupt record marker was accepted")


def main() -> int:
    test_exact_proxy_complement()
    print("PASS test_exact_proxy_complement")
    test_u_matrix_after_m_matrix()
    print("PASS test_u_matrix_after_m_matrix")
    test_corrupt_marker_rejected()
    print("PASS test_corrupt_marker_rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
