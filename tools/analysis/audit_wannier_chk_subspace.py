#!/usr/bin/env python3
"""Read the disentanglement subspace from a Wannier90 3.x ``.chk`` file.

This is a read-only diagnostic for the classic unformatted sequential
checkpoint layout used by Wannier90 3.1.  It stops after ``u_matrix_opt``;
the much larger overlap-matrix record is never loaded.

For an outer window W (dimension ``ndimwin``) and the selected Wannier
subspace P = U_dis U_dis^dagger, the complement inside W is Q = I_W - P.
For a coordinate-band proxy S (for this project, QE bands 169--200), the
reported quantity

    Tr(S Q) / dim(Q)

is a descriptive subspace-overlap fraction.  No acceptance threshold is
invented here.  Principal-overlap eigenvalues of S Q S are also reported;
all ones would mean that the proxy and true complement coincide.

The record order follows the official Wannier90 v3.1.0 ``w90chk2chk.F90``.
This parser deliberately validates every record marker that it touches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


SCHEMA = "wannier_chk_subspace_audit_v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dumps_strict(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


class CheckpointFormatError(RuntimeError):
    """The checkpoint does not match the validated sequential layout."""


class FortranSequentialReader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        first = handle.read(4)
        if len(first) != 4:
            raise CheckpointFormatError("checkpoint is too short")
        little = struct.unpack("<I", first)[0]
        big = struct.unpack(">I", first)[0]
        if little == 33:
            self.endian = "<"
        elif big == 33:
            self.endian = ">"
        else:
            raise CheckpointFormatError(
                f"unsupported first record marker: little={little}, big={big}"
            )
        handle.seek(0)

    def _marker(self) -> int:
        raw = self.handle.read(4)
        if len(raw) != 4:
            raise CheckpointFormatError("unexpected EOF while reading record marker")
        return int(struct.unpack(self.endian + "I", raw)[0])

    def read_record(self, expected: int | None = None) -> bytes:
        size = self._marker()
        if expected is not None and size != expected:
            raise CheckpointFormatError(
                f"record length {size} != expected {expected} at offset "
                f"{self.handle.tell() - 4}"
            )
        data = self.handle.read(size)
        if len(data) != size:
            raise CheckpointFormatError("unexpected EOF inside record")
        closing = self._marker()
        if closing != size:
            raise CheckpointFormatError(
                f"record marker mismatch: opening={size}, closing={closing}"
            )
        return data

    def locate_record(self, expected: int) -> tuple[int, int]:
        size = self._marker()
        if size != expected:
            raise CheckpointFormatError(
                f"record length {size} != expected {expected} at offset "
                f"{self.handle.tell() - 4}"
            )
        offset = self.handle.tell()
        self.handle.seek(size, 1)
        closing = self._marker()
        if closing != size:
            raise CheckpointFormatError(
                f"record marker mismatch: opening={size}, closing={closing}"
            )
        return offset, size


def _int32(data: bytes, endian: str) -> int:
    if len(data) != 4:
        raise CheckpointFormatError(f"expected int32 record, got {len(data)} bytes")
    return int(struct.unpack(endian + "i", data)[0])


@dataclass(frozen=True)
class CheckpointIndex:
    path: Path
    endian: str
    header: str
    num_bands: int
    excluded_bands: tuple[int, ...]
    num_kpts: int
    mp_grid: tuple[int, int, int]
    kpoints: np.ndarray
    nntot: int
    num_wann: int
    checkpoint: str
    omega_invariant: float
    lwindow: np.ndarray
    ndimwin: np.ndarray
    udis_offset: int
    udis_nbytes: int
    u_matrix_offset: int = -1
    u_matrix_nbytes: int = 0


def index_checkpoint(path: Path) -> CheckpointIndex:
    path = path.resolve()
    with path.open("rb") as handle:
        reader = FortranSequentialReader(handle)
        endian = reader.endian
        header = reader.read_record(33).decode("ascii", errors="replace").rstrip()
        num_bands = _int32(reader.read_record(4), endian)
        num_exclude = _int32(reader.read_record(4), endian)
        if num_bands <= 0 or num_exclude < 0 or num_exclude >= num_bands:
            raise CheckpointFormatError(
                f"invalid band counts: num_bands={num_bands}, excluded={num_exclude}"
            )
        excluded_raw = reader.read_record(4 * num_exclude)
        excluded = tuple(
            int(x)
            for x in np.frombuffer(excluded_raw, dtype=endian + "i4", count=num_exclude)
        )
        reader.read_record(9 * 8)  # real lattice
        reader.read_record(9 * 8)  # reciprocal lattice
        num_kpts = _int32(reader.read_record(4), endian)
        mp = np.frombuffer(reader.read_record(3 * 4), dtype=endian + "i4").astype(int)
        kpoints = np.frombuffer(
            reader.read_record(num_kpts * 3 * 8), dtype=endian + "f8"
        ).reshape(num_kpts, 3).astype(float)
        nntot = _int32(reader.read_record(4), endian)
        num_wann = _int32(reader.read_record(4), endian)
        checkpoint = reader.read_record(20).decode("ascii", errors="replace").strip()
        have_disentangled = _int32(reader.read_record(4), endian) != 0
        if not have_disentangled:
            raise CheckpointFormatError("checkpoint has no disentanglement U matrix")
        omega_invariant = float(
            np.frombuffer(reader.read_record(8), dtype=endian + "f8", count=1)[0]
        )
        logical_raw = reader.read_record(num_bands * num_kpts * 4)
        lwindow = (
            np.frombuffer(logical_raw, dtype=endian + "i4")
            .reshape(num_kpts, num_bands)
            .astype(bool)
        )
        ndimwin = (
            np.frombuffer(reader.read_record(num_kpts * 4), dtype=endian + "i4")
            .astype(int)
        )
        expected_udis = num_bands * num_wann * num_kpts * 16
        udis_offset, udis_nbytes = reader.locate_record(expected_udis)
        u_matrix_offset = -1
        u_matrix_nbytes = 0
        leftover = handle.read(1)
        if leftover:
            handle.seek(-1, 1)
            # Wannier90 3.1 CrI3 chk: u_matrix immediately after u_matrix_opt,
            # then m_matrix.  Do not assume M comes first.
            expected_u = num_wann * num_wann * num_kpts * 16
            u_matrix_offset, u_matrix_nbytes = reader.locate_record(expected_u)

    if np.any(ndimwin != lwindow.sum(axis=1)):
        bad = np.flatnonzero(ndimwin != lwindow.sum(axis=1))[:5]
        raise CheckpointFormatError(f"ndimwin/lwindow mismatch at k indices {bad.tolist()}")
    if num_wann > int(ndimwin.min()):
        raise CheckpointFormatError(
            f"num_wann={num_wann} exceeds min ndimwin={int(ndimwin.min())}"
        )
    return CheckpointIndex(
        path=path,
        endian=endian,
        header=header,
        num_bands=num_bands,
        excluded_bands=excluded,
        num_kpts=num_kpts,
        mp_grid=tuple(int(x) for x in mp),
        kpoints=kpoints,
        nntot=nntot,
        num_wann=num_wann,
        checkpoint=checkpoint,
        omega_invariant=omega_invariant,
        lwindow=lwindow,
        ndimwin=ndimwin,
        udis_offset=udis_offset,
        udis_nbytes=udis_nbytes,
        u_matrix_offset=u_matrix_offset,
        u_matrix_nbytes=u_matrix_nbytes,
    )


def open_u_matrix_store(index: CheckpointIndex):
    """Memmap of inner MLWF unitary ``u_matrix(i,j,k)``.

    On the production CrI3 checkpoint this record sits *immediately after*
    ``u_matrix_opt``, then ``m_matrix``.  C-order shape is
    ``(num_kpts, num_wann, num_wann)`` with the last index matching Wannier90's
    first (fast) Fortran index.  Transpose each k-slice to get the conventional
    ``(row, col)`` matrix: ``U_prod = U_dis @ U_inner``.
    """
    if index.u_matrix_offset < 0 or index.u_matrix_nbytes <= 0:
        raise CheckpointFormatError("checkpoint has no u_matrix record")
    dtype = np.dtype(index.endian + "c16")
    return np.memmap(
        index.path,
        mode="r",
        dtype=dtype,
        offset=index.u_matrix_offset,
        shape=(index.num_kpts, index.num_wann, index.num_wann),
        order="C",
    )


def _summary(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def audit_subspace(
    index: CheckpointIndex,
    proxy_first: int,
    proxy_last: int,
    include_per_k: bool = True,
) -> dict:
    if not (1 <= proxy_first <= proxy_last <= index.num_bands):
        raise ValueError("proxy band range is outside num_bands")
    proxy = np.arange(proxy_first - 1, proxy_last, dtype=int)
    proxy_dim = int(proxy.size)
    dtype = np.dtype(index.endian + "c16")
    udis_raw = np.memmap(
        index.path,
        mode="r",
        dtype=dtype,
        offset=index.udis_offset,
        shape=(index.num_kpts, index.num_wann, index.num_bands),
        order="C",
    )

    orth_error = np.empty(index.num_kpts)
    compact_unused_max = np.empty(index.num_kpts)
    projector_trace = np.empty(index.num_kpts)
    complement_dim = index.ndimwin.astype(int) - index.num_wann
    proxy_in_window = np.empty(index.num_kpts, dtype=int)
    proxy_q_trace = np.empty(index.num_kpts)
    proxy_q_fraction = np.empty(index.num_kpts)
    off_proxy_q_trace = np.empty(index.num_kpts)
    proxy_p_trace = np.empty(index.num_kpts)
    principal_min = np.empty(index.num_kpts)
    principal_median = np.empty(index.num_kpts)
    principal_max = np.empty(index.num_kpts)
    per_k: list[dict] = []
    qdiag_all = np.zeros((index.num_kpts, index.num_bands), dtype=float)
    eye_w = np.eye(index.num_wann, dtype=complex)
    eye_proxy = np.eye(proxy_dim, dtype=complex)

    for ik in range(index.num_kpts):
        # Stored order is i(Bloch band) fastest, then j(Wannier), then k.
        # In Wannier90 3.1 the outer-window rows are COMPACTED into
        # u_matrix_opt(1:ndimwin,:,:); lwindow maps them back to QE bands.
        # See wannier90 issue #608 for the documented old/new convention.
        u_stored = np.asarray(udis_raw[ik]).T  # (num_bands, num_wann)
        nwin = int(index.ndimwin[ik])
        window_rows = np.flatnonzero(index.lwindow[ik])
        if window_rows.size != nwin:
            raise CheckpointFormatError(f"window row count mismatch at k={ik + 1}")
        compact = u_stored[:nwin, :]
        compact_unused_max[ik] = float(
            np.max(np.abs(u_stored[nwin:, :])) if nwin < index.num_bands else 0.0
        )
        u = np.zeros_like(u_stored)
        u[window_rows, :] = compact
        gram = compact.conj().T @ compact
        orth_error[ik] = float(np.max(np.abs(gram - eye_w)))
        pdiag = np.sum(np.abs(u) ** 2, axis=1).real
        projector_trace[ik] = float(np.sum(pdiag[index.lwindow[ik]]))
        qdiag = index.lwindow[ik].astype(float) - pdiag
        qdiag_all[ik] = qdiag
        proxy_in_window[ik] = int(np.count_nonzero(index.lwindow[ik, proxy]))
        q_trace = float(np.sum(qdiag[proxy]))
        proxy_q_trace[ik] = q_trace
        qdim = int(complement_dim[ik])
        proxy_q_fraction[ik] = q_trace / qdim if qdim > 0 else 0.0
        off_proxy_q_trace[ik] = float(np.sum(qdiag) - q_trace)
        proxy_p_trace[ik] = float(np.sum(pdiag[proxy]))

        # Eigenvalues of S Q S.  Outside-window proxy rows are excluded by
        # the coordinate-window projector before subtracting P.
        sw = np.diag(index.lwindow[ik, proxy].astype(float))
        u_proxy = u[proxy, :]
        sqs = sw - u_proxy @ u_proxy.conj().T
        sqs = 0.5 * (sqs + sqs.conj().T)
        eig = np.linalg.eigvalsh(sqs).real
        principal_min[ik] = float(eig.min())
        principal_median[ik] = float(np.median(eig))
        principal_max[ik] = float(eig.max())
        if include_per_k:
            per_k.append(
                {
                    "k_index_1based": ik + 1,
                    "k_frac": [float(x) for x in index.kpoints[ik]],
                    "ndimwin": int(index.ndimwin[ik]),
                    "complement_dim": qdim,
                    "proxy_bands_in_window": int(proxy_in_window[ik]),
                    "proxy_q_trace": q_trace,
                    "off_proxy_q_trace": float(off_proxy_q_trace[ik]),
                    "proxy_p_trace": float(proxy_p_trace[ik]),
                    "proxy_q_fraction_of_complement": float(proxy_q_fraction[ik]),
                    "principal_overlap_eigenvalue_min": float(principal_min[ik]),
                    "principal_overlap_eigenvalue_median": float(principal_median[ik]),
                    "principal_overlap_eigenvalue_max": float(principal_max[ik]),
                    "udis_orthonormality_max_abs": float(orth_error[ik]),
                }
            )

    del udis_raw
    band_resolved = []
    for band in range(index.num_bands):
        if np.any(index.lwindow[:, band]):
            band_resolved.append(
                {
                    "qe_band_1based": band + 1,
                    "in_window_fraction": float(np.mean(index.lwindow[:, band])),
                    "q_diagonal": _summary(qdiag_all[:, band]),
                }
            )
    first_window_band = np.array(
        [int(np.flatnonzero(row)[0]) + 1 for row in index.lwindow], dtype=int
    )
    last_window_band = np.array(
        [int(np.flatnonzero(row)[-1]) + 1 for row in index.lwindow], dtype=int
    )

    report = {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_SUBSPACE_AUDIT_NO_FITTED_THRESHOLD",
        "definitions": {
            "P": "U_dis U_dis^dagger in the QE Bloch basis",
            "Q": "I_window - P",
            "proxy": f"coordinate subspace of QE bands {proxy_first}-{proxy_last}",
            "proxy_q_fraction_of_complement": "Tr(S_proxy Q) / dim(Q)",
            "principal_overlap_eigenvalues": "eigenvalues of S_proxy Q S_proxy",
        },
        "checkpoint": {
            "path_name": index.path.name,
            "header": index.header,
            "endianness": "little" if index.endian == "<" else "big",
            "num_bands": index.num_bands,
            "num_wann": index.num_wann,
            "num_kpts": index.num_kpts,
            "mp_grid": list(index.mp_grid),
            "nntot": index.nntot,
            "excluded_bands": list(index.excluded_bands),
            "checkpoint_stage": index.checkpoint,
            "omega_invariant": index.omega_invariant,
            "udis_record_offset_bytes": index.udis_offset,
            "udis_record_size_bytes": index.udis_nbytes,
        },
        "window": {
            "ndimwin": _summary(index.ndimwin),
            "complement_dim": _summary(complement_dim),
            "constant_complement_dimension": bool(np.all(complement_dim == complement_dim[0])),
            "first_qe_band_1based": _summary(first_window_band),
            "last_qe_band_1based": _summary(last_window_band),
            "constant_contiguous_band_range": bool(
                np.all(first_window_band == first_window_band[0])
                and np.all(last_window_band == last_window_band[0])
                and np.all(index.ndimwin == last_window_band - first_window_band + 1)
            ),
        },
        "numerical_validation": {
            "udis_orthonormality_max_abs": _summary(orth_error),
            "compact_storage_unused_rows_max_abs": _summary(compact_unused_max),
            "projector_trace_in_window": _summary(projector_trace),
            "expected_projector_trace": index.num_wann,
        },
        "proxy_subspace": {
            "first_band_1based": proxy_first,
            "last_band_1based": proxy_last,
            "dimension": proxy_dim,
            "bands_in_window": _summary(proxy_in_window),
            "q_trace": _summary(proxy_q_trace),
            "q_fraction_of_complement": _summary(proxy_q_fraction),
            "off_proxy_q_trace": _summary(off_proxy_q_trace),
            "p_trace_inside_proxy": _summary(proxy_p_trace),
            "principal_overlap_eigenvalue_min": _summary(principal_min),
            "principal_overlap_eigenvalue_median": _summary(principal_median),
            "principal_overlap_eigenvalue_max": _summary(principal_max),
            "interpretation": (
                "Descriptive only: values near one support the proxy; this script does not "
                "turn any observed value into an acceptance threshold."
            ),
            "band_resolved_q_diagonal_within_outer_window": band_resolved,
        },
    }
    if include_per_k:
        report["per_k"] = per_k
    return report


def build_report(
    chk: Path,
    proxy_first: int,
    proxy_last: int,
    include_per_k: bool,
    hash_input: bool,
) -> dict:
    index = index_checkpoint(chk)
    report = audit_subspace(index, proxy_first, proxy_last, include_per_k)
    report["provenance"] = {
        "input_sha256": sha256_file(chk) if hash_input else None,
        "generator": str(Path(__file__).as_posix()),
        "generator_sha256": sha256_file(Path(__file__)),
        "format_reference": (
            "https://github.com/wannier-developers/wannier90/blob/"
            "v3.1.0/src/w90chk2chk.F90"
        ),
        "compact_row_mapping_reference": (
            "https://github.com/wannier-developers/wannier90/issues/608"
        ),
    }
    body = dumps_strict(report).encode("utf-8")
    report["content_sha256_without_self"] = hashlib.sha256(body).hexdigest()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chk", type=Path, required=True)
    parser.add_argument("--proxy-first", type=int, default=169)
    parser.add_argument("--proxy-last", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-per-k", action="store_true")
    parser.add_argument("--skip-input-sha", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        args.chk,
        args.proxy_first,
        args.proxy_last,
        include_per_k=not args.no_per_k,
        hash_input=not args.skip_input_sha,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_strict(report), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        "proxy_q_fraction_of_complement",
        report["proxy_subspace"]["q_fraction_of_complement"],
    )
    print(
        "udis_orthonormality_max_abs",
        report["numerical_validation"]["udis_orthonormality_max_abs"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
