"""Shared-cache, block-worker, merge, validation, and budget runner.

The runner is deliberately an engineering wrapper around the already
validated midpoint propagator.  It does not alter the Hamiltonian, initial
state, detector, packet, gauge, or physical normalization.  The cache is
also intentionally explicit: a worker never rebuilds the Gamma12 grid, dark
basis, quadrature weights, or electronic vertices.

The command-line contract is:

``prepare``
    Build one manifest and one uncompressed shared cache.  ``--fixture`` and
    ``--source`` consume an existing backend fixture; ``--model`` builds a
    real finite-Gamma12 small-column continuous trajectory from the frozen
    model.  The latter is bounded by ``--max-columns`` and is not a
    production full-BZ run unless the user explicitly requests that size.
``worker``
    Propagate one explicit global column block for both driven and undriven
    paths, preserving the canonical stationary output row ``U0`` separately
    from the undriven current-source integral, and writing an atomically
    published block artifact.
``merge``
    Validate and combine complete block artifacts without accepting missing,
    duplicated, foreign, unfinished, or non-finite columns.  It also writes
    the finite-array ``n``, ``m``, and ``U0 V^T`` audit products; these are
    engineering outputs unless the manifest covers the complete physical
    input.
``validate``
    Run the fixture end-to-end comparison and explicit rejection tests.
``benchmark``
    Measure cold/warm preparation, driven/undriven device loops, transfers,
    and write costs for several block sizes.  The JSON keeps every repeat.

The file is self-contained so the same commands can be copied to a local GPU
or a server checkout.  Numerical operations remain delegated to the existing
validated backend modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

RUNNER_VERSION = "2026-10-06.r1"
ATOL = 1.0e-10
RTOL = 1.0e-8
TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_digest(payload: dict[str, Any]) -> str:
    basis = dict(payload)
    basis.pop("manifest_sha256", None)
    return _sha256_bytes(_json_bytes(basis))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ and publish it atomically.

    Uncompressed members are intentional here: workers should not repeatedly
    decompress a large shared cache while timing or slicing a block.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    try:
        np.savez(tmp, **arrays)
        with tmp.open("r+b") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def _peak_memory() -> dict[str, int | None]:
    """Best-effort process/GPU memory sample for the run ledger."""
    rss = None
    peak = None
    try:
        import psutil

        proc = psutil.Process()
        info = proc.memory_info()
        rss = int(info.rss)
        peak = int(getattr(info, "peak_wset", info.rss))
    except Exception:
        pass
    gpu_allocated = None
    gpu_reserved = None
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            gpu_allocated = int(torch.cuda.max_memory_allocated(dev))
            gpu_reserved = int(torch.cuda.max_memory_reserved(dev))
    except Exception:
        pass
    return {
        "rss_bytes": rss,
        "peak_working_set_bytes": peak,
        "gpu_peak_allocated_bytes": gpu_allocated,
        "gpu_peak_reserved_bytes": gpu_reserved,
    }


def _finite_arrays(arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Non-finite values in array {name}")


def _array_error(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "max_absolute": float("inf"),
            "max_relative": float("inf"),
            "allclose": False,
        }
    diff = np.asarray(a - b)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    scale = max(float(np.max(np.abs(b))) if b.size else 0.0, 1e-30)
    return {
        "shape": list(a.shape),
        "max_absolute": max_abs,
        "max_relative": max_abs / scale,
        "allclose": bool(np.all(np.abs(diff) <= ATOL + RTOL * np.abs(b))),
    }


def _source_code_hashes() -> dict[str, str]:
    names = (
        Path(__file__),
        TOOLS / "q8_m0_k1_packet.py",
        TOOLS / "q8_m0_classical.py",
        TOOLS / "q8_m0_column_step_torch_20261006.py",
        TOOLS / "q8_m0_continuum_columns_20261004.py",
        TOOLS / "q8_m0_p1_h2_finite_kcv_input_close_20261005.py",
        TOOLS / "q8_m0_p1_h2_finite_kcv_caller_fix_20261005.py",
        TOOLS / "q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006.py",
    )
    return {
        path.relative_to(REPO).as_posix(): _sha256_file(path)
        for path in names
        if path.exists()
    }



def _check_execution(manifest: dict[str, Any]) -> dict[str, str]:
    """Bind executed code separately from the code that generated a cache."""
    expected = manifest.get("execution_code_sha256")
    current = _source_code_hashes()
    if manifest.get("runner_version") != RUNNER_VERSION or expected != current:
        raise ValueError("Execution code mismatch; prepare/migrate an explicit new run, not old blocks.")
    return current


def _canonical_u0(cache: dict[str, np.ndarray], manifest: dict[str, Any]):
    """Use the frozen stationary port/packet map; never a time-integrated source.

    Photon A/free_A0 already carry sqrt(dnu).  Do not weight or normalize
    the selected subset again.  cache['u0'] is an electronic vertex, NOT U0.
    """
    q = _backend_modules().q
    from q8_m0_classical import p1_a2_protocol

    records = manifest["columns"]["records"]
    nph = int(manifest["physical_scope"]["n_photon_columns"])
    ndark = int(manifest["physical_scope"]["n_dark_columns"])
    if len(records) != nph + ndark:
        raise ValueError("Canonical output column count mismatch")
    if any(r["kind"] != "photon" for r in records[:nph]) or any(
        r["kind"] != "dark" for r in records[nph:]
    ):
        raise ValueError("Canonical reference requires manifest photon/dark order")
    labels = tuple((float(r["frequency"]), r["port"], int(r["polarization"]))
                   for r in records[:nph])
    photon = SimpleNamespace(initial=SimpleNamespace(A=cache["A"][:, :nph]),
                             free_A0=cache["free_A0"][:, :nph], labels=labels)
    bundle = SimpleNamespace(photon=photon, point=SimpleNamespace(A=cache["A"][:, nph:]),
                             n_photon_columns=nph, n_point_columns=ndark,
                             point_labels=tuple(("dark_combination", j) for j in range(ndark)))
    prot = p1_a2_protocol(float(manifest["physical_scope"].get("a_star", q.A_STAR)))
    omega = q.omega_grid(prot.omega0, prot.t_pulse)
    omega = omega[omega > 0]
    _, normalizer = q.normalize_phi(omega, q.phi_unnormalized(omega, 2 * prot.omega0, prot.t_pulse))
    row, _ = q.u0_from_bundle(bundle, cache["pol"],
                             lambda nu: q.phi_at(nu, prot, normalizer),
                             q.eta_of(float(cache["gamma"])))
    if row.shape != (2, len(records)) or not np.isfinite(row).all():
        raise ValueError("Invalid canonical output reference")
    return row, {"definition": "q.u0_from_bundle stationary right-output row",
                 "cache_key": "U0_canonical", "omega0": float(prot.omega0),
                 "t_pulse": float(prot.t_pulse), "phi_normalizer": float(normalizer),
                 "sqrt_dnu_already_in_columns": True, "subset_renormalized": False}


def _block_identity(manifest: dict[str, Any], window: str) -> dict[str, Any]:
    spec = _window_spec(manifest, window)
    identity = {"window": window, "t0": float(spec.get("t0", 0.0)),
                "h": float(manifest["time"]["h"]), "nstep": int(spec["nstep"]),
                "gamma": float(manifest["physical_scope"]["gamma"]),
                "spec": spec, "cache": manifest["shared_cache"]["sha256"]}
    return {"runner_version": RUNNER_VERSION, "window": window,
            "execution_code_sha256": manifest["execution_code_sha256"],
            "window_identity": _sha256_bytes(_json_bytes(identity)),
            "shared_cache_sha256": identity["cache"],
            "t0": identity["t0"], "h": identity["h"], "nstep": identity["nstep"],
            "gamma": identity["gamma"], "dtype": "complex128"}


def _cached_u0(manifest_path: Path, manifest: dict[str, Any]) -> np.ndarray:
    path = manifest_path.parent / manifest["shared_cache"]["path"]
    if _sha256_file(path) != manifest["shared_cache"]["sha256"]:
        raise ValueError("Shared cache hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        return np.array(data["U0_canonical"], copy=True)


def _backend_modules():
    import q8_m0_p1_h2_finite_kcv_gamma12_backend_short_20261006 as backend
    import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
    from q8_m0_column_step_torch_20261006 import (
        accumulate_torch,
        cpu_accumulate,
    )
    from q8_m0_continuum_columns_20261004 import ColumnState
    from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import free_A_increment

    return SimpleNamespace(
        backend=backend,
        q=q,
        accumulate_torch=accumulate_torch,
        cpu_accumulate=cpu_accumulate,
        ColumnState=ColumnState,
        free_A_increment=free_A_increment,
    )


class _PhotonBlock:
    """Photon reference view whose columns are already in global block order."""

    def __init__(
        self,
        omega: np.ndarray,
        free_A0: np.ndarray,
        labels: list[dict[str, Any]],
    ) -> None:
        self.omega = np.asarray(omega, float)
        self.free_A0 = np.asarray(free_A0, complex)
        self.labels = tuple(labels)
        if self.free_A0.shape != (2, self.omega.size):
            raise ValueError("Photon block free_A0 shape mismatch.")

    def free_A(self, t: float) -> np.ndarray:
        return self.free_A0 * np.exp(-1j * self.omega * float(t))[None, :]


def _block_free_A_increment(ref, t0: float, t1: float, n_point: int = 0):
    del n_point
    return ref.free_A(t1) - ref.free_A(t0)


def _labels_for_fixture(omega: np.ndarray, n_point: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    n_photon = int(omega.size)
    if n_photon % 4:
        raise ValueError("Fixture photon columns must be four columns per frequency.")
    for j, value in enumerate(np.asarray(omega, float)):
        slot = j % 4
        labels.append(
            {
                "column_id": j,
                "kind": "photon",
                "frequency": float(value),
                "port": "L" if slot < 2 else "R",
                "polarization": int(slot % 2),
                "local_index": j,
            }
        )
    for j in range(int(n_point)):
        labels.append(
            {
                "column_id": n_photon + j,
                "kind": "dark",
                "frequency": 0.0,
                "port": None,
                "polarization": None,
                "local_index": j,
            }
        )
    return labels


def _labels_for_selected_frequencies(
    frequencies: np.ndarray, n_dark: int
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for i, value in enumerate(np.asarray(frequencies, float)):
        for port, pol in (("L", 0), ("L", 1), ("R", 0), ("R", 1)):
            labels.append(
                {
                    "column_id": len(labels),
                    "kind": "photon",
                    "frequency": float(value),
                    "port": port,
                    "polarization": pol,
                    "frequency_index": int(i),
                }
            )
    n_photon = len(labels)
    for j in range(int(n_dark)):
        labels.append(
            {
                "column_id": n_photon + j,
                "kind": "dark",
                "frequency": 0.0,
                "port": None,
                "polarization": None,
                "dark_index": int(j),
            }
        )
    return labels


def _column_records(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in labels]


def _cache_arrays_from_fixture(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "b",
        "bdag",
        "A",
        "omega",
        "free_A0",
        "vd0_u",
        "vd0_d",
        "vu0_u",
        "vu0_d",
        "vdp_u",
        "vdp_d",
        "vup_u",
        "vup_d",
        "k0",
        "kp",
        "pol",
        "h",
        "gamma",
        "n_point",
        "t_pump",
        "oms",
        "wts",
    }
    missing = sorted(required - set(fixture))
    if missing:
        raise KeyError(f"Fixture missing arrays: {missing}")
    arrays = {key: np.asarray(value) for key, value in fixture.items()}
    _finite_arrays(arrays)
    ncol = int(arrays["b"].shape[1])
    npoint = int(np.asarray(arrays["n_point"]).item())
    nph = int(arrays["omega"].size)
    if ncol != nph + npoint:
        raise ValueError("Fixture initial columns do not equal photon+dark columns.")
    # The historical fixture stores omega/free_A0 only for photon columns.
    # Normalize the shared-cache schema once here so arbitrary global blocks
    # (including blocks crossing into the dark basis) need no special case.
    if arrays["free_A0"].shape != (2, nph):
        raise ValueError("Fixture free_A0 shape mismatch.")
    if npoint:
        arrays["omega"] = np.concatenate(
            (np.asarray(arrays["omega"], float), np.zeros(npoint, float))
        )
        arrays["free_A0"] = np.concatenate(
            (
                np.asarray(arrays["free_A0"], complex),
                np.zeros((2, npoint), complex),
            ),
            axis=1,
        )
    if arrays["A"].shape != (2, ncol):
        raise ValueError("Fixture A shape mismatch.")
    return arrays


def _fixture_manifest(
    run_root: Path,
    cache_path: Path,
    cache_sha: str,
    arrays: dict[str, np.ndarray],
    fixture_path: Path,
) -> dict[str, Any]:
    ncol = int(arrays["b"].shape[1])
    npoint = int(np.asarray(arrays["n_point"]).item())
    nph = ncol - npoint
    labels = _labels_for_fixture(arrays["omega"][:nph], npoint)
    nstep = int(np.asarray(arrays["vd0_u"]).shape[0])
    windows = {
        "start": {
            "t0": 0.0,
            "nstep": nstep,
            "kernel_key": "k0",
            "driven_u_key": "vd0_u",
            "driven_d_key": "vd0_d",
            "undriven_u_key": "vu0_u",
            "undriven_d_key": "vu0_d",
        },
        "pump": {
            "t0": float(np.asarray(arrays["t_pump"]).item()),
            "nstep": int(np.asarray(arrays["vdp_u"]).shape[0]),
            "kernel_key": "kp",
            "driven_u_key": "vdp_u",
            "driven_d_key": "vdp_d",
            "undriven_u_key": "vup_u",
            "undriven_d_key": "vup_d",
        },
    }
    payload: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "kind": "finite_gamma12_runner_manifest",
        "mode": "fixture",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "path": str(fixture_path),
            "sha256": _sha256_file(fixture_path),
        },
        "physical_scope": {
            "neel": "+N",
            "gamma": float(np.asarray(arrays["gamma"]).item()),
            "n_electronic": int(arrays["b"].shape[0]),
            "n_photon_columns": nph,
            "n_dark_columns": npoint,
            "n_columns": ncol,
            "complex_precision": "complex128",
            "preserve_driven_undriven": True,
            "sqrt_dnu_applied_once_in_fixture": True,
            "fixture_is_not_bz_convergence": True,
        },
        "time": {
            "h": float(np.asarray(arrays["h"]).item()),
            "windows": windows,
            "fixture_continuous_from_t0": False,
        },
        "quadrature": {
            "frequency_nodes": np.asarray(arrays["oms"], float).tolist(),
            "weights": np.asarray(arrays["wts"], float).tolist(),
            "weight_application": "sqrt(dnu) already applied once to photon columns",
        },
        "columns": {
            "global_order": "photon frequency, L/R, polarization, then dark basis",
            "records": _column_records(labels),
            "required_ids": list(range(ncol)),
        },
        "shared_cache": {
            "path": str(cache_path.relative_to(run_root)),
            "sha256": cache_sha,
            "format": "uncompressed_npz_loaded_once_per_worker",
        },
        "code_sha256": _source_code_hashes(),
        "windows": windows,
        "cache_contract": {
            "initial_arrays": ["b", "bdag", "A"],
            "raw_rows": "worker writes driven_raw_u, undriven_raw_u, driven_raw_v, undriven_raw_v",
            "u0_definition": "U0=canonical stationary output; U=U0+delta_U",
            "moments": "n=V* V^T; m=U V^T; retain U0 V^T",
        },
        "manifest_sha256": None,
    }
    payload["manifest_sha256"] = _manifest_digest(payload)
    return payload


def _choose_real_frequencies(oms: np.ndarray, max_photon_columns: int) -> np.ndarray:
    """Select real nodes across low/H2/mid/high, four columns per node."""
    max_nodes = max(int(max_photon_columns) // 4, 1)
    x = np.asarray(oms, float)
    bands = (
        (1.0e-3, 0.02, 0.20),
        (0.02, 0.30, 0.45),
        (0.30, 2.50, 0.20),
        (6.0, 12.0, 0.15),
    )
    picks: list[int] = []
    for lo, hi, fraction in bands:
        idx = np.flatnonzero((x >= lo) & (x < hi))
        take = min(idx.size, max(1, int(round(max_nodes * fraction))))
        if take:
            picks.extend(idx[np.unique(np.linspace(0, idx.size - 1, take).round().astype(int))].tolist())
    picks = sorted(set(picks))
    if len(picks) < max_nodes:
        remaining = [i for i in range(x.size) if i not in set(picks)]
        take = min(len(remaining), max_nodes - len(picks))
        if take:
            picks.extend(
                np.asarray(remaining, int)[
                    np.unique(np.linspace(0, len(remaining) - 1, take).round().astype(int))
                ].tolist()
            )
    picks = sorted(set(picks))[:max_nodes]
    if not picks:
        raise ValueError("No positive-frequency nodes selected.")
    return x[np.asarray(picks, int)]


def _real_model_cache(
    run_root: Path,
    *,
    nstep: int,
    ntpc: int,
    max_columns: int,
    n_dark: int,
    neel: int,
    a_star: float,
    n_panel: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a bounded real +N continuous cache from the frozen model."""
    mods = _backend_modules()
    be, q = mods.backend, mods.q
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_continuum_columns_20261004 import stationary_photon_columns
    from q8_m0_gsheet import gamma_self
    from q8_m0_incoming_spectral_completeness_20261004 import (
        dark_subspace,
        stationary_dark_basis_columns,
    )
    from q8_m0_model import build_geometry
    from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
    from q8_m0_s1_kernel import parallel_perp_basis

    started = time.perf_counter()
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    prot_d = p1_a2_protocol(float(a_star))
    prot_u = p1_a2_protocol(0.0)
    h = float(prot_d.period / int(ntpc))
    sample, _ = grid_sample(geo, 12, 0.0, neel=int(neel))
    oms, wts = q.input_quadrature(
        sample["gap"],
        sample["u"],
        gamma,
        n_panel=int(n_panel),
    )
    n_dark_keep = min(int(n_dark), int(sample["gap"].size))
    frequencies = _choose_real_frequencies(
        oms,
        max(4, int(max_columns) - n_dark_keep),
    )
    dnu = {round(float(o), 12): float(w) for o, w in zip(oms, wts)}
    photon = stationary_photon_columns(
        sample["gap"],
        sample["u"],
        sample["D"],
        gamma,
        frequencies,
    )
    class _Bundle:
        def __init__(self, ph):
            self.photon = ph
            self.n_photon_columns = int(ph.initial.b.shape[1])

    q.attach_weights(_Bundle(photon), dnu)
    basis, info, _ = dark_subspace(sample["gap"], sample["u"])
    dark_state, _, _ = stationary_dark_basis_columns(
        sample["gap"],
        sample["u"],
        basis,
        info,
    )
    n_dark_keep = min(n_dark_keep, int(dark_state.b.shape[1]))
    initial, n_point = be.pack_photon_dark(photon, dark_state, n_dark_keep)
    if initial.b.shape[1] > int(max_columns):
        keep = int(max_columns)
        initial = mods.ColumnState(
            initial.b[:, :keep],
            initial.bdag[:, :keep],
            initial.A[:, :keep],
        )
        n_point = max(0, keep - int(photon.initial.b.shape[1]))
    pol = np.vstack(parallel_perp_basis(geo))
    # Keep the whole requested trajectory in the shared cache.  This is the
    # continuous t=0 -> t_end path; no stationary reset is used.
    k, _ = be.kernel_at_times(
        gamma,
        prot_d,
        (np.arange(int(nstep)) + 0.5) * h,
    )
    vd = be.vertices_window(geo, sample, prot_d, int(nstep), h, 0)
    vu = be.vertices_window(geo, sample, prot_u, int(nstep), h, 0)
    arrays = {
        "b": np.asarray(initial.b, complex),
        "bdag": np.asarray(initial.bdag, complex),
        "A": np.asarray(initial.A, complex),
        "omega": np.concatenate(
            (
                np.asarray(photon.omega[: initial.b.shape[1] - n_point], float),
                np.zeros(int(n_point), float),
            )
        ),
        "free_A0": np.concatenate(
            (
                np.asarray(photon.free_A0[:, : initial.b.shape[1] - n_point], complex),
                np.zeros((2, int(n_point)), complex),
            ),
            axis=1,
        ),
        "driven_u": np.asarray([x[0] for x in vd], complex),
        "driven_d": np.asarray([x[1] for x in vd], complex),
        "undriven_u": np.asarray([x[0] for x in vu], complex),
        "undriven_d": np.asarray([x[1] for x in vu], complex),
        "k": np.asarray(k, complex),
        "pol": np.asarray(pol, complex),
        "h": np.asarray(h),
        "gamma": np.asarray(gamma),
        "n_point": np.asarray(n_point),
        "t_pump": np.asarray(0.0),
        "oms": np.asarray(oms, float),
        "wts": np.asarray(wts, float),
        "selected_frequencies": np.asarray(frequencies, float),
        "gaps": np.asarray(sample["gap"], float),
        "u0": np.asarray(sample["u"], complex),
        "D0": np.asarray(sample["D"], complex),
    }
    labels = _labels_for_selected_frequencies(frequencies, n_point)
    labels = labels[: int(initial.b.shape[1])]
    info = {
        "mode": "real_model_continuous",
        "neel": int(neel),
        "a_star": float(a_star),
        "gamma": gamma,
        "ntpc": int(ntpc),
        "nstep": int(nstep),
        "t_end": float(nstep * h),
        "n_panel": int(n_panel),
        "n_selected_frequencies": int(frequencies.size),
        "n_columns": int(initial.b.shape[1]),
        "n_dark": int(n_point),
        "labels": labels,
        "prepare_wall_s": time.perf_counter() - started,
        "continuous_from_t0": True,
        "full_frozen_p1_waveform": True,
        "packet_not_renormalized": True,
    }
    return arrays, info


def _real_manifest(
    run_root: Path,
    cache_path: Path,
    cache_sha: str,
    arrays: dict[str, np.ndarray],
    info: dict[str, Any],
) -> dict[str, Any]:
    labels = info["labels"]
    ncol = int(arrays["b"].shape[1])
    nstep = int(np.asarray(arrays["driven_u"]).shape[0])
    windows = {
        "continuous": {
            "t0": 0.0,
            "nstep": nstep,
            "kernel_key": "k",
            "driven_u_key": "driven_u",
            "driven_d_key": "driven_d",
            "undriven_u_key": "undriven_u",
            "undriven_d_key": "undriven_d",
        }
    }
    payload: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "kind": "finite_gamma12_runner_manifest",
        "mode": "real_model_continuous",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "path": "frozen local model builder",
            "sha256": None,
        },
        "physical_scope": {
            "neel": int(info["neel"]),
            "a_star": float(info["a_star"]),
            "gamma": float(info["gamma"]),
            "n_electronic": int(arrays["b"].shape[0]),
            "n_photon_columns": int(ncol - info["n_dark"]),
            "n_dark_columns": int(info["n_dark"]),
            "n_columns": ncol,
            "complex_precision": "complex128",
            "preserve_driven_undriven": True,
            "sqrt_dnu_applied_once_in_prepare": True,
            "fixture_is_not_bz_convergence": True,
        },
        "time": {
            "h": float(np.asarray(arrays["h"]).item()),
            "ntpc": int(info["ntpc"]),
            "nstep": nstep,
            "t_end": float(info["t_end"]),
            "windows": windows,
            "continuous_from_t0": True,
        },
        "quadrature": {
            "frequency_nodes": np.asarray(arrays["oms"], float).tolist(),
            "weights": np.asarray(arrays["wts"], float).tolist(),
            "selected_frequencies": np.asarray(arrays["selected_frequencies"], float).tolist(),
            "weight_application": "sqrt(dnu) applied once before block slicing",
        },
        "columns": {
            "global_order": "selected photon frequency, L/R, polarization, then dark basis",
            "records": _column_records(labels),
            "required_ids": list(range(ncol)),
        },
        "shared_cache": {
            "path": str(cache_path.relative_to(run_root)),
            "sha256": cache_sha,
            "format": "uncompressed_npz_loaded_once_per_worker",
        },
        "code_sha256": _source_code_hashes(),
        "windows": windows,
        "cache_contract": {
            "initial_arrays": ["b", "bdag", "A"],
            "raw_rows": "worker writes driven_raw_u, undriven_raw_u, driven_raw_v, undriven_raw_v",
            "u0_definition": "U0=canonical stationary output; U=U0+delta_U",
            "moments": "n=V* V^T; m=U V^T; retain U0 V^T",
        },
        "model_builder": info,
        "manifest_sha256": None,
    }
    payload["manifest_sha256"] = _manifest_digest(payload)
    return payload


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    expected = manifest.get("manifest_sha256")
    actual = _manifest_digest(manifest)
    if not expected or expected != actual:
        raise ValueError(f"Manifest digest mismatch: expected={expected}, actual={actual}")
    return manifest


def _load_manifest_cache(manifest_path: Path, manifest: dict[str, Any]):
    _check_execution(manifest)
    root = manifest_path.parent
    rel = Path(manifest["shared_cache"]["path"])
    cache_path = root / rel
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    got = _sha256_file(cache_path)
    expected = manifest["shared_cache"]["sha256"]
    if got != expected:
        raise ValueError(f"Shared cache hash mismatch: {got} != {expected}")
    return cache_path, _load_npz(cache_path)


def _write_manifest_and_cache(
    run_root: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    *,
    force: bool,
) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    cache_path = run_root / "shared_cache.npz"
    manifest_path = run_root / "manifest.json"
    if not force and (cache_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Refuse to overwrite existing run root files: {run_root}"
        )
    arrays = dict(arrays)
    arrays["U0_canonical"], manifest["output_reference"] = _canonical_u0(arrays, manifest)
    manifest["runner_version"] = RUNNER_VERSION
    manifest["execution_code_sha256"] = _source_code_hashes()
    manifest["cache_contract"]["u0_definition"] = "U0=canonical stationary output; U=U0+delta_U"
    _write_npz_atomic(cache_path, arrays)
    cache_sha = _sha256_file(cache_path)
    manifest["shared_cache"]["sha256"] = cache_sha
    manifest["shared_cache"]["path"] = cache_path.name
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _prepare(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    if bool(args.fixture) + bool(args.source) + bool(args.model) != 1:
        raise ValueError("Choose exactly one of --fixture, --source, or --model.")
    if args.fixture or args.source:
        source = Path(args.fixture or args.source).resolve()
        fixture = _cache_arrays_from_fixture(_load_npz(source))
        cache_path = run_root / "shared_cache.npz"
        # Build the manifest after the cache hash is known; the helper returns
        # a complete payload but its placeholder cache hash is replaced below.
        manifest = _fixture_manifest(run_root, cache_path, "", fixture, source)
        manifest_path = _write_manifest_and_cache(
            run_root, fixture, manifest, force=bool(args.force)
        )
    else:
        arrays, info = _real_model_cache(
            run_root,
            nstep=int(args.steps),
            ntpc=int(args.ntpc),
            max_columns=int(args.max_columns),
            n_dark=int(args.n_dark),
            neel=int(args.neel),
            a_star=float(args.a_star),
            n_panel=int(args.n_panel),
        )
        cache_path = run_root / "shared_cache.npz"
        manifest = _real_manifest(run_root, cache_path, "", arrays, info)
        manifest_path = _write_manifest_and_cache(
            run_root, arrays, manifest, force=bool(args.force)
        )
    saved = _read_manifest(manifest_path)
    saved_windows = saved.get("time", {}).get("windows", {})
    saved_nstep = saved.get("time", {}).get("nstep")
    if saved_nstep is None and saved_windows:
        saved_nstep = next(iter(saved_windows.values())).get("nstep")
    saved_continuous = saved.get("time", {}).get("continuous_from_t0")
    if saved_continuous is None:
        saved_continuous = saved.get("time", {}).get("fixture_continuous_from_t0")
    print(
        json.dumps(
            {
                "status": "prepared",
                "manifest": str(manifest_path),
                "manifest_sha256": saved["manifest_sha256"],
                "shared_cache": str(manifest_path.parent / saved["shared_cache"]["path"]),
                "shared_cache_sha256": saved["shared_cache"]["sha256"],
                "mode": saved["mode"],
                "n_columns": saved["physical_scope"]["n_columns"],
                "nstep": saved_nstep,
                "continuous_from_t0": saved_continuous,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _window_spec(manifest: dict[str, Any], window: str) -> dict[str, Any]:
    windows = manifest.get("windows", {})
    if window not in windows:
        raise KeyError(f"Unknown window {window}; available={sorted(windows)}")
    return windows[window]


def _block_inputs(
    manifest: dict[str, Any],
    cache: dict[str, np.ndarray],
    window: str,
    column_ids: Iterable[int],
):
    labels = manifest["columns"]["records"]
    required = manifest["columns"]["required_ids"]
    ncol = len(required)
    if required != list(range(ncol)):
        raise ValueError("Only contiguous global IDs are supported by this runner.")
    ids = np.asarray(list(column_ids), dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("A block must contain at least one column ID.")
    if len(set(int(x) for x in ids)) != int(ids.size):
        raise ValueError("Duplicate column IDs inside a block.")
    if np.any(ids < 0) or np.any(ids >= ncol):
        raise ValueError(f"Column ID outside manifest range [0,{ncol}).")
    spec = _window_spec(manifest, window)
    initial = _backend_modules().ColumnState(
        np.asarray(cache["b"][:, ids], complex),
        np.asarray(cache["bdag"][:, ids], complex),
        np.asarray(cache["A"][:, ids], complex),
    )
    ph = _PhotonBlock(
        np.asarray(cache["omega"][ids], float),
        np.asarray(cache["free_A0"][:, ids], complex),
        [labels[int(i)] for i in ids],
    )
    vertices = {
        "driven": (
            np.asarray(cache[spec["driven_u_key"]], complex),
            np.asarray(cache[spec["driven_d_key"]], complex),
        ),
        "undriven": (
            np.asarray(cache[spec["undriven_u_key"]], complex),
            np.asarray(cache[spec["undriven_d_key"]], complex),
        ),
    }
    nstep = int(spec["nstep"])
    for path, (u_arr, d_arr) in vertices.items():
        if u_arr.shape[0] != nstep or d_arr.shape[0] != nstep:
            raise ValueError(f"{path} vertex step count mismatch.")
        if u_arr.shape[1:] != (2, initial.b.shape[0]):
            raise ValueError(f"{path} vertex u shape mismatch.")
        if d_arr.shape[1:] != (2, 2):
            raise ValueError(f"{path} vertex D shape mismatch.")
    k = np.asarray(cache[spec["kernel_key"]], complex)
    if k.shape != (nstep,):
        raise ValueError("Kernel shape mismatch.")
    pol = np.asarray(cache["pol"], complex)
    return {
        "ids": ids,
        "labels": [labels[int(i)] for i in ids],
        "initial": initial,
        "u0_canonical": np.asarray(cache["U0_canonical"][:, ids], complex),
        "photon": ph,
        "vertices": vertices,
        "k": k,
        "pol": pol,
        "nstep": nstep,
        "h": float(np.asarray(cache["h"]).item()),
        "gamma": float(np.asarray(cache["gamma"]).item()),
        "t0": float(spec.get("t0", 0.0)),
    }


def _run_one(
    data: dict[str, Any],
    path: str,
    backend_name: str,
    mods,
) -> dict[str, Any]:
    if path not in ("driven", "undriven"):
        raise ValueError(path)
    started = time.perf_counter()
    initial = data["initial"].copy()
    photon = data["photon"]
    u_arr, d_arr = data["vertices"][path]
    vertices = list(zip(u_arr, d_arr))
    if backend_name == "cpu":
        rec = mods.cpu_accumulate(
            photon,
            data["gamma"],
            initial,
            0,
            data["k"],
            vertices,
            data["nstep"],
            data["h"],
            data["pol"],
            _block_free_A_increment,
            t0=data["t0"],
        )
    elif backend_name == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda is unavailable.")
        device = torch.device("cuda:0")
        rec = mods.accumulate_torch(
            photon,
            data["gamma"],
            initial,
            0,
            data["k"],
            vertices,
            data["nstep"],
            data["h"],
            data["pol"],
            _block_free_A_increment,
            torch,
            device,
            t0=data["t0"],
        )
    else:
        raise ValueError(backend_name)
    rec["wrapper_wall_s"] = time.perf_counter() - started
    return rec


def _worker(args: argparse.Namespace) -> int:
    worker_started = time.perf_counter()
    manifest_path = Path(args.manifest).resolve()
    cache_started = time.perf_counter()
    manifest = _read_manifest(manifest_path)
    cache_path, cache = _load_manifest_cache(manifest_path, manifest)
    cache_load_wall = time.perf_counter() - cache_started
    ncol_manifest = int(manifest["physical_scope"]["n_columns"])
    if args.column_ids:
        ids_arg = Path(args.column_ids)
        if ids_arg.exists():
            if ids_arg.suffix.lower() == ".npy":
                requested_ids = np.asarray(np.load(ids_arg), np.int64).ravel().tolist()
            else:
                requested_ids = json.loads(ids_arg.read_text(encoding="utf-8"))
        else:
            requested_ids = [
                int(item.strip())
                for item in str(args.column_ids).split(",")
                if item.strip()
            ]
        ids = np.asarray(requested_ids, np.int64)
        if ids.size == 0:
            raise ValueError("--column-ids did not contain any IDs.")
    else:
        if args.column_start is None or args.column_stop is None:
            raise ValueError("Provide both --column-start/--column-stop or --column-ids.")
        start, stop = int(args.column_start), int(args.column_stop)
        if start < 0 or stop <= start or stop > ncol_manifest:
            raise ValueError("Worker block range is outside manifest.")
        ids = np.arange(start, stop, dtype=np.int64)
    if len(set(int(x) for x in ids)) != int(ids.size):
        raise ValueError("Worker block contains duplicate IDs.")
    if np.any(ids < 0) or np.any(ids >= ncol_manifest):
        raise ValueError("Worker block contains an ID outside the manifest.")
    start, stop = int(ids.min()), int(ids.max()) + 1
    block_dir = manifest_path.parent / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    stem = f"block_{args.block_id}_{start:08d}_{stop:08d}"
    array_path = block_dir / f"{stem}.npz"
    meta_path = block_dir / f"{stem}.json"
    if not args.overwrite and (array_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Refuse to overwrite completed/partial block {stem}")
    data_started = time.perf_counter()
    data = _block_inputs(manifest, cache, args.window, ids)
    data_prep_wall = time.perf_counter() - data_started
    mods = _backend_modules()
    propagation_started = time.perf_counter()
    drec = _run_one(data, "driven", args.backend, mods)
    urec = _run_one(data, "undriven", args.backend, mods)
    propagation_wall = time.perf_counter() - propagation_started
    driven_u = np.asarray(drec["delta_u"], complex)
    undriven_u = np.asarray(urec["delta_u"], complex)
    driven_v = np.asarray(drec["v"], complex)
    undriven_v = np.asarray(urec["v"], complex)
    arrays = {
        "column_id": np.asarray(data["ids"], np.int64),
        "u0": data["u0_canonical"],
        "driven_raw_u": driven_u,
        "undriven_raw_u": undriven_u,
        "delta_u": driven_u - undriven_u,
        "driven_raw_v": driven_v,
        "undriven_raw_v": undriven_v,
        "v": driven_v - undriven_v,
        "driven_final_b": drec["state"].b,
        "driven_final_bdag": drec["state"].bdag,
        "driven_final_A": drec["state"].A,
        "undriven_final_b": urec["state"].b,
        "undriven_final_bdag": urec["state"].bdag,
        "undriven_final_A": urec["state"].A,
        "driven_final_I": np.asarray(drec["final_I"], complex),
        "undriven_final_I": np.asarray(urec["final_I"], complex),
    }
    _finite_arrays(arrays)
    array_write_started = time.perf_counter()
    tmp_array = array_path.parent / f".{array_path.name}.{os.getpid()}.tmp.npz"
    _write_npz_atomic(tmp_array, arrays)
    os.replace(tmp_array, array_path)
    array_write_wall = time.perf_counter() - array_write_started
    hash_started = time.perf_counter()
    artifact_sha = _sha256_file(array_path)
    artifact_hash_wall = time.perf_counter() - hash_started
    timing = {
        "cache_load_s": float(cache_load_wall),
        "worker_prepare_and_slice_s": float(data_prep_wall),
        "driven": {
            key: drec.get(key)
            for key in (
                "wall_s",
                "h2d_setup_s",
                "loop_s",
                "d2h_s",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        },
        "undriven": {
            key: urec.get(key)
            for key in (
                "wall_s",
                "h2d_setup_s",
                "loop_s",
                "d2h_s",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        },
        "sum_propagation_s": float(drec["wall_s"] + urec["wall_s"]),
        "propagation_wrapper_s": float(propagation_wall),
        "array_write_s": float(array_write_wall),
        "artifact_hash_s": float(artifact_hash_wall),
        "total_wrapper_s": float(time.perf_counter() - worker_started),
    }
    meta = {
        **_block_identity(manifest, args.window),
        "status": "complete",
        "runner_version": RUNNER_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "shared_cache_sha256": manifest["shared_cache"]["sha256"],
        "block_id": str(args.block_id),
        "window": str(args.window),
        "backend": str(args.backend),
        "dtype": "complex128",
        "column_start": start,
        "column_stop": stop,
        "column_ids": data["ids"].tolist(),
        "n_columns": int(data["ids"].size),
        "n_electronic": int(data["initial"].b.shape[0]),
        "nstep": int(data["nstep"]),
        "t0": float(data["t0"]),
        "h": float(data["h"]),
        "gamma": float(data["gamma"]),
        "u0_definition": "u0=canonical stationary output",
        "delta_u_definition": "driven_raw_u-undriven_raw_u",
        "v_definition": "driven_raw_v-undriven_raw_v",
        "artifact": {
            "array_path": array_path.name,
            "sha256": artifact_sha,
        },
        "timing": timing,
        "memory": _peak_memory(),
        "complete": True,
    }
    _write_json_atomic(meta_path, meta)
    print(
        json.dumps(
            {
                "status": "complete",
                "block": str(array_path),
                "metadata": str(meta_path),
                "backend": args.backend,
                "window": args.window,
                "column_range": [start, stop],
                "driven_s": drec["wall_s"],
                "undriven_s": urec["wall_s"],
                "sum_s": drec["wall_s"] + urec["wall_s"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _compare_blocks(args: argparse.Namespace) -> int:
    """Compare two completed block artifacts with identical global IDs."""
    left_meta_path = Path(args.left).resolve()
    right_meta_path = Path(args.right).resolve()
    if right_meta_path.suffix.lower() == ".npz":
        left_meta = json.loads(left_meta_path.read_text(encoding="utf-8"))
        left_arrays = _load_npz(left_meta_path.parent / left_meta["artifact"]["array_path"])
        right_arrays = _load_npz(right_meta_path)
        right_meta = json.loads(right_meta_path.with_suffix(".json").read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "window_identity", "execution_code_sha256"):
            if not left_meta.get(key) or left_meta[key] != right_meta.get(key):
                raise ValueError(f"Comparison identity mismatch: {key}")
        if not np.array_equal(left_arrays["column_id"], right_arrays["column_id"]):
            raise ValueError("Comparison column IDs mismatch")
        left_arrays["U"] = left_arrays["u0"] + left_arrays["delta_u"]
        names = {
            "U": "U",
            "u0": "u0",
            "delta_u": "delta_u",
            "v": "V",
        }
        errors = {
            right_name: _array_error(left_arrays[left_name], right_arrays[right_name])
            for left_name, right_name in names.items()
        }
        payload = {
            "status": "block_vs_merged_compare",
            "left": str(left_meta_path),
            "right": str(right_meta_path),
            "left_backend": left_meta.get("backend"),
            "right_kind": "merged",
            "window": left_meta.get("window"),
            "errors": errors,
            "ok": all(item["allclose"] for item in errors.values()),
            "tolerances": {"atol": ATOL, "rtol": RTOL},
        }
        out = Path(args.output).resolve() if args.output else right_meta_path.with_name(
            right_meta_path.stem + "_vs_" + left_meta_path.stem + ".json"
        )
        if out.exists() and not args.overwrite:
            raise FileExistsError(out)
        _write_json_atomic(out, payload)
        print(json.dumps({**payload, "output": str(out)}, indent=2), flush=True)
        return 0 if payload["ok"] else 1
    left_meta = json.loads(left_meta_path.read_text(encoding="utf-8"))
    right_meta = json.loads(right_meta_path.read_text(encoding="utf-8"))
    for key in ("manifest_sha256", "window_identity", "execution_code_sha256"):
        if not left_meta.get(key) or left_meta[key] != right_meta.get(key):
            raise ValueError(f"Comparison identity mismatch: {key}")
    if left_meta.get("column_ids") != right_meta.get("column_ids"):
        raise ValueError("CPU/GPU comparison requires identical global column IDs.")
    left_arrays = _load_npz(left_meta_path.parent / left_meta["artifact"]["array_path"])
    right_arrays = _load_npz(right_meta_path.parent / right_meta["artifact"]["array_path"])
    names = (
        "u0",
        "driven_final_b",
        "driven_final_bdag",
        "driven_final_A",
        "undriven_final_b",
        "undriven_final_bdag",
        "undriven_final_A",
        "driven_final_I",
        "undriven_final_I",
        "driven_raw_u",
        "undriven_raw_u",
        "delta_u",
        "v",
    )
    errors = {
        name: _array_error(left_arrays[name], right_arrays[name])
        for name in names
        if name in left_arrays and name in right_arrays
    }
    payload = {
        "status": "block_compare",
        "left": str(left_meta_path),
        "right": str(right_meta_path),
        "left_backend": left_meta.get("backend"),
        "right_backend": right_meta.get("backend"),
        "window": left_meta.get("window"),
        "column_ids": left_meta.get("column_ids"),
        "errors": errors,
        "ok": bool(errors) and all(item["allclose"] for item in errors.values()),
        "tolerances": {"atol": ATOL, "rtol": RTOL},
    }
    out = Path(args.output).resolve() if args.output else right_meta_path.with_name(
        right_meta_path.stem + "_vs_" + left_meta_path.stem + ".json"
    )
    if out.exists() and not args.overwrite:
        raise FileExistsError(out)
    _write_json_atomic(out, payload)
    print(json.dumps({**payload, "output": str(out)}, indent=2), flush=True)
    return 0 if payload["ok"] else 1


def _discover_blocks(manifest_path: Path) -> list[Path]:
    block_dir = manifest_path.parent / "blocks"
    if not block_dir.exists():
        return []
    return sorted(block_dir.glob("block_*.json"))


def _validate_block_metadata(
    manifest: dict[str, Any],
    manifest_path: Path,
    meta_paths: list[Path],
    *,
    require_all: bool,
    window: str | None = None,
) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    _check_execution(manifest)
    expected_u0 = _cached_u0(manifest_path, manifest)
    selected_window = window
    expected_n = int(manifest["physical_scope"]["n_columns"])
    expected_ids = set(int(x) for x in manifest["columns"]["required_ids"])
    if not meta_paths:
        raise ValueError("No block metadata supplied.")
    seen: set[int] = set()
    out = []
    for meta_path in meta_paths:
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("status") != "complete" or not meta.get("complete"):
            raise ValueError(f"Block is not complete: {meta_path}")
        if meta.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"Manifest mismatch: {meta_path}")
        if selected_window is None:
            selected_window = meta.get("window")
        if selected_window not in manifest["windows"] or meta.get("window") != selected_window:
            raise ValueError(f"Window mismatch: {meta_path}")
        for key, value in _block_identity(manifest, selected_window).items():
            if meta.get(key) != value:
                raise ValueError(f"Block identity mismatch ({key}): {meta_path}")
        ids = [int(x) for x in meta.get("column_ids", [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs inside block: {meta_path}")
        if seen.intersection(ids):
            raise ValueError(f"Duplicate IDs across blocks: {meta_path}")
        if not set(ids).issubset(expected_ids):
            raise ValueError(f"Unknown column ID in block: {meta_path}")
        array_path = meta_path.parent / meta["artifact"]["array_path"]
        if not array_path.exists():
            raise FileNotFoundError(array_path)
        if _sha256_file(array_path) != meta["artifact"]["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {array_path}")
        arrays = _load_npz(array_path)
        if arrays.get("column_id") is None or arrays["column_id"].tolist() != ids:
            raise ValueError(f"Column ID array mismatch: {array_path}")
        expected_names = {
            "column_id",
            "u0",
            "driven_raw_u",
            "undriven_raw_u",
            "delta_u",
            "driven_raw_v",
            "undriven_raw_v",
            "v",
        }
        if not expected_names.issubset(arrays):
            raise ValueError(f"Block missing output rows: {array_path}")
        n = len(ids)
        for name in expected_names - {"column_id"}:
            if arrays[name].shape != (2, n):
                raise ValueError(f"Shape mismatch {name} in {array_path}")
        _finite_arrays({k: v for k, v in arrays.items() if k != "column_id"})
        if not np.allclose(
            arrays["delta_u"],
            arrays["driven_raw_u"] - arrays["undriven_raw_u"],
            atol=ATOL,
            rtol=RTOL,
        ):
            raise ValueError(f"delta_u definition mismatch: {array_path}")
        if not np.allclose(arrays["v"], arrays["driven_raw_v"] - arrays["undriven_raw_v"], atol=ATOL, rtol=RTOL):
            raise ValueError(f"V definition mismatch: {array_path}")
        if not np.allclose(arrays["u0"], expected_u0[:, ids], atol=ATOL, rtol=RTOL):
            raise ValueError(f"Canonical U0 mismatch: {array_path}")
        seen.update(ids)
        out.append((meta, arrays))
    if require_all and seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise ValueError(f"Block coverage mismatch missing={missing[:8]} extra={extra[:8]}")
    return out


def _merge(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_manifest(manifest_path)
    meta_paths = [Path(x).resolve() for x in args.blocks] if args.blocks else _discover_blocks(manifest_path)
    records = _validate_block_metadata(
        manifest,
        manifest_path,
        meta_paths,
        require_all=bool(args.require_all),
        window=args.window,
    )
    ncol = int(manifest["physical_scope"]["n_columns"])
    placed: dict[str, np.ndarray] = {}
    for name in (
        "u0",
        "driven_raw_u",
        "undriven_raw_u",
        "delta_u",
        "driven_raw_v",
        "undriven_raw_v",
        "v",
    ):
        placed[name] = np.empty((2, ncol), complex)
    filled = np.zeros(ncol, dtype=bool)
    for meta, arrays in records:
        ids = np.asarray(meta["column_ids"], np.int64)
        for name in placed:
            placed[name][:, ids] = arrays[name]
        filled[ids] = True
    if args.require_all and not np.all(filled):
        raise ValueError("Merge requested all columns but some remain unfilled.")
    ids = np.flatnonzero(filled).astype(np.int64)
    out_arrays = {
        "column_id": ids,
        **{name: value[:, ids] for name, value in placed.items()},
    }
    u = out_arrays["u0"] + out_arrays["delta_u"]
    v = out_arrays["v"]
    u0 = out_arrays["u0"]
    n = v.conj() @ v.T
    m = u @ v.T
    u0vt = u0 @ v.T
    source_source = out_arrays["delta_u"] @ v.T
    out_arrays.update(
        {
            "U": u,
            "V": v,
            "n": n,
            "m": m,
            "m_U0VT": u0vt,
            "m_source_source": source_source,
        }
    )
    _finite_arrays(out_arrays)
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite merge output {output}")
    _write_npz_atomic(output, out_arrays)
    merge_meta = output.with_suffix(".json")
    actual_window = records[0][0]["window"]
    payload = {
        **_block_identity(manifest, actual_window),
        "status": "complete",
        "runner_version": RUNNER_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "window": actual_window,
        "require_all": bool(args.require_all),
        "n_columns_merged": int(ids.size),
        "n_columns_expected": ncol,
        "missing_column_ids": np.flatnonzero(~filled).tolist(),
        "blocks": [str(path) for path in meta_paths],
        "definitions": {
            "U0": "canonical stationary output",
            "U": "U0_canonical+delta_u",
            "V": "driven_raw_v-undriven_raw_v",
            "n": "V.conj() @ V.T",
            "m": "U @ V.T",
            "m_U0VT": "U0 @ V.T retained explicitly",
        },
        "audit": {
            "u_minus_u0_delta_u": _array_error(
                out_arrays["U"], out_arrays["u0"] + out_arrays["delta_u"]
            ),
            "normal_ccr_note": "partial column merge is not a physical CCR qualification",
            "production_physical_claim": False,
        },
    }
    _write_json_atomic(merge_meta, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "metadata": str(merge_meta),
                "n_columns_merged": int(ids.size),
                "n_columns_expected": ncol,
                "missing": payload["missing_column_ids"][:8],
                "m_maxabs": float(np.max(np.abs(m))) if m.size else 0.0,
                "u0vt_maxabs": float(np.max(np.abs(u0vt))) if u0vt.size else 0.0,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _validate_fixture(args: argparse.Namespace) -> int:
    """Exercise C1 and explicit failure modes without changing physics."""
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_manifest(manifest_path)
    if manifest["mode"] != "fixture":
        raise ValueError("validate currently targets the 16-step fixture.")
    cache_path, cache = _load_manifest_cache(manifest_path, manifest)
    ncol = int(manifest["physical_scope"]["n_columns"])
    block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    if sum(block_sizes) <= 0:
        raise ValueError("Need at least one positive block size.")
    # Use a new validation directory, never the worker's existing block files.
    validation_root = manifest_path.parent / "validation_c1"
    if validation_root.exists() and not args.overwrite:
        raise FileExistsError(validation_root)
    validation_root.mkdir(parents=True, exist_ok=True)
    block_meta: list[Path] = []
    start = 0
    index = 0
    while start < ncol:
        size = block_sizes[index % len(block_sizes)]
        stop = min(ncol, start + size)
        # Reuse the worker implementation in an isolated manifest directory
        # by creating a shallow cache copy and invoking its core directly.
        data = _block_inputs(
            manifest,
            cache,
            args.window,
            np.arange(start, stop, dtype=np.int64),
        )
        mods = _backend_modules()
        drec = _run_one(data, "driven", args.backend, mods)
        urec = _run_one(data, "undriven", args.backend, mods)
        arrays = {
            "column_id": data["ids"],
            "u0": data["u0_canonical"],
            "driven_raw_u": drec["delta_u"],
            "undriven_raw_u": urec["delta_u"],
            "delta_u": drec["delta_u"] - urec["delta_u"],
            "driven_raw_v": drec["v"],
            "undriven_raw_v": urec["v"],
            "v": drec["v"] - urec["v"],
        }
        stem = f"block_c1_{start:08d}_{stop:08d}"
        array_path = validation_root / f"{stem}.npz"
        meta_path = validation_root / f"{stem}.json"
        _write_npz_atomic(array_path, arrays)
        _write_json_atomic(
            meta_path,
            {
                **_block_identity(manifest, args.window),
                "status": "complete",
                "complete": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "shared_cache_sha256": manifest["shared_cache"]["sha256"],
                "block_id": stem,
                "window": args.window,
                "backend": args.backend,
                "column_start": start,
                "column_stop": stop,
                "column_ids": data["ids"].tolist(),
                "artifact": {
                    "array_path": array_path.name,
                    "sha256": _sha256_file(array_path),
                },
            },
        )
        block_meta.append(meta_path)
        start = stop
        index += 1
    records = _validate_block_metadata(
        manifest,
        manifest_path,
        block_meta,
        require_all=True,
        window=args.window,
    )
    merged: dict[str, np.ndarray] = {}
    for name in ("delta_u", "v", "u0"):
        merged[name] = np.empty((2, ncol), complex)
    for meta, arrays in records:
        ids = np.asarray(meta["column_ids"], int)
        for name in merged:
            merged[name][:, ids] = arrays[name]
    reference_name = "dU_start" if args.window == "start" else "dU_pump"
    reference_v_name = "v_start" if args.window == "start" else "v_pump"
    reference_path = Path(args.reference).resolve() if args.reference else None
    reference_errors: dict[str, Any] = {}
    if reference_path and reference_path.exists():
        ref = _load_npz(reference_path)
        reference_errors = {
            "delta_u": _array_error(merged["delta_u"], ref[reference_name]),
            "v": _array_error(merged["v"], ref[reference_v_name]),
        }
    reference_errors["canonical_u0"] = _array_error(merged["u0"], _canonical_u0(cache, manifest)[0])
    # Test coverage/manifest rejection without mutating the successful blocks.
    rejection: dict[str, Any] = {}
    try:
        _validate_block_metadata(
            manifest,
            manifest_path,
            block_meta[:-1],
            require_all=True,
            window=args.window,
        )
        rejection["missing_block_rejected"] = False
    except Exception as exc:
        rejection["missing_block_rejected"] = True
        rejection["missing_block_error"] = type(exc).__name__
    try:
        _validate_block_metadata(
            manifest,
            manifest_path,
            block_meta + [block_meta[0]],
            require_all=False,
            window=args.window,
        )
        rejection["duplicate_block_rejected"] = False
    except Exception as exc:
        rejection["duplicate_block_rejected"] = True
        rejection["duplicate_block_error"] = type(exc).__name__
    try:
        foreign = json.loads(block_meta[0].read_text(encoding="utf-8"))
        foreign["manifest_sha256"] = "foreign-manifest"
        foreign_path = validation_root / "foreign.json"
        _write_json_atomic(foreign_path, foreign)
        _validate_block_metadata(
            manifest,
            manifest_path,
            [foreign_path],
            require_all=False,
            window=args.window,
        )
        rejection["manifest_mismatch_rejected"] = False
    except Exception as exc:
        rejection["manifest_mismatch_rejected"] = True
        rejection["manifest_mismatch_error"] = type(exc).__name__
    ok = bool(
        all(item.get("allclose", False) for item in reference_errors.values())
        and all(
            rejection.get(key, False)
            for key in (
                "missing_block_rejected",
                "duplicate_block_rejected",
                "manifest_mismatch_rejected",
            )
        )
    )
    payload = {
        "status": "c1_fixture_validation",
        "ok": ok,
        "manifest": str(manifest_path),
        "cache": str(cache_path),
        "backend": args.backend,
        "window": args.window,
        "block_sizes": block_sizes,
        "block_metadata": [str(x) for x in block_meta],
        "reference_errors": reference_errors,
        "rejection_tests": rejection,
        "scope": {
            "fixture_steps": int(_window_spec(manifest, args.window)["nstep"]),
            "not_continuous_from_t0": True,
            "not_production": True,
        },
    }
    out = validation_root / "c1_validation.json"
    _write_json_atomic(out, payload)
    print(json.dumps({**payload, "out": str(out)}, indent=2), flush=True)
    return 0 if ok else 1


def _benchmark(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_manifest(manifest_path)
    cache_path, cache = _load_manifest_cache(manifest_path, manifest)
    mods = _backend_modules()
    sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    repeats = max(int(args.repeats), 1)
    warmups = max(int(args.warmups), 0)
    ncol = int(manifest["physical_scope"]["n_columns"])
    records: list[dict[str, Any]] = []
    for size in sizes:
        start, stop = 0, min(ncol, size)
        data = _block_inputs(
            manifest,
            cache,
            args.window,
            np.arange(start, stop, dtype=np.int64),
        )
        for warm in range(warmups):
            _run_one(data, "driven", args.backend, mods)
            _run_one(data, "undriven", args.backend, mods)
        for repeat in range(repeats):
            t0 = time.perf_counter()
            drec = _run_one(data, "driven", args.backend, mods)
            t1 = time.perf_counter()
            urec = _run_one(data, "undriven", args.backend, mods)
            t2 = time.perf_counter()
            records.append(
                {
                    "block_size": int(stop - start),
                    "column_start": start,
                    "column_stop": stop,
                    "repeat": repeat,
                    "backend": args.backend,
                    "window": args.window,
                    "driven": {
                        "wrapper_s": t1 - t0,
                        "wall_s": drec["wall_s"],
                        "h2d_setup_s": drec.get("h2d_setup_s"),
                        "loop_s": drec.get("loop_s"),
                        "d2h_s": drec.get("d2h_s"),
                    },
                    "undriven": {
                        "wrapper_s": t2 - t1,
                        "wall_s": urec["wall_s"],
                        "h2d_setup_s": urec.get("h2d_setup_s"),
                        "loop_s": urec.get("loop_s"),
                        "d2h_s": urec.get("d2h_s"),
                    },
                    "sum_wall_s": drec["wall_s"] + urec["wall_s"],
                    "memory_after": _peak_memory(),
                }
            )
    payload = {
        "status": "benchmark_complete",
        "runner_version": RUNNER_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "cache_sha256": manifest["shared_cache"]["sha256"],
        "backend": args.backend,
        "window": args.window,
        "warmups": warmups,
        "repeats": repeats,
        "records": records,
        "scope": {
            "preparation_once": True,
            "driven_and_undriven_measured_separately": True,
            "gpu_synchronization_delegated_to_accumulate_torch": args.backend == "cuda",
            "long_job_not_started": True,
            "full_9216_step_cost_not_claimed": True,
        },
    }
    out = Path(args.output).resolve()
    if out.exists() and not args.overwrite:
        raise FileExistsError(out)
    _write_json_atomic(out, payload)
    print(json.dumps({"status": "complete", "output": str(out), **payload}, indent=2), flush=True)
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing manifest, shared cache, blocks, and merged output.",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Numerical backend selected for worker execution.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finite Gamma12 shared-cache/block/merge runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="create manifest and shared cache")
    _add_common_arguments(prepare)
    group = prepare.add_mutually_exclusive_group()
    group.add_argument("--fixture", type=Path, help="validated short-test fixture")
    group.add_argument("--source", type=Path, help="NPZ using the fixture cache schema")
    group.add_argument(
        "--model",
        action="store_true",
        help="build a bounded real +N continuous cache from the frozen model",
    )
    prepare.add_argument("--steps", type=int, default=16)
    prepare.add_argument("--ntpc", type=int, default=1024)
    prepare.add_argument("--max-columns", type=int, default=1024)
    prepare.add_argument("--n-dark", type=int, default=64)
    prepare.add_argument("--neel", type=int, default=1)
    prepare.add_argument("--a-star", type=float, default=0.8)
    prepare.add_argument("--n-panel", type=int, default=96)
    prepare.add_argument("--force", action="store_true")

    worker = sub.add_parser("worker", help="run one explicit global column block")
    _add_common_arguments(worker)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--window", default="start")
    worker.add_argument("--block-id", required=True)
    worker_group = worker.add_mutually_exclusive_group()
    worker_group.add_argument("--column-start", type=int)
    worker_group.add_argument(
        "--column-ids",
        help="comma-separated IDs or a .npy/.json file with explicit global IDs",
    )
    worker.add_argument("--column-stop", type=int)
    worker.add_argument("--device", default=None)
    worker.add_argument("--overwrite", action="store_true")

    merge = sub.add_parser("merge", help="validate and combine completed blocks")
    _add_common_arguments(merge)
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--block", action="append", type=Path, dest="blocks")
    merge.add_argument("--require-all", action="store_true")
    merge.add_argument("--window", default=None)
    merge.add_argument("--overwrite", action="store_true")

    validate = sub.add_parser("validate", help="run the fixture C1 end-to-end audit")
    _add_common_arguments(validate)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--window", default="start")
    validate.add_argument("--block-sizes", default="64,128,256")
    validate.add_argument("--reference", type=Path)
    validate.add_argument("--overwrite", action="store_true")

    benchmark = sub.add_parser("benchmark", help="measure actual block costs")
    _add_common_arguments(benchmark)
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--window", default="start")
    benchmark.add_argument("--block-sizes", default="64,128,256")
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--overwrite", action="store_true")

    compare = sub.add_parser("compare", help="compare two completed block artifacts")
    _add_common_arguments(compare)
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "worker":
        return _worker(args)
    if args.command == "merge":
        return _merge(args)
    if args.command == "validate":
        return _validate_fixture(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "compare":
        return _compare_blocks(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
