"""Bounded-memory preparation for the frozen Gamma12 r1 equations.

Only storage changes: individual .npy files can be memory-mapped, and the
input columns and time-dependent vertices are assembled in bounded blocks.
No finite-frequency subset is normalized as a replacement for the full
frozen quadrature.  The canonical U0 is always the stationary output map.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

VERSION = "2026-10-07.local-r2"
R1_SHA256 = "224086831b4b79d94a8da9bd9b42b228fa293a1a1a991654210ceeef2c004998"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(payload: dict) -> str:
    content = dict(payload)
    content.pop("manifest_sha256", None)
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def execution_hashes() -> dict[str, str]:
    """Hash entry points and all statically imported repository dependencies.

    The older runner has dynamic imports inside functions; ast.walk also
    visits those.  This records the actual local dependency closure rather
    than silently omitting the Hamiltonian or spectral definitions.
    """
    r1 = TOOLS / "q8_m0_gamma12_runner_20261006.py"
    if sha256_file(r1) != R1_SHA256:
        raise ValueError("Frozen r1 runner hash changed; do not silently migrate")
    todo = [Path(__file__).resolve(), r1]
    local_runner = TOOLS / "q8_m0_gamma12_local_runner_20261007.py"
    if local_runner.exists():
        todo.append(local_runner)
    visited: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                dependency = TOOLS / (name.split(".")[0] + ".py")
                if dependency.is_file() and dependency not in visited:
                    todo.append(dependency.resolve())
    return {path.relative_to(REPO).as_posix(): sha256_file(path)
            for path in sorted(visited)}


def load_manifest(root: Path, verify_files: bool = False) -> dict:
    root = Path(root).resolve()
    path = root if root.is_file() else root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("manifest_sha256") != manifest_digest(payload):
        raise ValueError("Manifest version/hash mismatch")
    if payload.get("execution_code_sha256") != execution_hashes():
        raise ValueError("Executed code differs from the prepared manifest")
    for record in payload["files"].values():
        file = (path.parent / record["path"]).resolve()
        if not file.is_relative_to(path.parent) or not file.is_file():
            raise ValueError("Missing or outside-root cache file")
        if verify_files and sha256_file(file) != record["sha256"]:
            raise ValueError(f"Cache hash mismatch: {record['path']}")
    return payload


def load_arrays(root: Path, manifest: dict, mmap_mode: str = "r") -> dict:
    root = Path(root).resolve()
    if root.is_file():
        root = root.parent
    result = {}
    for name, record in manifest["files"].items():
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Cache path escapes run root")
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"Cache shape/dtype mismatch: {name}")
        result[name] = value
    return result


def _log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def _new_root(root: Path) -> Path:
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Preparation requires a new empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "arrays").mkdir()
    return root


def _new_array(root: Path, name: str, shape: tuple, dtype) -> None:
    value = np.lib.format.open_memmap(root / "arrays" / f"{name}.npy", mode="w+",
                                      dtype=dtype, shape=shape)
    value.flush()
    value._mmap.close()


def _write_slice(root: Path, name: str, key, value) -> None:
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite cache data: {name}")
    mapped = np.load(root / "arrays" / f"{name}.npy", mmap_mode="r+")
    try:
        mapped[key] = value
        mapped.flush()
    finally:
        mapped._mmap.close()


def _save_array(root: Path, name: str, value) -> None:
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite cache data: {name}")
    np.save(root / "arrays" / f"{name}.npy", value, allow_pickle=False)


def _file_records(root: Path) -> dict:
    records = {}
    for path in sorted((root / "arrays").glob("*.npy")):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(value.shape), str(value.dtype)
        value._mmap.close()
        records[path.stem] = {"path": path.relative_to(root).as_posix(),
                              "sha256": sha256_file(path), "shape": shape,
                              "dtype": dtype, "bytes": path.stat().st_size}
    return records


def _publish(root: Path, scope: dict, labels: list, timing: dict,
             *, t0: float = 0.0, mode: str, extra: dict | None = None) -> Path:
    _log("hashing completed cache files (streaming)")
    payload = {
        "version": VERSION, "mode": mode, "kind": "gamma12_local_memmap_manifest",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "r1_baseline_sha256": R1_SHA256,
        "execution_code_sha256": execution_hashes(), "files": _file_records(root),
        "physical_scope": scope,
        "time": {"t0": float(t0), "h": scope["h"], "ntpc": scope["ntpc"],
                 "nstep": scope["nstep"]},
        "columns": {"records": [dict(row, id=row["column_id"]) for row in labels],
                    "weighting": "sqrt(dnu) once in photon initial columns; dark mass 1"},
        "timing": timing,
        "definitions": {"U0": "frozen q.u0_from_bundle stationary right-output row",
                        "U": "U0_canonical + driven_raw_u - undriven_raw_u",
                        "V": "driven_raw_v - undriven_raw_v",
                        "physical_changes": False, "production_qualification": False},
    }
    if extra:
        payload.update(extra)
    payload["manifest_sha256"] = manifest_digest(payload)
    path = root / "manifest.json"
    write_json_atomic(path, payload)
    _log(f"manifest published: {path}")
    return path


def _vertex_files(root, geo, sample, protocol, name, nstep, h, flush_steps=128):
    """Identical quarter-point/half-frame order as frozen vertices_window."""
    from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import (
        advance_half_frame, vertices_from_frame,
    )
    nel = int(sample["gap"].size)
    _new_array(root, f"{name}_u", (nstep, 2, nel), np.complex128)
    _new_array(root, f"{name}_d", (nstep, 2, 2), np.complex128)
    frame = sample["frame"].copy()
    for start in range(0, nstep, flush_steps):
        stop = min(start + flush_steps, nstep)
        us = np.empty((stop - start, 2, nel), complex)
        ds = np.empty((stop - start, 2, 2), complex)
        for step in range(start, stop):
            t = step * h
            frame = advance_half_frame(geo, sample["kc"], protocol, sample["neel"],
                                       frame, t + h / 4.0, h)
            us[step-start], ds[step-start] = vertices_from_frame(
                geo, sample["kc"], protocol, sample["neel"], t + h / 2.0,
                frame, sample["weights"])
            frame = advance_half_frame(geo, sample["kc"], protocol, sample["neel"],
                                       frame, t + 3.0 * h / 4.0, h)
        _write_slice(root, f"{name}_u", slice(start, stop), us)
        _write_slice(root, f"{name}_d", slice(start, stop), ds)
        if start == 0 or stop % 1024 == 0 or stop == nstep:
            _log(f"{name} vertices: {stop}/{nstep}")


def prepare(run_root: Path, nstep: int = 9216, ntpc: int = 1024,
            n_panel: int = 96, neel: int = 1, block_size: int = 1024) -> Path:
    import q8_m0_gamma12_runner_20261006 as r1
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_continuum_columns_20261004 import stationary_photon_columns
    from q8_m0_gsheet import gamma_self
    from q8_m0_incoming_spectral_completeness_20261004 import (
        dark_subspace, stationary_dark_basis_columns,
    )
    from q8_m0_model import build_geometry
    from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
    from q8_m0_s1_kernel import parallel_perp_basis
    if min(nstep, ntpc, n_panel, block_size) <= 0 or neel not in (-1, 1):
        raise ValueError("Positive sizes and neel in (-1,1) required")
    code_before = execution_hashes()
    root = _new_root(run_root)
    start = time.perf_counter()
    be, q = r1._backend_modules().backend, r1._backend_modules().q
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    prot = p1_a2_protocol(q.A_STAR)
    h = float(prot.period / ntpc)
    sample, inversion = grid_sample(geo, 12, 0.0, neel=neel)
    _log("building full frozen input quadrature and complete dark basis")
    oms, wts = q.input_quadrature(sample["gap"], sample["u"], gamma, n_panel=n_panel)
    if np.any(wts <= 0) or np.any(np.diff(oms) <= 0):
        raise ValueError("Quadrature must be strictly increasing with positive weights")
    keys = np.round(oms, 12)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Weight-map frequency collision")
    basis, info, near = dark_subspace(sample["gap"], sample["u"])
    dark, dark_labels, dark_gaps = stationary_dark_basis_columns(
        sample["gap"], sample["u"], basis, info)
    nel, ndark = dark.b.shape
    nph, ncol = 4 * len(oms), 4 * len(oms) + ndark
    expected_bytes = 2 * ncol * nel * 16 + 2 * nstep * 2 * nel * 16
    if shutil.disk_usage(root).free < expected_bytes * 1.1 + 512 * 1024**2:
        raise OSError("Insufficient free disk for complete column/vertex cache")
    _log(f"full input: {len(oms)} frequencies, {nph} photon + {ndark} dark = {ncol} columns")
    for name, shape, dtype in (
        ("b_cols", (ncol, nel), complex), ("bdag_cols", (ncol, nel), complex),
        ("A_cols", (ncol, 2), complex), ("free_A0_cols", (ncol, 2), complex),
        ("omega", (ncol,), float), ("U0_canonical", (2, ncol), complex),
    ):
        _new_array(root, name, shape, dtype)
    pol = np.asarray(np.vstack(parallel_perp_basis(geo)), complex)
    om_packet = q.omega_grid(prot.omega0, prot.t_pulse)
    om_packet = om_packet[om_packet > 0]
    _, normalizer = q.normalize_phi(om_packet, q.phi_unnormalized(
        om_packet, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: q.phi_at(nu, prot, normalizer)
    eta = q.eta_of(gamma)
    freq_block = max(1, block_size // 4)
    empty_point = SimpleNamespace(A=np.zeros((2, 0), complex))
    for first in range(0, len(oms), freq_block):
        last = min(first + freq_block, len(oms))
        frequencies = oms[first:last]
        photon = stationary_photon_columns(sample["gap"], sample["u"], sample["D"],
                                           gamma, frequencies)
        bundle = SimpleNamespace(photon=photon, point=empty_point,
                                 n_photon_columns=4 * len(frequencies), n_point_columns=0,
                                 point_labels=())
        q.attach_weights(bundle, {round(float(o), 12): float(w)
                                  for o, w in zip(frequencies, wts[first:last])})
        u0, _ = q.u0_from_bundle(bundle, pol, phi_of, eta)
        sl = slice(4 * first, 4 * last)
        for name, value in (("b_cols", photon.initial.b.T),
                            ("bdag_cols", photon.initial.bdag.T),
                            ("A_cols", photon.initial.A.T),
                            ("free_A0_cols", photon.free_A0.T), ("omega", photon.omega)):
            _write_slice(root, name, sl, value)
        _write_slice(root, "U0_canonical", (slice(None), sl), u0)
        if first == 0 or last == len(oms) or last % (freq_block * 8) == 0:
            _log(f"initial photon columns: {4 * last}/{nph}")
        del bundle, photon
    photon_empty = SimpleNamespace(initial=SimpleNamespace(A=np.zeros((2, 0), complex)),
                                   free_A0=np.zeros((2, 0), complex), labels=())
    bundle_dark = SimpleNamespace(photon=photon_empty, point=dark, n_photon_columns=0,
                                  n_point_columns=ndark, point_labels=dark_labels)
    u0_dark, _ = q.u0_from_bundle(bundle_dark, pol, phi_of, eta)
    for first in range(0, ndark, block_size):
        last = min(first + block_size, ndark)
        sl = slice(nph + first, nph + last)
        for name, value in (("b_cols", dark.b[:, first:last].T),
                            ("bdag_cols", dark.bdag[:, first:last].T),
                            ("A_cols", dark.A[:, first:last].T),
                            ("free_A0_cols", np.zeros((last-first, 2), complex)),
                            ("omega", np.zeros(last-first))):
            _write_slice(root, name, sl, value)
        _write_slice(root, "U0_canonical", (slice(None), sl), u0_dark[:, first:last])
    dark_current = float(np.max(np.abs(sample["u"].conj() @ basis)))
    del dark, basis, bundle_dark
    t_columns = time.perf_counter() - start
    _log("building unchanged detector kernel over full requested time window")
    kernel, kernel_normalizer = be.kernel_at_times(gamma, prot, (np.arange(nstep)+0.5)*h)
    for name, value in {"k": kernel, "pol": pol, "h": h, "gamma": gamma,
                        "gaps": sample["gap"], "u0": sample["u"], "D0": sample["D"],
                        "oms": oms, "wts": wts, "dark_gaps": dark_gaps,
                        "kf": sample["kf"], "kc": sample["kc"],
                        "weights": sample["weights"]}.items():
        _save_array(root, name, value)
    t_kernel = time.perf_counter()
    _vertex_files(root, geo, sample, prot, "driven", nstep, h)
    t_driven = time.perf_counter()
    _vertex_files(root, geo, sample, p1_a2_protocol(0.0), "undriven", nstep, h)
    if execution_hashes() != code_before:
        raise ValueError("Code changed during preparation; manifest not published")
    scope = {"neel": int(neel), "a_star": float(q.A_STAR), "gamma": gamma,
             "n_electronic": nel, "n_photon_columns": nph, "n_dark_columns": ndark,
             "n_columns": ncol, "ntpc": ntpc, "nstep": nstep, "h": h,
             "n_panel": n_panel, "full_quadrature": True, "nk": 12, "shift": 0.0,
             "continuous_from_t0": True, "all_dark_columns": True,
             "finite_input_band": [float(oms[0]), float(oms[-1])],
             "full_quadrature_meaning": "all frozen finite positive-frequency quadrature nodes, not infinite-band qualification",
             "block_size": block_size, "complex_precision": "complex128"}
    timing = {"columns_and_static_s": t_columns,
              "kernel_s": t_kernel - start - t_columns,
              "driven_vertices_s": t_driven-t_kernel,
              "undriven_vertices_s": time.perf_counter()-t_driven,
              "before_hash_s": time.perf_counter()-start}
    return _publish(root, scope, r1._labels_for_selected_frequencies(oms, ndark), timing,
                    mode="complete_finite_gamma12",
                    extra={"preparation_checks": {"inversion": inversion,
                           "dark_current_maxabs": dark_current,
                           "near_degenerate_pair_count": len(near),
                           "packet_normalizer": float(normalizer),
                           "kernel_packet_normalizer": float(kernel_normalizer),
                           "packet_mass": q.packet_mass_dnu(oms, wts, prot, normalizer)}})


def prepare_fixture(run_root: Path, fixture_path: Path, window: str = "start") -> Path:
    """Storage-only conversion of the frozen fixture, for independent regression."""
    import q8_m0_gamma12_runner_20261006 as r1
    start = time.perf_counter()
    execution_hashes()
    source = Path(fixture_path).resolve()
    with np.load(source, allow_pickle=False) as data:
        arrays = r1._cache_arrays_from_fixture({key: np.asarray(data[key]) for key in data.files})
    # The frozen manifest stores the cache path relative to the run root.
    # The fixture itself stays outside that root; only its hash is recorded.
    old = r1._fixture_manifest(Path(run_root), Path(run_root) / "source_fixture.npz",
                               sha256_file(source), arrays, source)
    spec = old["windows"][window]
    row, row_info = r1._canonical_u0(arrays, old)
    root = _new_root(run_root)
    converted = {"b_cols": arrays["b"].T, "bdag_cols": arrays["bdag"].T,
                 "A_cols": arrays["A"].T, "free_A0_cols": arrays["free_A0"].T,
                 "omega": arrays["omega"], "U0_canonical": row,
                 "driven_u": arrays[spec["driven_u_key"]],
                 "driven_d": arrays[spec["driven_d_key"]],
                 "undriven_u": arrays[spec["undriven_u_key"]],
                 "undriven_d": arrays[spec["undriven_d_key"]],
                 "k": arrays[spec["kernel_key"]], "pol": arrays["pol"],
                 "h": arrays["h"], "gamma": arrays["gamma"],
                 "oms": arrays["oms"], "wts": arrays["wts"]}
    for name in ("gaps", "u0", "D0"):
        if name in arrays:
            converted[name] = arrays[name]
    for name, value in converted.items():
        _save_array(root, name, value)
    scope = dict(old["physical_scope"], ntpc=1024, nstep=spec["nstep"],
                 h=float(arrays["h"]), full_quadrature=False, a_star=0.8,
                 n_panel=None)
    return _publish(root, scope, old["columns"]["records"],
                    {"prepare_s": time.perf_counter()-start}, mode="fixture",
                    t0=spec["t0"], extra={"canonical_reference": row_info,
                                         "source_fixture_sha256": sha256_file(source),
                                         "fixture_window": window})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9216)
    parser.add_argument("--ntpc", type=int, default=1024)
    parser.add_argument("--n-panel", type=int, default=96)
    parser.add_argument("--neel", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()
    prepare(args.run_root, args.steps, args.ntpc, args.n_panel, args.neel, args.block_size)
