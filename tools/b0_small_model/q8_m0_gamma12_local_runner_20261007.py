"""Memory-bounded local Γ12 runner. Frozen r1 equations, new storage only.

Independent input columns are batched; every column includes every k coordinate.
No low-rank truncation, no precision change, no replacement of canonical U0.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import q8_m0_gamma12_runner_20261006 as r1
import q8_m0_gamma12_local_prepare_20261007 as prep

VERSION = "2026-10-07.local-r2"
_VERIFIED = {}
ROW_NAMES = ("u0", "driven_raw_u", "undriven_raw_u", "delta_u",
             "driven_raw_v", "undriven_raw_v", "v")


def import_fixture(source, root, window="start"):
    return prep.prepare_fixture(Path(root), Path(source), window=window)


def _load(root):
    root = Path(root).resolve()
    manifest = prep.load_manifest(root)
    identity = tuple((key, (root / rec["path"]).stat().st_size,
                      (root / rec["path"]).stat().st_mtime_ns)
                     for key, rec in sorted(manifest["files"].items()))
    tag = (str(root), manifest["manifest_sha256"], identity)
    if tag not in _VERIFIED:
        prep.load_manifest(root, verify_files=True)
        _VERIFIED.clear()
        _VERIFIED[tag] = True
    return manifest, prep.load_arrays(root, manifest)


def _ids(manifest, ids):
    ids = np.asarray(list(ids), np.int64)
    n = manifest["physical_scope"]["n_columns"]
    if ids.ndim != 1 or not ids.size or np.unique(ids).size != ids.size:
        raise ValueError("Empty/duplicate input column IDs")
    if ids.min() < 0 or ids.max() >= n or np.any(np.diff(ids) <= 0):
        raise ValueError("IDs must be unique, sorted, and inside manifest")
    return ids


def block_inputs(manifest, arrays, ids, nstep_override=None):
    ids = _ids(manifest, ids)
    nstep = int(manifest["time"]["nstep"])
    if nstep_override is not None:
        if not 0 < int(nstep_override) <= nstep:
            raise ValueError("Invalid short-test step count")
        nstep = int(nstep_override)
    mods = r1._backend_modules()
    records = manifest["columns"]["records"]
    state = mods.ColumnState(np.array(arrays["b_cols"][ids].T, copy=True),
                             np.array(arrays["bdag_cols"][ids].T, copy=True),
                             np.array(arrays["A_cols"][ids].T, copy=True))
    photon = r1._PhotonBlock(np.array(arrays["omega"][ids]),
                            np.array(arrays["free_A0_cols"][ids].T),
                            [records[int(i)] for i in ids])
    return dict(ids=ids, initial=state, photon=photon,
                u0_canonical=np.array(arrays["U0_canonical"][:, ids]),
                vertices={p: (arrays[p+"_u"][:nstep], arrays[p+"_d"][:nstep])
                          for p in ("driven", "undriven")},
                k=np.array(arrays["k"][:nstep], copy=True),
                pol=np.array(arrays["pol"], copy=True), nstep=nstep,
                h=float(manifest["time"]["h"]),
                gamma=float(manifest["physical_scope"]["gamma"]),
                t0=float(manifest["time"].get("t0", 0.0)))


def worker(run_root, ids, backend="cuda", block_id="local", nstep_override=None):
    root = Path(run_root).resolve()
    started = time.perf_counter()
    manifest, cache = _load(root)
    data = block_inputs(manifest, cache, ids, nstep_override)
    ids = data["ids"]
    if not block_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid block identifier")
    stem = f"block_{block_id}_{ids[0]:08d}_{ids[-1]+1:08d}"
    array_path = root / "blocks" / (stem + ".npz")
    meta_path = array_path.with_suffix(".json")
    if meta_path.exists() and array_path.exists():
        raise FileExistsError(f"Refuse overwrite: {array_path}")
    # A crash after the array write and before the sidecar leaves an orphan.
    # A completed pair is never deleted; run_all skips those after validation.
    if array_path.exists() or meta_path.exists():
        array_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    preparation_s = time.perf_counter() - started
    mods = r1._backend_modules()
    drec = r1._run_one(data, "driven", backend, mods)
    urec = r1._run_one(data, "undriven", backend, mods)
    arrays = {"column_id": ids, "u0": data["u0_canonical"],
              "driven_raw_u": drec["delta_u"], "undriven_raw_u": urec["delta_u"],
              "delta_u": drec["delta_u"]-urec["delta_u"],
              "driven_raw_v": drec["v"], "undriven_raw_v": urec["v"],
              "v": drec["v"]-urec["v"]}
    for name, rec in (("driven", drec), ("undriven", urec)):
        for field in ("b", "bdag", "A"):
            arrays[name+"_final_"+field] = getattr(rec["state"], field)
        arrays[name+"_final_I"] = rec["final_I"]
    r1._finite_arrays(arrays)
    write_started = time.perf_counter()
    r1._write_npz_atomic(array_path, arrays)
    write_s = time.perf_counter()-write_started
    digest = r1._sha256_file(array_path)
    timing = {"prepare_load_slice_s": preparation_s, "array_write_s": write_s,
              "driven": {k: drec.get(k) for k in ("wall_s", "loop_s", "h2d_setup_s", "d2h_s", "peak_allocated_bytes", "peak_reserved_bytes")},
              "undriven": {k: urec.get(k) for k in ("wall_s", "loop_s", "h2d_setup_s", "d2h_s", "peak_allocated_bytes", "peak_reserved_bytes")},
              "sum_propagation_s": drec["wall_s"]+urec["wall_s"],
              "total_wrapper_s": time.perf_counter()-started}
    meta = dict(version=VERSION, status="complete", complete=True, backend=backend,
                manifest_sha256=manifest["manifest_sha256"],
                execution_code_sha256=manifest["execution_code_sha256"],
                column_ids=ids.tolist(), n_columns=int(ids.size),
                n_electronic=int(data["initial"].b.shape[0]),
                nstep=data["nstep"], h=data["h"], t0=data["t0"], gamma=data["gamma"],
                u0_definition="canonical stationary output",
                artifact={"array_path": array_path.name, "sha256": digest},
                timing=timing, memory=r1._peak_memory(),
                finite_partial_block_not_physical_result=True)
    r1._write_json_atomic(meta_path, meta)
    print(json.dumps({"block": stem, "columns": len(ids), "steps": data["nstep"],
                      "wall_s": timing["total_wrapper_s"]}), flush=True)
    return meta_path


def validate_block(root, manifest, cache, meta_path, expected_steps=None):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "complete" or meta.get("complete") is not True:
        raise ValueError("Incomplete block")
    for k in ("manifest_sha256", "execution_code_sha256"):
        if meta.get(k) != manifest[k]:
            raise ValueError("Block identity mismatch: "+k)
    if meta.get("version") != VERSION or meta.get("u0_definition") != "canonical stationary output":
        raise ValueError("Block version/U0 definition mismatch")
    for k, value in (("h", manifest["time"]["h"]),
                     ("t0", manifest["time"].get("t0", 0.0)),
                     ("gamma", manifest["physical_scope"]["gamma"])):
        if meta.get(k) != value:
            raise ValueError("Time/physics mismatch: "+k)
    steps = int(manifest["time"]["nstep"] if expected_steps is None else expected_steps)
    if meta.get("nstep") != steps:
        raise ValueError("Block step count mismatch")
    ids = _ids(manifest, meta.get("column_ids", []))
    path = meta_path.parent / meta["artifact"]["array_path"]
    if path.parent.resolve() != meta_path.parent.resolve():
        raise ValueError("Artifact outside block directory")
    if r1._sha256_file(path) != meta["artifact"]["sha256"]:
        raise ValueError("Artifact hash mismatch")
    # One block at a time. Large final states are never retained across blocks.
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["column_id"], ids):
            raise ValueError("Column ID array mismatch")
        for name in z.files:
            a = z[name]
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite block array: "+name)
            del a
        rows = {name: z[name] for name in ROW_NAMES}
    if any(a.shape != (2, len(ids)) for a in rows.values()):
        raise ValueError("Output row shape mismatch")
    for actual, expected in ((rows["u0"], cache["U0_canonical"][:, ids]),
                             (rows["delta_u"], rows["driven_raw_u"]-rows["undriven_raw_u"]),
                             (rows["v"], rows["driven_raw_v"]-rows["undriven_raw_v"])):
        if not np.allclose(actual, expected, atol=r1.ATOL, rtol=r1.RTOL):
            raise ValueError("Canonical U0 or driven-minus-undriven mismatch")
    return meta, rows


def merge(run_root, require_all=True, block_paths=None, output_name="merged.npz"):
    root = Path(run_root).resolve()
    manifest, cache = _load(root)
    paths = list(block_paths) if block_paths is not None else sorted((root/"blocks").glob("*.json"))
    if not paths:
        raise ValueError("No blocks")
    ncol = int(manifest["physical_scope"]["n_columns"])
    seen = np.zeros(ncol, bool)
    rows = {name: np.zeros((2, ncol), complex) for name in ROW_NAMES}
    for path in paths:
        meta, block = validate_block(root, manifest, cache, path)
        ids = np.array(meta["column_ids"], int)
        if np.any(seen[ids]):
            raise ValueError("Duplicate block/column")
        for name in rows:
            rows[name][:, ids] = block[name]
        seen[ids] = True
    if require_all and not np.all(seen):
        raise ValueError("Missing input columns")
    ids = np.flatnonzero(seen)
    rows = {k: v[:, ids] for k, v in rows.items()}
    U = rows["u0"]+rows["delta_u"]
    V = rows["v"]
    n, m = V.conj() @ V.T, U @ V.T
    normal = U @ U.conj().T - V @ V.conj().T - np.eye(2)
    anomalous = U @ V.T - V @ U.T
    cov = np.block([[np.eye(2)/2+n.real+m.real, m.imag+n.imag],
                    [m.imag-n.imag, np.eye(2)/2+n.real-m.real]])
    symplectic = np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
    arrays = dict(rows, column_id=ids, U=U, V=V, n=n, m=m,
                  m_U0VT=rows["u0"]@V.T, m_source_source=rows["delta_u"]@V.T,
                  normal_ccr_residual=normal, anomalous_ccr_residual=anomalous,
                  quadrature_covariance=cov)
    r1._finite_arrays(arrays)
    output = root/output_name
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    r1._write_npz_atomic(output, arrays)
    meta = dict(version=VERSION, manifest_sha256=manifest["manifest_sha256"],
                status="finite_grid_candidate_not_convergence_qualified",
                n_columns=int(len(ids)), required_columns=ncol, complete_columns=bool(np.all(seen)),
                physical_scope=manifest["physical_scope"], time=manifest["time"],
                n_trace=float(np.trace(n).real),
                n={"real":n.real.tolist(),"imag":n.imag.tolist()},
                m={"real":m.real.tolist(),"imag":m.imag.tolist()},
                normal_ccr_max=float(np.max(np.abs(normal))),
                normal_ccr_fro=float(np.linalg.norm(normal)),
                anomalous_ccr_max=float(np.max(np.abs(anomalous))),
                n_hermiticity=float(np.max(np.abs(n-n.conj().T))),
                n_eigenvalues=np.linalg.eigvalsh(n).tolist(),
                m_symmetry=float(np.max(np.abs(m-m.T))),
                uncertainty_eigenvalues=np.linalg.eigvalsh((cov+cov.T)/2+0.5j*symplectic).tolist(),
                output_sha256=r1._sha256_file(output),
                magnetic_signal_claimed=False,squeezing_claimed=False,entanglement_claimed=False,
                dynamic_time_frequency_bz_error_table_complete=False,
                blocks=[str(Path(p).relative_to(root)) for p in paths])
    r1._write_json_atomic(output.with_suffix(".json"),meta)
    print(json.dumps({"merged":str(output),"n_trace":meta["n_trace"],
                      "normal_ccr_max":meta["normal_ccr_max"]}),flush=True)
    return output


def run_all(root, block_size=1024, limit_blocks=None):
    root = Path(root).resolve()
    manifest, cache = _load(root)
    n = int(manifest["physical_scope"]["n_columns"])
    paths = []
    started = time.perf_counter()
    for index, start in enumerate(range(0,n,int(block_size))):
        if limit_blocks is not None and index >= limit_blocks:
            break
        stop = min(start+int(block_size),n)
        path=root/"blocks"/f"block_local_{start:08d}_{stop:08d}.json"
        if path.exists():
            meta,_ = validate_block(root,manifest,cache,path)
            if meta["column_ids"] != list(range(start,stop)):
                raise ValueError("Resume block range mismatch")
        else:
            path=worker(root,range(start,stop),"cuda","local")
        paths.append(path)
        r1._write_json_atomic(root/"progress.json",dict(status="running",
                             updated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             pid=os.getpid(), completed_columns=stop,total_columns=n,
                             completed_blocks=len(paths),wall_s=time.perf_counter()-started,
                             latest_block=str(path),physical_result_claimed=False))
        gc.collect()
    if len(paths) == (n+block_size-1)//block_size:
        result = merge(root,block_paths=paths)
        r1._write_json_atomic(root/"progress.json",dict(status="completed_finite_candidate",
                             updated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             completed_columns=n,total_columns=n,output=str(result),
                             wall_s=time.perf_counter()-started,physical_qualification_pending=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","worker","run-all","merge"))
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--block-size",type=int,default=1024)
    parser.add_argument("--steps",type=int,default=9216)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--stop",type=int,default=1024)
    parser.add_argument("--backend",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--block-id",default="local")
    parser.add_argument("--limit-blocks",type=int)
    args=parser.parse_args()
    if args.action=="prepare": prep.prepare(args.run_root,nstep=args.steps,block_size=args.block_size)
    elif args.action=="worker": worker(args.run_root,range(args.start,args.stop),args.backend,args.block_id)
    elif args.action=="merge": merge(args.run_root)
    else: run_all(args.run_root,args.block_size,args.limit_blocks)


if __name__=="__main__":
    try: main()
    except BaseException as exc:
        print(f"STOPPED {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
