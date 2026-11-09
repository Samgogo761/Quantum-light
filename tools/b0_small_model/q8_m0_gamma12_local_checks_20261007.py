"""Independent bounded checks for the low-memory local runner.

The frozen r1 caller and its literal port formula are the reference, rather
than two executions of the new wrapper.  This checks storage/scheduling only;
it does not repeat static physics qualifications or launch a long trajectory.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

import q8_m0_gamma12_runner_20261006 as r1


def same(actual, expected):
    err = r1._array_error(np.asarray(actual), np.asarray(expected))
    if not err["allclose"]:
        raise AssertionError(err)
    return err


def independent_u0(cache, manifest):
    """Stationary right-port row, without calling either U0 assembler."""
    q = r1._backend_modules().q
    from q8_m0_classical import p1_a2_protocol
    prot = p1_a2_protocol(float(manifest["physical_scope"].get("a_star", .8)))
    omega = q.omega_grid(prot.omega0, prot.t_pulse)
    omega = omega[omega > 0]
    _, norm = q.normalize_phi(omega, q.phi_unnormalized(omega, 2 * prot.omega0, prot.t_pulse))
    eta = q.eta_of(float(cache["gamma"]))
    row = np.zeros_like(cache["A"])
    for j, label in enumerate(manifest["columns"]["records"]):
        if label["kind"] == "dark":
            continue
        nu = float(label["frequency"])
        a_out = cache["A"][:, j].copy()
        if label["port"] == "R":
            a_out -= cache["free_A0"][:, j]
        row[:, j] = q.phi_at(nu, prot, norm) * np.sqrt(nu) / eta * (cache["pol"] @ a_out)
    return row


def reference(fixture_path, root, window):
    raw = r1._load_npz(fixture_path)
    cache = r1._cache_arrays_from_fixture(raw)
    manifest = r1._fixture_manifest(root, root / "unused.npz", "", cache, fixture_path)
    cache["U0_canonical"] = independent_u0(cache, manifest)
    ids = np.arange(cache["b"].shape[1], dtype=np.int64)
    data = r1._block_inputs(manifest, cache, window, ids)
    mods = r1._backend_modules()
    d = r1._run_one(data, "driven", "cpu", mods)
    z = r1._run_one(data, "undriven", "cpu", mods)
    rows = {
        "column_id": ids,
        "u0": cache["U0_canonical"],
        "driven_raw_u": d["delta_u"], "undriven_raw_u": z["delta_u"],
        "driven_raw_v": d["v"], "undriven_raw_v": z["v"],
        "delta_u": d["delta_u"] - z["delta_u"], "v": d["v"] - z["v"],
        "driven_final_b": d["state"].b, "driven_final_bdag": d["state"].bdag,
        "driven_final_A": d["state"].A, "driven_final_I": d["final_I"],
        "undriven_final_b": z["state"].b, "undriven_final_bdag": z["state"].bdag,
        "undriven_final_A": z["state"].A, "undriven_final_I": z["final_I"],
    }
    U = rows["u0"] + rows["delta_u"]
    V = rows["v"]
    rows.update(U=U, V=V, n=V.conj() @ V.T, m=U @ V.T,
                m_U0VT=rows["u0"] @ V.T,
                m_source_source=rows["delta_u"] @ V.T,
                ccr_normal=U @ U.conj().T - V @ V.conj().T,
                ccr_anomalous=U @ V.T - V @ U.T)
    return cache, manifest, rows


def read_block(meta_path):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, r1._load_npz(meta_path.parent / meta["artifact"]["array_path"])


def compare_block(meta_path, ref):
    _, arr = read_block(meta_path)
    ids = np.asarray(arr["column_id"], int)
    keys = ("u0", "driven_raw_u", "undriven_raw_u", "driven_raw_v", "undriven_raw_v",
            "delta_u", "v", "driven_final_b", "driven_final_bdag", "driven_final_A",
            "driven_final_I", "undriven_final_b", "undriven_final_bdag", "undriven_final_A",
            "undriven_final_I")
    return {key: same(arr[key], ref[key][:, ids]) for key in keys}


def rejection_checks(local, run_root, block_paths):
    out = {}
    bad_dir = run_root / "rejected_cases"
    bad_dir.mkdir()

    def reject(name, paths, require_all=True):
        try:
            local.merge(run_root, require_all=require_all, block_paths=paths,
                        output_name="SHOULD_NOT_EXIST_" + name + ".npz")
        except (ValueError, RuntimeError, KeyError) as exc:
            out[name] = {"rejected": True, "error": type(exc).__name__, "message": str(exc)}
        else:
            raise AssertionError("Invalid input accepted: " + name)

    def altered(name, changes, arrays=None):
        original_path = Path(block_paths[0])
        meta, payload = read_block(original_path)
        meta = copy.deepcopy(meta)
        meta.update(changes)
        array_path = original_path.parent / meta["artifact"]["array_path"]
        if arrays is not None:
            array_path = bad_dir / (name + ".npz")
            r1._write_npz_atomic(array_path, arrays(payload))
        meta["artifact"] = {"array_path": str(array_path.resolve()),
                            "sha256": r1._sha256_file(array_path)}
        target = bad_dir / (name + ".json")
        r1._write_json_atomic(target, meta)
        return target

    reject("missing_block", block_paths[:-1])
    reject("duplicate_block", block_paths + [block_paths[0]])
    bad = altered("foreign_manifest", {"manifest_sha256": "foreign"})
    reject("foreign_manifest", [bad] + block_paths[1:])
    bad = altered("bad_hash", {})
    m = json.loads(bad.read_text(encoding="utf-8"))
    m["artifact"]["sha256"] = "0" * 64
    r1._write_json_atomic(bad, m)
    reject("artifact_hash", [bad] + block_paths[1:])
    original, _ = read_block(block_paths[0])
    bad = altered("wrong_time", {"nstep": original["nstep"] + 1})
    reject("time_steps", [bad] + block_paths[1:])
    bad = altered("foreign_code", {"execution_code_sha256": {"foreign": "0" * 64}})
    reject("block_execution_code", [bad] + block_paths[1:])
    def bad_u0(arr):
        arr["u0"] = arr["undriven_raw_u"].copy()
        return arr
    bad = altered("false_U0", {}, bad_u0)
    reject("false_stationary_U0", [bad] + block_paths[1:])
    return out


def vertex_order_check(out: Path):
    """The fixture path copies stored vertices. This checks the generator itself."""
    import q8_m0_gamma12_local_prepare_20261007 as prepare
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_model import build_geometry
    from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample

    nstep = 6
    geo = build_geometry()
    sample, _ = grid_sample(geo, 12, 0.0, neel=1)
    prot = p1_a2_protocol(0.8)
    h = float(prot.period / 1024.0)
    frozen = r1._backend_modules().backend.vertices_window(geo, sample, prot, nstep, h, 0)
    root = prepare._new_root(out / "vertex_order")
    prepare._vertex_files(root, geo, sample, prot, "driven", nstep, h, flush_steps=2)
    got_u = np.load(root / "arrays" / "driven_u.npy")
    got_d = np.load(root / "arrays" / "driven_d.npy")
    exp_u = np.asarray([item[0] for item in frozen], complex)
    exp_d = np.asarray([item[1] for item in frozen], complex)
    return {"u": same(got_u, exp_u), "d": same(got_d, exp_d), "nstep": nstep,
            "flush_steps": 2, "matches_frozen_vertices_window": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=r1.REPO / "results/q_line/p1_h2_finite_kcv_gamma12_backend_cuda_resume_20261006/backend_fixture.npz")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    import q8_m0_gamma12_local_runner_20261007 as local
    import q8_m0_gamma12_local_prepare_20261007 as prepare
    started = time.perf_counter()
    r1_hash = r1._sha256_file(Path(r1.__file__))
    report = {"scope": "16-step storage and backend regression; not a new physical qualification",
              "r1_sha256": r1_hash, "fixtures": {}}
    report["vertex_order"] = vertex_order_check(out)
    for window in ("start", "pump"):
        run_root = out / window
        prepare.prepare_fixture(run_root, args.fixture.resolve(), window=window)
        cache, manifest, ref = reference(args.fixture.resolve(), out, window)
        count = cache["b"].shape[1]
        rec = {"n_columns": count, "n_dark": manifest["physical_scope"]["n_dark_columns"],
               "column_coverage": "all fixture photon and dark columns"}
        # Keep anti-annihilation amplitudes as independent coordinates.
        rec["bdag_minus_conjugate_b_max"] = float(np.max(np.abs(cache["bdag"] - cache["b"].conj())))
        if rec["bdag_minus_conjugate_b_max"] <= 1e-10:
            raise AssertionError("Fixture cannot detect bdag/conjugate(b) confusion")
        cpu = local.worker(run_root, np.arange(count), "cpu", "whole_cpu")
        rec["cpu_against_r1"] = compare_block(cpu, ref)
        split_backend = "cpu" if args.cpu_only else "cuda"
        if not args.cpu_only:
            gpu = local.worker(run_root, np.arange(count), "cuda", "whole_cuda")
            rec["cuda_against_r1"] = compare_block(gpu, ref)
        paths = []
        # Interleaving detects incorrect placement, including photon/dark seams.
        for j in range(3):
            paths.append(local.worker(run_root, np.arange(j, count, 3), split_backend, f"split_{j}"))
        merged_path = local.merge(run_root, require_all=True, block_paths=paths,
                                  output_name="merged_split.npz")
        merged = r1._load_npz(Path(merged_path))
        rec["streaming_merge_against_independent_full_reference"] = {
            key: same(merged[key], ref[key]) for key in
            ("U", "V", "u0", "delta_u", "n", "m", "m_U0VT", "m_source_source")}
        rec["normal_ccr_contraction"] = same(merged["U"] @ merged["U"].conj().T - merged["V"] @ merged["V"].conj().T, ref["ccr_normal"])
        rec["anomalous_ccr_contraction"] = same(merged["U"] @ merged["V"].T - merged["V"] @ merged["U"].T, ref["ccr_anomalous"])
        rec["U0_independent_formula"] = same(merged["u0"], independent_u0(cache, manifest))
        if window == "pump":
            rec["rejections"] = rejection_checks(local, run_root, paths)
        report["fixtures"][window] = rec
        r1._write_json_atomic(out / "checks_progress.json", report)
    if r1._sha256_file(Path(r1.__file__)) != r1_hash:
        raise AssertionError("Frozen r1 was modified")
    report.update(all_checks_pass=True, elapsed_s=time.perf_counter() - started,
                  cpu_cuda_checked=not args.cpu_only, frozen_r1_unchanged=True)
    r1._write_json_atomic(out / "local_checks.json", report)
    print(json.dumps({"all_checks_pass": True, "report": str(out / "local_checks.json"),
                      "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
