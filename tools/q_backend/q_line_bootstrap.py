#!/usr/bin/env python3
"""Inventory or rerun the existing Q-line development regressions, without edits.

This stdlib-only runner does not import model code. A regression PASS is an
engineering observation, never approval of the physics or quantum-output claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "q_line"
TEST_PATHS = (
    "tools/b0_small_model/test_b0_gates.py",
    "tools/b0_small_model/test_b1_dev_closed_wick.py",
    "tools/b0_small_model/test_pt_afm_toy.py",
    "tools/b0_small_model/test_pt_afm_two_time.py",
    "tools/b0_small_model/test_q_frozen_spec.py",
    "tools/b0_small_model/test_q_harness_gates.py",
    "tools/b0_small_model/test_q8_tperp_static_audit.py",
    "tools/b0_small_model/test_q8_m0.py",
    "tools/q_backend/test_q_backend.py",
)
SOURCE_PATHS = (
    "tools/b0_small_model/__init__.py",
    "tools/b0_small_model/pt_afm_toy.py",
    "tools/b0_small_model/fock.py",
    "tools/b0_small_model/wick.py",
    "tools/b0_small_model/qrt.py",
    "tools/b0_small_model/b1_dev_closed_wick.py",
    "tools/b0_small_model/q8_m0_model.py",
    "tools/b0_small_model/q8_m0_static.py",
    "tools/b0_small_model/q8_m0_classical.py",
    "tools/b0_small_model/q8_m0_p0p1.py",
    "tools/b0_small_model/PT_AFM_TOY.md",
    "tools/b0_small_model/Q_THEORY_FROZEN_SPEC.md",
    "tools/b0_small_model/B1_DEV_INTERFACE.md",
    "tools/q_backend/__init__.py",
    "tools/q_backend/q_backend.py",
    "tools/q_backend/README.md",
    "tools/q_backend/q_line_bootstrap.py",
    "tools/q_backend/Q_LINE_STARTUP.md",
    "docs/04_BSV量子光/线Q_PT-AFM玩具闭系统Wick第一阶段_20260831.md",
    "docs/04_BSV量子光/线Q_框架冻结与CPU两时基线_20260903.md",
    "docs/04_BSV量子光/线Q_独立研究入口与最小路线_20260907.md",
    "docs/04_BSV量子光/线Q_首轮物理审核单_20260907.md",
    "docs/04_BSV量子光/线Q_4band_harness资格关闭_20260907.md",
) + TEST_PATHS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_inventory() -> list[dict]:
    records = []
    for relative_path in SOURCE_PATHS:
        record = {"path": relative_path, "status": "unreadable", "sha256": None}
        try:
            data = (REPO_ROOT / relative_path).read_bytes()
        except OSError as exc:
            record["error"] = str(exc)
        else:
            record.update(status="readable", size_bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest())
        records.append(record)
    return records


def dependency_inventory() -> dict:
    # Inspect package metadata without importing numpy or any model module.
    try:
        numpy_spec = importlib.util.find_spec("numpy")
    except (ImportError, ValueError) as exc:
        numpy_spec = None
        detection_error = str(exc)
    else:
        detection_error = None
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    return {
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "numpy": {"module_found": numpy_spec is not None, "version": numpy_version,
                  "version_source": "installed_distribution_metadata",
                  "import_verified": False, "detection_error": detection_error},
        "platform": platform.platform(),
    }


def make_output_dir(requested: str | None) -> Path:
    # Resolve both paths so '..', symlinks, and Windows junctions cannot escape.
    allowed = RESULTS_ROOT.resolve()
    if not allowed.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("results/q_line resolves outside the repository")
    if requested is None:
        candidate = allowed / "startup_20260907"
        if candidate.exists():
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            candidate = allowed / f"startup_20260907_{suffix}_{uuid.uuid4().hex[:8]}"
    else:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValueError("output must be a subdirectory of repository results/q_line")
    # Requiring a new directory is deliberately stricter than refusing nonempty
    # ones. No old report or log can be overwritten, including after a failed run.
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def run_test(record: dict, output_dir: Path, timeout: float) -> bool:
    name = Path(record["path"]).stem
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    command = [sys.executable, "-B", str(REPO_ROOT / record["path"])]
    record.update(command=command, cwd=str(REPO_ROOT), started_at=utc_now(),
                  stdout_log=stdout_path.name, stderr_log=stderr_path.name)
    start = time.perf_counter()
    interrupted = False
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout,
                                    stderr=stderr, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            record.update(status="timeout", error=f"Exceeded {timeout:g} seconds")
        except KeyboardInterrupt:
            record.update(status="interrupted", error="Interrupted by user")
            interrupted = True
        except OSError as exc:
            record.update(status="launch_error", error=str(exc))
        else:
            record.update(returncode=result.returncode,
                          status="passed" if result.returncode == 0 else "failed")
    record.update(duration_seconds=round(time.perf_counter() - start, 6),
                  finished_at=utc_now())
    return interrupted


def positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-existing-regressions", action="store_true",
                        help="run the existing development suites (including 4-band harness gates)")
    parser.add_argument("--output-dir", help="new directory inside results/q_line; "
                        "relative paths are resolved from the repository")
    parser.add_argument("--timeout-seconds", type=positive_timeout, default=300.0,
                        help="per-suite timeout, default: 300; no physics parameter")
    args = parser.parse_args()
    try:
        output_dir = make_output_dir(args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    start = time.perf_counter()
    sources = source_inventory()
    dependencies = dependency_inventory()
    ready = (all(item["status"] == "readable" for item in sources)
             and dependencies["numpy"]["module_found"])
    tests = [{"path": path, "status": "not_run", "duration_seconds": None,
              "returncode": None, "stdout_log": None, "stderr_log": None}
             for path in TEST_PATHS]
    report = {
        "schema_version": 1, "started_at": utc_now(), "repository": str(REPO_ROOT),
        "output_directory": str(output_dir),
        "mode": "existing_regressions" if args.run_existing_regressions else "inventory",
        "scope": "existing_development_regressions_only",
        "physics_review_status": "pending_user_review",
        "interpretation": "PASS only records existing engineering regressions. It does "
                          "not approve a Hamiltonian, gauge, SBE, dephasing, correlation "
                          "definition, or research design; it is not a new quantum study.",
        "quantum_output_claims": "not_evaluated",
        "gpu_status": "not_implemented_by_this_runner",
        "dependencies": dependencies, "sources_before": sources,
        "tests": tests, "per_suite_timeout_seconds": args.timeout_seconds,
    }
    interrupted = False
    print(f"Output: {output_dir}", flush=True)
    print("Physics review: pending_user_review; engineering checks only.", flush=True)
    if args.run_existing_regressions and ready:
        for record in tests:
            print(f"Running {record['path']} ...", flush=True)
            interrupted = run_test(record, output_dir, args.timeout_seconds)
            print(f"  {record['status']} ({record['duration_seconds']:.3f} s)", flush=True)
            if interrupted:
                break
    else:
        reason = "inventory mode; no tests executed" if not args.run_existing_regressions \
            else "inventory incomplete; no tests executed"
        for record in tests:
            record["reason"] = reason
        print(reason, flush=True)

    sources_after = source_inventory()
    changed = [before["path"] for before, after in zip(sources, sources_after)
               if before != after]
    if not ready:
        status = "inventory_incomplete"
    elif changed:
        status = "source_changed_during_run"
    elif not args.run_existing_regressions:
        status = "inventory_ready_tests_not_run"
    elif interrupted:
        status = "existing_regressions_interrupted"
    elif all(item["status"] == "passed" for item in tests):
        status = "existing_regressions_pass"
    else:
        status = "existing_regressions_failed"
    report.update(engineering_status=status, sources_after=sources_after,
                  changed_source_paths=changed, finished_at=utc_now(),
                  duration_seconds=round(time.perf_counter() - start, 6),
                  test_summary={key: sum(item["status"] == key for item in tests)
                                for key in ("passed", "failed", "not_run", "timeout",
                                            "launch_error", "interrupted")})
    report_path = output_dir / "startup_report.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Engineering status: {status}\nReport: {report_path}", flush=True)
    return 0 if status in {"inventory_ready_tests_not_run", "existing_regressions_pass"} \
        else 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
