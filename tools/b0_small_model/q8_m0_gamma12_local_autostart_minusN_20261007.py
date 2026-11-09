"""Start the -N local GPU run after the +N run finishes successfully.

Same laser, detectors, phase reference, grid, and step count. Only the Neel
sign changes. This does not compare the two domains and does not claim a
magnetic, squeezing, or entanglement result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUS = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007"
MINUS = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_local_minusN_20261007"
LOG = REPO / "results/q_line/gamma12_local_minusN_autostart_20261007.log"
PY = Path(sys.executable)
PREPARE = REPO / "tools/b0_small_model/q8_m0_gamma12_local_prepare_20261007.py"
RUNNER = REPO / "tools/b0_small_model/q8_m0_gamma12_local_runner_20261007.py"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    import ctypes
    access = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(access, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok) and int(code.value) == 259


def read_progress() -> dict:
    path = PLUS / "progress.json"
    last_error = None
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Cannot read +N progress: {last_error}")


def run(args: list[str]) -> None:
    log("launch " + " ".join(args))
    completed = subprocess.run(args, cwd=REPO, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit {completed.returncode}: {args[1]}")


def main() -> None:
    log(f"waiting for +N completion; watcher pid {os.getpid()}")
    dead_polls = 0
    while True:
        progress = read_progress()
        status = progress.get("status")
        done = int(progress.get("completed_columns", 0))
        total = int(progress.get("total_columns", 0))
        pid = int(progress.get("pid", 0))
        log(f"+N status={status} columns={done}/{total} pid={pid}")
        if status == "completed_finite_candidate":
            merged = PLUS / "merged.npz"
            if not merged.is_file():
                raise RuntimeError("+N reports completion but merged.npz is missing")
            break
        if status == "running" and pid_alive(pid):
            dead_polls = 0
        else:
            dead_polls += 1
            if dead_polls >= 3:
                raise RuntimeError("+N is no longer running and did not reach completion; -N not started")
        time.sleep(30)
    if MINUS.exists() and any(MINUS.iterdir()):
        raise RuntimeError(f"-N directory already exists and is not empty: {MINUS}")
    run([str(PY), str(PREPARE), "--run-root", str(MINUS), "--steps", "9216",
         "--ntpc", "1024", "--n-panel", "96", "--neel", "-1", "--block-size", "1024"])
    manifest = json.loads((MINUS / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["physical_scope"]["neel"]) != -1:
        raise RuntimeError("Prepared manifest is not -N")
    if int(manifest["physical_scope"]["n_columns"]) != 78472:
        raise RuntimeError("Prepared -N column count is not the full finite grid")
    run([str(PY), str(RUNNER), "run-all", "--run-root", str(MINUS), "--block-size", "1024"])
    final = json.loads((MINUS / "progress.json").read_text(encoding="utf-8"))
    if final.get("status") != "completed_finite_candidate":
        raise RuntimeError(f"-N ended with status {final.get('status')}")
    log("-N finite-grid propagation finished; domain comparison not claimed")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log(f"STOPPED {type(exc).__name__}: {exc}")
        raise
