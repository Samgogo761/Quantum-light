#!/usr/bin/env python3
"""Build a runtime SCF namelist: I/O and restart overlay only.

The committed CrI3_scf_soc_a1h_clean.in stays identical to April beyond outdir.
This writer only changes disk_io / restart_mode / max_seconds so a killed SCF
can leave a restartable .save. It does not change the Hamiltonian.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def wall_left_seconds() -> int:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return 244800
    try:
        raw = subprocess.check_output(
            ["squeue", "-h", "-j", str(job), "-o", "%L"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 244800
    if raw in ("", "NOT_SET", "UNLIMITED"):
        return 244800
    days = 0
    if "-" in raw:
        day_s, raw = raw.split("-", 1)
        days = int(day_s)
    parts = [int(x) for x in raw.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = parts[0]
    else:
        return 244800
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def save_restartable(save_dir: Path) -> bool:
    if not save_dir.is_dir():
        return False
    if (save_dir / "charge-density.dat").is_file():
        return True
    if (save_dir / "charge-density.hdf5").is_file():
        return True
    return any(save_dir.glob("wfc*.dat")) or any(save_dir.glob("wfc*.hdf5"))


def upsert_control(text: str, keys: dict[str, str]) -> str:
    m = re.search(r"(?im)^&control\b", text)
    if not m:
        raise SystemExit("no &CONTROL namelist")
    start = m.start()
    m_end = re.search(r"(?m)^\s*/\s*$", text[start:])
    if not m_end:
        raise SystemExit("no closing slash for &CONTROL")
    end = start + m_end.start()
    block = text[start:end]
    for name, value in keys.items():
        line = f"    {name} = {value}\n"
        found = False
        out_lines = []
        for ln in block.splitlines(keepends=True):
            if ln.lstrip().lower().startswith(name.lower()) and "=" in ln:
                out_lines.append(line if not ln.endswith("\n") else line)
                found = True
            else:
                out_lines.append(ln)
        if not found:
            if out_lines and not out_lines[-1].endswith("\n"):
                out_lines[-1] += "\n"
            out_lines.append(line)
        block = "".join(out_lines)
        if not block.endswith("\n"):
            block += "\n"
    return text[:start] + block + text[end:]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("clean_in")
    p.add_argument("save_dir")
    p.add_argument("run_in")
    p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--nscf-reserve", type=int, default=9000)
    args = p.parse_args()
    if args.max_seconds <= 0:
        args.max_seconds = max(1800, wall_left_seconds() - args.nscf_reserve)
    text = Path(args.clean_in).read_text(encoding="utf-8", errors="replace")
    restart = save_restartable(Path(args.save_dir))
    text = upsert_control(
        text,
        {
            "disk_io": "'high'",
            "restart_mode": "'restart'" if restart else "'from_scratch'",
            "max_seconds": str(int(args.max_seconds)),
        },
    )
    out = Path(args.run_in)
    out.write_text(text.replace("\r\n", "\n"), encoding="utf-8")
    print(
        f"A1H_SCF_RUN restart={'restart' if restart else 'from_scratch'} "
        f"disk_io=high max_seconds={args.max_seconds} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
