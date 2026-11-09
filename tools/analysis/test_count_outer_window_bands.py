#!/usr/bin/env python3
"""Tests for 232 outer-window band counting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import count_outer_window_bands as C


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def fake_nscf(k_e: list[np.ndarray]) -> str:
    chunks = ["the Fermi energy is     0.0800 ev\n"]
    for i, e in enumerate(k_e):
        chunks.append(f"     k = {i}.0000 0.0000 0.0000 (  10 PWs)   bands (ev):\n\n")
        line = "  ".join(f"{x:10.4f}" for x in e)
        chunks.append(f"   {line}\n")
    return "".join(chunks)


def test_parse_and_strict_gt() -> None:
    # 6 bands/k. Window [-6, E]. Need N > 2 when patched below via monkey values.
    e0 = np.array([-7.0, -5.0, -1.0, 1.0, 3.0, 9.0])
    e1 = np.array([-7.5, -5.5, -0.5, 1.5, 4.0, 8.5])
    text = fake_nscf([e0, e1])
    blocks = C.parse_qe_nscf_kblocks(text)
    require(len(blocks) == 2, len(blocks))
    require(blocks[0]["e_eV"].size == 6, blocks[0]["e_eV"].size)
    require(C.n_in_window(e0, -6.0, 8.0) == 4, C.n_in_window(e0, -6.0, 8.0))
    require(C.min_n_at(blocks, -6.0, 3.0) == 3, C.min_n_at(blocks, -6.0, 3.0))
    emax = C.smallest_emax_for_min_n(blocks, -6.0, 4)
    require(emax == 4.0, emax)


def test_reprint_takes_last() -> None:
    e = np.array([-5.0, 0.0, 2.0])
    text = fake_nscf([e, e]) + fake_nscf([e + 10.0, e + 10.0])
    blocks = C.parse_qe_nscf_kblocks(text)
    picked = C.select_kblocks(blocks, 2)
    require(picked[0]["e_eV"][0] == 5.0, picked[0]["e_eV"][0])


def test_build_report_insufficient() -> None:
    e0 = np.linspace(-6.0, 8.0, 10)
    e1 = np.linspace(-6.0, 7.0, 10)
    blocks = [
        {"k_cart": [0.0, 0.0, 0.0], "e_eV": e0},
        {"k_cart": [1.0, 0.0, 0.0], "e_eV": e1},
    ]
    report = C.build_report(blocks)
    require(report["nbnd"] == 10, report["nbnd"])
    require(report["n_k"] == 2, report["n_k"])
    require(report["target_n_surplus"] == 299, report["target_n_surplus"])
    require(report["nbnd_sufficient_for_target"] is False, report)
    require(report["e_max_choice"] == "nbnd_insufficient", report["e_max_choice"])


def test_build_report_hits_target(tmp: Path) -> None:
    rng = np.linspace(-6.0, 20.0, 320)
    blocks = [{"k_cart": [0.0, 0.0, 0.0], "e_eV": rng.copy()} for _ in range(4)]
    report = C.build_report(blocks)
    require(report["nbnd_sufficient_for_target"] is True, report)
    require(report["min_n_at_chosen"] >= 299, report["min_n_at_chosen"])
    require(report["e_max_chosen_eV"] == report["e_max_target_surplus_eV"], report)
    nscf = tmp / "fake.out"
    nscf.write_text(fake_nscf([rng] * 4), encoding="utf-8")
    out = tmp / "count.json"
    import json

    sys.argv = ["count_outer_window_bands.py", "--nscf", str(nscf), "--out", str(out), "--n-k", "4"]
    code = C.main()
    require(code == 0, code)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    require(loaded["n_k"] == 4, loaded["n_k"])
    require(loaded["hot_k_n_at_8eV"]["gamma"] == loaded["n_at_production_8eV"]["per_k"][0], loaded)


def main() -> int:
    test_parse_and_strict_gt()
    test_reprint_takes_last()
    test_build_report_insufficient()
    tmp = Path(__file__).resolve().parents[2] / ".codex_tmp" / "ref232_count_test"
    tmp.mkdir(parents=True, exist_ok=True)
    test_build_report_hits_target(tmp)
    print("test_count_outer_window_bands: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
