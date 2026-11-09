#!/usr/bin/env python3
"""Tests for 232_I-d input construction. Cannot fit 232 WFs in [-6,8]."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import build_232_id_inputs as B


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


STUB_WIN = """num_wann = 112
num_bands = 400
dis_win_min  = -6.0
dis_win_max  =  8.0
dis_froz_min = -1.5
dis_froz_max =  0.8
mp_grid = 24 24 1
begin projections
Cr1: d
Cr2: d
I:   p
end projections
"""


def test_patch_win() -> None:
    out = B.patch_win(STUB_WIN)
    require("num_wann = 232" in out, out)
    require("num_wann = 112" not in out, out)
    require("13.2" in out, out)
    require("dis_win_max  =  8.0" not in out, out)
    require("I:   d" in out, out)
    require("I:   p" in out, out)
    require("dis_froz_min = -1.5" in out, out)


def test_cannot_use_8eV() -> None:
    e = np.zeros((4, 400))
    e[:, :56] = -10.0
    e[:, 56:230] = np.linspace(-5.0, 7.9, 174)
    e[:, 230:] = 9.0
    c8 = B.window_count(e, -6.0, 8.0)
    require(c8["max"] < 232, c8)
    require(not c8["ok_strict_gt_232"], c8)
    e[:, 230:330] = np.linspace(8.1, 13.2, 100)
    c13 = B.window_count(e, -6.0, 13.2)
    require(c13["ok_strict_gt_232"], c13)


def test_radial_id_count(tmp: Path) -> None:
    r = np.linspace(0.01, 10.0, 8)
    R = np.exp(-r)
    shells = {
        "Cr_d": {"R": R, "mean_r_bohr": 1.0, "channel_indices": [1], "rejected_exploding_indices": []},
        "I_p": {"R": R, "mean_r_bohr": 2.0, "channel_indices": [2], "rejected_exploding_indices": []},
        "I_d": {"R": R, "mean_r_bohr": 5.0, "channel_indices": [7, 8], "rejected_exploding_indices": [9, 10]},
    }
    ids: list[int] = []
    for i, name in enumerate(B.SHELLS_232, start=1):
        ids.extend([i] * B.N_BY_SHELL[name])
    require(len(ids) == 232, len(ids))
    require(ids.count(1) == 40, ids.count(1))
    require(ids.count(2) == 72, ids.count(2))
    require(ids.count(3) == 120, ids.count(3))
    _ = tmp, shells, r


def main() -> int:
    test_patch_win()
    test_cannot_use_8eV()
    test_radial_id_count(Path("."))
    print("test_build_232_id_inputs: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
