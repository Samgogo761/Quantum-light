#!/usr/bin/env python3
"""Backend routing gates: CPU wrapper == pt_afm_toy, GPU stub refuses, no forbidden knobs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q_backend import (  # noqa: E402
    GPU_ACCEPTANCE_PROTOCOL,
    GPU_STATUS,
    CPUReferenceBackend,
    GPUBackendStub,
    backend_exposes_forbidden_knob,
    get_backend,
)
import pt_afm_toy as toy  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


P = toy.ToyParams(nk1d=4, ncyc=1.0, nt_per_cycle=8, e0=0.05, omega=0.25)


def test_cpu_wrapper_is_identical_to_reference() -> None:
    be = get_backend("cpu_reference")
    a = be.run_mean_current(P)
    b = toy.run_toy(P)
    require(np.array_equal(a["mean_je"], b["mean_je"]), "wrapper changed mean_je")
    ta = be.run_two_time(P, blocks=False)
    tb = toy.two_time_cjj(P, blocks=False)
    require(np.array_equal(ta["c_greater_plus"], tb["c_greater_plus"]), "wrapper changed C")


def test_gpu_stub_refuses() -> None:
    be = get_backend("gpu_stub")
    try:
        be.run_two_time(P)
    except NotImplementedError as exc:
        require(GPU_STATUS in str(exc), str(exc))
    else:
        raise RuntimeError("GPU stub must not run")


def test_no_forbidden_knobs() -> None:
    for cls in (CPUReferenceBackend, GPUBackendStub):
        bad = backend_exposes_forbidden_knob(cls)
        require(not bad, f"forbidden knobs exposed: {bad}")


def test_unknown_backend() -> None:
    try:
        get_backend("lg_cov_gpu")
    except ValueError:
        return
    raise RuntimeError("unknown backend accepted")


def test_protocol_has_fp64_and_structural_zero_rule() -> None:
    require("FP64" in GPU_ACCEPTANCE_PROTOCOL["precision"], "FP64 missing")
    require("floor" in GPU_ACCEPTANCE_PROTOCOL["structural_zero_reporting"], "floor rule missing")


def main() -> None:
    test_cpu_wrapper_is_identical_to_reference()
    test_gpu_stub_refuses()
    test_no_forbidden_knobs()
    test_unknown_backend()
    test_protocol_has_fp64_and_structural_zero_rule()
    print("test_q_backend: PASS")


if __name__ == "__main__":
    main()
