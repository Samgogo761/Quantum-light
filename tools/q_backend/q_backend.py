#!/usr/bin/env python3
"""Line-Q backend interface.  Physics lives in tools/b0_small_model; this layer only routes.

Rules (Q_THEORY_FROZEN_SPEC.md):
  * no knob here may change physics: no T2, gauge, integrator, dipole_mode, lg_cov, dtype;
  * CPUReferenceBackend calls pt_afm_toy directly -- it is the only judge;
  * GPUBackendStub raises until a CPU profile justifies implementing it;
  * every result must pass pt_afm_toy.assert_frozen_spec.
This directory is separate from the material-line CPU SBE code on purpose.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Mapping, Protocol

_B0 = Path(__file__).resolve().parents[1] / "b0_small_model"
if str(_B0) not in sys.path:
    sys.path.insert(0, str(_B0))

import pt_afm_toy as toy  # noqa: E402

GPU_STATUS = "NOT_IMPLEMENTED_UNTIL_CPU_PROFILE"


class QBackend(Protocol):
    name: str

    def run_mean_current(self, p: toy.ToyParams) -> Mapping[str, object]: ...

    def run_two_time(self, p: toy.ToyParams, *, blocks: bool = True) -> Mapping[str, object]: ...


class CPUReferenceBackend:
    """Thin wrapper.  Does not re-implement any equation."""

    name = "cpu_reference"

    def run_mean_current(self, p: toy.ToyParams) -> Mapping[str, object]:
        r = toy.run_toy(p)
        toy.assert_frozen_spec(r)
        return r

    def run_two_time(self, p: toy.ToyParams, *, blocks: bool = True) -> Mapping[str, object]:
        r = toy.two_time_cjj(p, blocks=blocks)
        toy.assert_frozen_spec(r)
        return r


class GPUBackendStub:
    name = "gpu_stub"
    status = GPU_STATUS

    def run_mean_current(self, p: toy.ToyParams) -> Mapping[str, object]:
        raise NotImplementedError(GPU_STATUS)

    def run_two_time(self, p: toy.ToyParams, *, blocks: bool = True) -> Mapping[str, object]:
        raise NotImplementedError(GPU_STATUS)


BACKENDS = {"cpu_reference": CPUReferenceBackend, "gpu_stub": GPUBackendStub}


def get_backend(name: str) -> QBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; allowed {sorted(BACKENDS)}")
    return BACKENDS[name]()


def backend_exposes_forbidden_knob(cls) -> list[str]:
    """Static check: no public method may take a physics-changing keyword."""
    found = []
    for _, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
        for param in inspect.signature(fn).parameters:
            if param in toy.FORBIDDEN_BACKEND_KNOBS:
                found.append(f"{cls.__name__}.{fn.__name__}({param})")
    return found


# Acceptance protocol for a future GPU backend (written, not executed here).
GPU_ACCEPTANCE_PROTOCOL = {
    "precision": "FP64 / complex128 only; no TF32/FP32 in v1",
    "single_step": "rho_{n+1}^GPU vs rho_{n+1}^CPU, max-abs on the same k grid",
    "full_time": "J_+(t), J_-(t), J_e(t), J_o(t) max-abs vs CPU",
    "equal_time": "C(t,t) vs CPU",
    "two_time": "sampled (t_i,t_j,k_l) and full small C(t,t') vs CPU",
    "structural_zero_reporting": (
        "for J_o and C_+ - C_-: report absolute error, absolute magnitude and the CPU "
        "numerical floor; a relative error of an ~0 quantity is not a metric"
    ),
    "spec": "result metadata must pass pt_afm_toy.assert_frozen_spec",
    "forbidden": "no new k points for occupancy, no BSV/Husimi samples, no 8-band, no T2",
}
