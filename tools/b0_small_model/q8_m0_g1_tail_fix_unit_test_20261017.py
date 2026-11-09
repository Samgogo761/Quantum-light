"""Fast unit test of the 2026-10-17 fixes to the two tail diagnostics.

Engineering only: checks the four repaired behaviours on synthetic arrays.
Does not validate physics and does not replace the real 8->20 cycle regression.

Checks
1. pack() -> JSON round-trip with complex matrices (no TypeError).
2. 4-cycle sliding window on 12 cycles of data yields 9 positions (not 1).
3. Two-sided frequency grid is symmetric and contains 0.
4. Full Gram with U0 retains the U0 dU^† + h.c. cross term (delta-only drops it).
"""
from __future__ import annotations

import json

import numpy as np

# Re-implement the exact expressions used by the repaired scripts.
pack = lambda m: {"real": np.real(np.asarray(m, complex)).tolist(),
                  "imag": np.imag(np.asarray(m, complex)).tolist(),
                  "maxabs": float(np.max(np.abs(m)))}

rng = np.random.default_rng(0)

# 1. JSON round-trip
m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
blob = json.dumps({"m": pack(m)}, ensure_ascii=False)
back = json.loads(blob)["m"]
rec = np.array(back["real"]) + 1j * np.array(back["imag"])
assert np.allclose(rec, m), "pack round-trip failed"

# 2. sliding window
n_di = 12 * 1024
di = rng.normal(size=(2, n_di)) + 1j * rng.normal(size=(2, n_di))
sliding = [float(np.max(np.abs(np.mean(di[:, j:j + 4 * 1024], axis=1))))
           for j in range(0, di.shape[1] - 4 * 1024 + 1, 1024)]
assert len(sliding) == 9, f"expected 9 sliding positions, got {len(sliding)}"

# 3. two-sided frequencies
h = 125.66370614359172 / 1024.0
dw = 2.0 * np.pi / (n_di * h)
n_side = int(np.ceil(0.05 / dw))
freqs = np.arange(-n_side, n_side + 1) * dw
assert freqs[0] < 0 < freqs[-1] and np.isclose(freqs[0], -freqs[-1], atol=1e-15)
assert np.any(freqs == 0.0)

# 4. U0 cross term retained
u0 = rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))
du = 1e-3 * (rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5)))
v = 1e-4 * (rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5)))
normal_full = (u0 + du) @ (u0 + du).conj().T - v @ v.conj().T
normal_delta = du @ du.conj().T - v @ v.conj().T
cross = u0 @ du.conj().T + du @ u0.conj().T
static = u0 @ u0.conj().T
assert np.max(np.abs(normal_full - (normal_delta + cross + static))) < 1e-12
assert np.max(np.abs(normal_full - normal_delta)) > 1e-3, "cross term unexpectedly absent"

print(json.dumps({
    "checks": {"pack_json_roundtrip": "pass", "sliding_positions": len(sliding),
               "two_sided_freqs": [float(freqs[0]), 0.0, float(freqs[-1])],
               "u0_cross_term_retained": "pass"},
    "note": "engineering unit test only; real 8->20 cycle regression runs separately",
}, ensure_ascii=False, indent=1))
