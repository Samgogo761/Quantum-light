#!/usr/bin/env python3
"""Physical-kernel regression for the two-port positive-frequency audit."""
from __future__ import annotations

import unittest

import numpy as np

from q8_m0_stage1s_continuum import (
    box_grid,
    compare,
    field_from_kernel,
    open_grid,
    photon_retarded_kernel,
)


class ContinuumKernelTest(unittest.TestCase):
    def test_periodic_box_missing_zero_mode_has_linear_retarded_tail(self):
        gamma = 0.0010807892763646022
        tp = 1005.3096491487338
        t_obs = 8 * tp
        dt = tp / 256
        lag = dt * np.arange(257)
        om, wt = box_grid(t_obs, 0.4)
        got_kernel = photon_retarded_kernel(lag, om, wt, gamma)
        exact_kernel = np.r_[0.0, gamma * (1 - 2 * lag[1:] / t_obs)]
        # A finite UV cutoff changes the short-lag kernel.  The smooth pulse
        # averages that boundary layer, allowing a clean periodic-tail check.
        current = np.sin(np.pi * lag / tp)[:, None] ** 2
        got = field_from_kernel(current, got_kernel, dt)
        exact = field_from_kernel(current, exact_kernel, dt)
        target = field_from_kernel(current,
                                   np.r_[0.0, np.full(256, gamma)], dt)
        self.assertLess(np.max(np.abs(got - exact)) /
                        np.max(np.abs(target)), 0.001)
        self.assertAlmostEqual(exact[-1, 0] / target[-1, 0], 0.875, places=12)

    def test_open_positive_quadrature_recovers_pulsed_maxwell_field(self):
        gamma = 0.0010807892763646022
        tp = 1005.3096491487338
        dt = tp / 256
        times = dt * np.arange(257)
        current = np.sin(np.pi * times / tp)[:, None] ** 2
        lag = times
        target = field_from_kernel(current,
                                   np.r_[0.0, np.full(256, gamma)], dt)
        om, wt = open_grid(0.4, 192)
        self.assertGreater(om.min(), 0.0)
        photon = field_from_kernel(
            current, photon_retarded_kernel(lag, om, wt, gamma), dt,
        )
        err = compare(photon, target, current, dt, gamma)
        self.assertLess(err["A_rel_max"], 0.001)
        self.assertLess(err["dA_rel_max"], 0.001)


if __name__ == "__main__":
    unittest.main()
