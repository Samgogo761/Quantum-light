"""Small, independent checks for the stage-1W matrix exponential."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_stage1w_propagator_crosscheck import expm_nopy


class ExponentialTests(unittest.TestCase):
    def test_zero_and_diagonal(self):
        np.testing.assert_allclose(expm_nopy(np.zeros((2, 2))), np.eye(2))
        a = np.diag([0.1, -0.1])
        np.testing.assert_allclose(expm_nopy(a), np.diag(np.exp([0.1, -0.1])), atol=1e-14)

    def test_rotation_is_symplectic(self):
        j = np.array([[0.0, 1.0], [-1.0, 0.0]])
        theta = 0.7
        got = expm_nopy(theta * j)
        want = np.cos(theta) * np.eye(2) + np.sin(theta) * j
        np.testing.assert_allclose(got, want, atol=1e-14)
        np.testing.assert_allclose(got.T @ j @ got, j, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
