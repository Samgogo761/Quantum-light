#!/usr/bin/env python3
"""Full-channel finite stationary ground covariance, without dense W_in.

This is an interface for the existing quadratic photon/p-h approximation,
not an exact many-electron ground state or a continuum existence proof.
All physical photon frequencies, vertices, and the ORIGINAL contact D must
be supplied. No cutoff, contact cancellation, or vacuum quench is selected.
Coordinates: all q, then all p; photons before particle-hole oscillators.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


def sym(a):
    return (a + a.T) / 2


def j_action(x):
    x = np.asarray(x)
    n = x.shape[0] // 2
    return np.concatenate((x[n:], -x[:n]), axis=0)


@dataclass(frozen=True)
class StationaryGroundOperator:
    frequencies: np.ndarray
    n_photon: int
    ca: np.ndarray
    ci: np.ndarray
    contact: np.ndarray
    provenance: str

    @classmethod
    def from_vertices(cls, photon_frequencies, gaps, photon_vertices,
                      weighted_electronic_vertices, contact, *, provenance):
        """Vertices have shape (2, n_osc).

        photon_vertices = eta * F, with eta and quadrature weights supplied
        by the approved photon normalization. electronic vertices = sqrt(w)*X.
        No aggregate spectral factor L can substitute for the raw X channels.
        """
        pf = np.array(photon_frequencies, dtype=float, copy=True)
        eg = np.array(gaps, dtype=float, copy=True)
        pv = np.array(photon_vertices, dtype=complex, copy=True)
        ev = np.array(weighted_electronic_vertices, dtype=complex, copy=True)
        dd0 = np.asarray(contact)
        if np.iscomplexobj(dd0) and np.any(dd0.imag != 0):
            raise ValueError("Stationary quadrature contact must be real.")
        dd = np.array(dd0, dtype=float, copy=True)
        if pf.ndim != 1 or eg.ndim != 1 or not pf.size or not eg.size:
            raise ValueError("Explicit nonempty photon frequencies and p-h gaps required.")
        if pv.shape != (2, pf.size) or ev.shape != (2, eg.size):
            raise ValueError("Supply raw vertices for two in-plane current components.")
        if dd.shape != (2, 2) or not np.array_equal(dd, dd.T):
            raise ValueError("Supply symmetric original D; this interface never repairs D.")
        if not provenance:
            raise ValueError("Explicit provenance required.")
        if not all(np.isfinite(a).all() for a in (pf, eg, pv, ev, dd)):
            raise ValueError("Nonfinite input.")
        if np.any(pf <= 0) or np.any(eg <= 0):
            raise ValueError("Zero/negative modes require separate mathematical treatment.")
        freq = np.r_[pf, eg]
        npht = pf.size
        n = freq.size
        ca = np.zeros((2, 2*n))
        ci = np.zeros_like(ca)
        ca[:, :npht] = np.sqrt(2) * pv.real
        ca[:, n:n+npht] = np.sqrt(2) * pv.imag
        ci[:, npht:n] = np.sqrt(2) * ev.real
        ci[:, n+npht:] = np.sqrt(2) * ev.imag
        for a in (freq, ca, ci, dd):
            a.setflags(write=False)
        return cls(freq, npht, ca, ci, dd, str(provenance))

    @property
    def dimension(self):
        return 2 * self.frequencies.size

    def _rhs(self, rhs):
        rhs = np.asarray(rhs)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.dimension:
            raise ValueError("Wrong coordinate/RHS shape.")
        return rhs

    def h_action(self, rhs):
        rhs = self._rhs(rhs)
        f = np.tile(self.frequencies, 2)
        free = f * rhs if rhs.ndim == 1 else f[:, None] * rhs
        ar = self.ca @ rhs
        return (free - self.ca.T @ (self.ci @ rhs)
                - self.ci.T @ ar + self.ca.T @ (self.contact @ ar))

    def generator_action(self, rhs):
        return j_action(self.h_action(rhs))

    def positivity_certificate(self):
        """Finite exact Schur-complement criterion, reduced to at most 2x2.

        Whitening the photon Schur complement gives I+F.T M F.
        Thin QR of F.T confines nontrivial eigenvalues to I+R M R.T.
        The ORIGINAL static residual M is kept, even when it is negative.
        This finite positivity test is NOT a continuum uniform lower bound.
        """
        n = self.frequencies.size
        p = np.r_[np.arange(self.n_photon), n + np.arange(self.n_photon)]
        e = np.r_[np.arange(self.n_photon, n),
                  n + np.arange(self.n_photon, n)]
        diagonal = np.tile(self.frequencies, 2)
        cie = self.ci[:, e]
        residual = self.contact - (cie / diagonal[e]) @ cie.T
        white_rows = self.ca[:, p] / np.sqrt(diagonal[p])
        _, rr = np.linalg.qr(white_rows.T, mode="reduced")
        small = np.eye(rr.shape[0]) + rr @ residual @ rr.T
        eigen = np.linalg.eigvalsh(sym(small))
        lower = float(min(1., eigen.min()))
        threshold = 100 * np.finfo(float).eps * max(1., np.linalg.norm(small, 2))
        return {
            "status": "finite_positive_certified" if lower > threshold else
                      "finite_positive_not_certified",
            "whitened_schur_spectral_lower_bound": lower,
            "roundoff_guard": float(threshold),
            "original_static_residual": residual.tolist(),
            "contact_replaced": False,
            "continuum_claim": False,
        }

    def free_resolvent_action(self, nu, rhs):
        """(h_free + nu J)^-1 rhs; nu is integration variable, not a cutoff."""
        rhs = self._rhs(rhs)
        nu = float(nu)
        if not np.isfinite(nu) or nu < 0:
            raise ValueError("nu must be finite and nonnegative.")
        n = self.frequencies.size
        freq = self.frequencies if rhs.ndim == 1 else self.frequencies[:, None]
        denominator = freq**2 + nu**2
        return np.concatenate(((freq * rhs[:n] - nu * rhs[n:]) / denominator,
                               (nu * rhs[:n] + freq * rhs[n:]) / denominator))

    def resolvent_action(self, nu, rhs):
        """Full Woodbury inverse, with free/dark part retained."""
        rhs = self._rhs(rhs)
        uu = np.column_stack((self.ca.T, self.ci.T))
        eye = np.eye(2)
        qinv = np.block([[np.zeros((2, 2)), -eye], [-eye, -self.contact]])
        br = self.free_resolvent_action(nu, rhs)
        bu = self.free_resolvent_action(nu, uu)
        correction = np.linalg.solve(qinv + uu.T @ bu, uu.T @ br)
        return br - bu @ correction

    def projected_ground_covariance(self, rows, *, order=96, scale=None):
        """F Gamma_0 F.T = (1/pi) int_0^infty sym[F(h+nu J)^-1 F.T] dnu.

        Returns only the requested covariance block, without forming W or Gamma.
        Projection rows must be real quadrature functionals with explicit units.
        Gauss quadrature covers the whole nu half-line by a rational map.
        """
        raw = np.asarray(rows)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("Use real quadrature rows; complex optical rows need conversion.")
        rows = np.asarray(raw.real, float)
        if rows.ndim != 2 or rows.shape[1] != self.dimension or not np.isfinite(rows).all():
            raise ValueError("Invalid observation rows.")
        certificate = self.positivity_certificate()
        if certificate["status"] != "finite_positive_certified":
            raise ValueError("Ground covariance not certified; no contact repair or eigenvalue clipping.")
        if not isinstance(order, (int, np.integer)) or order < 8:
            raise ValueError("Quadrature order must be an integer >=8.")
        if scale is None:
            scale = float(np.exp(np.mean(np.log(self.frequencies))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Positive numerical quadrature scale required.")
        nodes, weights = leggauss(order)
        xx, ww = (nodes + 1)/2, weights/2
        result = np.zeros((rows.shape[0], rows.shape[0]))
        for x, weight in zip(xx, ww):
            nu = scale * x / (1-x)
            jacobian = scale / (1-x)**2
            projected = rows @ self.resolvent_action(nu, rows.T)
            result += weight * jacobian * sym(projected) / np.pi
        return sym(result)
