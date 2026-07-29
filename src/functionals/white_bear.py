"""White-Bear (WB) and White-Bear II (WBII) Fundamental Measure Theory functionals.

Implements the White-Bear (WB) and White-Bear II (WBII) hard-sphere excess free energy
density functionals and their exact analytical partial derivatives, low-density Taylor
expansions (for n3 < 1e-4), bulk pressures, and chemical potentials as specified in
Roth (2010) Section 4.3 and Section 8.1.
"""

from __future__ import annotations

import math

import numpy as np

from src.functionals.base import FMTFunctional
from src.weighted_densities import WeightedDensities


def _f4_wb_series(n3: np.ndarray) -> np.ndarray:
    """Low-density Taylor series for WB f4_WB(n3) term for n3 < 1e-4.

    Expansion:
        f4_WB(n3) = (1 / 24*pi) * (1 + (16/9)*n3 + (5/2)*n3^2 + (16/5)*n3^3 + (35/9)*n3^4)
    """
    # Polynomial coefficients from lowest degree to highest:
    # 1.0, 16/9, 5/2, 16/5, 35/9
    poly = 1.0 + (16.0 / 9.0) * n3 + 2.5 * n3**2 + 3.2 * n3**3 + (35.0 / 9.0) * n3**4
    return poly / (24.0 * math.pi)


def _df4_wb_series(n3: np.ndarray) -> np.ndarray:
    """Derivative of low-density Taylor series d(f4_WB)/dn3 for n3 < 1e-4."""
    poly_der = (16.0 / 9.0) + 5.0 * n3 + 9.6 * n3**2 + (140.0 / 9.0) * n3**3
    return poly_der / (24.0 * math.pi)


def _phi2_series(n3: np.ndarray) -> np.ndarray:
    """Low-density series for WBII auxiliary function phi2(n3).

    phi2(n3) = (1/3)*n3^2 + (1/6)*n3^3 + (1/10)*n3^4
    """
    return (1.0 / 3.0) * n3**2 + (1.0 / 6.0) * n3**3 + 0.1 * n3**4


def _dphi2_series(n3: np.ndarray) -> np.ndarray:
    """Derivative of low-density series d(phi2)/dn3."""
    return (2.0 / 3.0) * n3 + 0.5 * n3**2 + 0.4 * n3**3


def _phi3_series(n3: np.ndarray) -> np.ndarray:
    """Low-density series for WBII auxiliary function phi3(n3).

    phi3(n3) = (4/3)*n3 - (1/6)*n3^2 - (1/15)*n3^3 - (1/30)*n3^4
    """
    return (4.0 / 3.0) * n3 - (1.0 / 6.0) * n3**2 - (1.0 / 15.0) * n3**3 - (1.0 / 30.0) * n3**4


def _dphi3_series(n3: np.ndarray) -> np.ndarray:
    """Derivative of low-density series d(phi3)/dn3."""
    return (4.0 / 3.0) - (1.0 / 3.0) * n3 - 0.2 * n3**2 - (2.0 / 15.0) * n3**3


class WhiteBearFunctional(FMTFunctional):
    """White-Bear (WB) hard-sphere FMT excess free energy functional (Roth Sec. 4.3 Eq. 23).

    Phi_WB = -n0*ln(1-n3) + (n1*n2 - v1*v2)/(1-n3) + (n2^3 - 3*n2*v2^2) * f4_WB(n3)
    where f4_WB(n3) = (n3 + (1-n3)^2 * ln(1-n3)) / (36*pi * n3^2 * (1-n3)^2).
    """

    def __init__(self, low_density_cutoff: float = 1e-3) -> None:
        self.low_density_cutoff = low_density_cutoff

    def f4_wb(self, n3: np.ndarray) -> np.ndarray:
        """Evaluate scalar weight correction function f4_WB(n3)."""
        mask = np.abs(n3) < self.low_density_cutoff
        res = np.empty_like(n3, dtype=float)

        n3_safe = np.where(mask, self.low_density_cutoff, n3)
        one_m_n3 = 1.0 - n3_safe
        num = n3_safe + (one_m_n3**2) * np.log(one_m_n3)
        den = 36.0 * math.pi * (n3_safe**2) * (one_m_n3**2)
        analytic = num / den

        res[~mask] = analytic[~mask]
        res[mask] = _f4_wb_series(n3[mask])
        return res

    def df4_wb(self, n3: np.ndarray) -> np.ndarray:
        """Evaluate derivative df4_WB / dn3."""
        mask = np.abs(n3) < self.low_density_cutoff
        res = np.empty_like(n3, dtype=float)

        n3_safe = np.where(mask, self.low_density_cutoff, n3)
        one_m_n3 = 1.0 - n3_safe
        log_term = np.log(one_m_n3)

        u = n3_safe + (one_m_n3**2) * log_term
        u_p = n3_safe - 2.0 * one_m_n3 * log_term
        v = 36.0 * math.pi * (n3_safe**2) * (one_m_n3**2)
        v_p = 72.0 * math.pi * n3_safe * one_m_n3 * (1.0 - 2.0 * n3_safe)

        analytic = (u_p * v - u * v_p) / (v**2)
        res[~mask] = analytic[~mask]
        res[mask] = _df4_wb_series(n3[mask])
        return res

    def evaluate_phi(self, wd: WeightedDensities) -> np.ndarray:
        """Evaluate local excess free energy density Phi_WB(z)."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        v1, v2 = wd.v1, wd.v2

        one_m_n3 = 1.0 - n3
        term1 = -n0 * np.log(one_m_n3)
        term2 = (n1 * n2 - v1 * v2) / one_m_n3
        term3 = (n2**3 - 3.0 * n2 * (v2**2)) * self.f4_wb(n3)

        return term1 + term2 + term3

    def evaluate_derivatives(self, wd: WeightedDensities) -> dict[str, np.ndarray]:
        """Evaluate exact analytical partial derivatives (dPhi / dn_alpha) for WB functional."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        v1, v2 = wd.v1, wd.v2

        one_m_n3 = 1.0 - n3
        f4_val = self.f4_wb(n3)
        df4_val = self.df4_wb(n3)

        d_n0 = -np.log(one_m_n3)
        d_n1 = n2 / one_m_n3
        d_v1 = -v2 / one_m_n3

        d_n2 = (n1 / one_m_n3) + 3.0 * (n2**2 - v2**2) * f4_val
        d_v2 = (-v1 / one_m_n3) - 6.0 * n2 * v2 * f4_val

        d_n3 = (n0 / one_m_n3) + (n1 * n2 - v1 * v2) / (one_m_n3**2) + (n2**3 - 3.0 * n2 * (v2**2)) * df4_val

        return {
            "n0": d_n0,
            "n1": d_n1,
            "n2": d_n2,
            "n3": d_n3,
            "v1": d_v1,
            "v2": d_v2,
        }

    def compute_bulk_pressure(self, eta: float, sigma: float = 1.0) -> float:
        """Compute analytical bulk fluid pressure beta * p from Carnahan-Starling EoS."""
        rho_bulk = eta / ((math.pi / 6.0) * (sigma**3))
        p_cs = rho_bulk * (1.0 + eta + eta**2 - eta**3) / ((1.0 - eta) ** 3)
        return float(p_cs)


class WhiteBearIIFunctional(FMTFunctional):
    """White-Bear II (WBII) hard-sphere FMT excess free energy functional (Roth Sec. 4.3 Eq. 352).

    Phi_WBII = -n0*ln(1-n3) + (n1*n2 - v1*v2) * f2_WBII(n3) + (n2^3 - 3*n2*v2^2) * f4_WBII(n3)
    where:
        f2_WBII(n3) = (1 + (1/3)*phi2(n3)) / (1 - n3)
        f4_WBII(n3) = (1 - (1/3)*phi3(n3)) / (24*pi * (1 - n3)^2).
    """

    def __init__(self, low_density_cutoff: float = 1e-3) -> None:
        self.low_density_cutoff = low_density_cutoff

    def phi2(self, n3: np.ndarray) -> np.ndarray:
        """Auxiliary function phi2(n3)."""
        mask = np.abs(n3) < self.low_density_cutoff
        res = np.empty_like(n3, dtype=float)

        n3_safe = np.where(mask, self.low_density_cutoff, n3)
        analytic = (2.0 * n3_safe - n3_safe**2 + 2.0 * (1.0 - n3_safe) * np.log(1.0 - n3_safe)) / n3_safe
        res[~mask] = analytic[~mask]
        res[mask] = _phi2_series(n3[mask])
        return res

    def dphi2(self, n3: np.ndarray) -> np.ndarray:
        """Derivative d(phi2)/dn3."""
        mask = np.abs(n3) < self.low_density_cutoff
        res = np.empty_like(n3, dtype=float)

        n3_safe = np.where(mask, self.low_density_cutoff, n3)
        analytic = (-(n3_safe**2) - 2.0 * n3_safe - 2.0 * np.log(1.0 - n3_safe)) / (n3_safe**2)
        res[~mask] = analytic[~mask]
        res[mask] = _dphi2_series(n3[mask])
        return res

    def phi3(self, n3: np.ndarray) -> np.ndarray:
        """Auxiliary function phi3(n3)."""
        mask = np.abs(n3) < self.low_density_cutoff
        res = np.empty_like(n3, dtype=float)

        n3_safe = np.where(mask, self.low_density_cutoff, n3)
        analytic = (
            2.0 * n3_safe - 3.0 * n3_safe**2 + 2.0 * n3_safe**3 + 2.0 * ((1.0 - n3_safe) ** 2) * np.log(1.0 - n3_safe)
        ) / (n3_safe**2)
        res[~mask] = analytic[~mask]
        res[mask] = _phi3_series(n3[mask])
        return res

    def dphi3(self, n3: np.ndarray) -> np.ndarray:
        """Derivative d(phi3)/dn3."""
        mask = np.abs(n3) < self.low_density_cutoff
        res = np.empty_like(n3, dtype=float)

        n3_safe = np.where(mask, self.low_density_cutoff, n3)
        analytic = (-4.0 * n3_safe + 2.0 * n3_safe**2 + 2.0 * n3_safe**3 - 4.0 * (1.0 - n3_safe) * np.log(1.0 - n3_safe)) / (
            n3_safe**3
        )
        res[~mask] = analytic[~mask]
        res[mask] = _dphi3_series(n3[mask])
        return res

    def f2_wb2(self, n3: np.ndarray) -> np.ndarray:
        """Pre-factor f2_WBII(n3)."""
        return (1.0 + (1.0 / 3.0) * self.phi2(n3)) / (1.0 - n3)

    def df2_wb2(self, n3: np.ndarray) -> np.ndarray:
        """Derivative df2_WBII / dn3."""
        one_m_n3 = 1.0 - n3
        num = (1.0 / 3.0) * self.dphi2(n3) * one_m_n3 + 1.0 + (1.0 / 3.0) * self.phi2(n3)
        return num / (one_m_n3**2)

    def f4_wb2(self, n3: np.ndarray) -> np.ndarray:
        """Pre-factor f4_WBII(n3)."""
        return (1.0 - (1.0 / 3.0) * self.phi3(n3)) / (24.0 * math.pi * ((1.0 - n3) ** 2))

    def df4_wb2(self, n3: np.ndarray) -> np.ndarray:
        """Derivative df4_WBII / dn3."""
        one_m_n3 = 1.0 - n3
        num = -(1.0 / 3.0) * self.dphi3(n3) * one_m_n3 + 2.0 * (1.0 - (1.0 / 3.0) * self.phi3(n3))
        return num / (24.0 * math.pi * (one_m_n3**3))

    def evaluate_phi(self, wd: WeightedDensities) -> np.ndarray:
        """Evaluate local excess free energy density Phi_WBII(z)."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        v1, v2 = wd.v1, wd.v2

        term1 = -n0 * np.log(1.0 - n3)
        term2 = (n1 * n2 - v1 * v2) * self.f2_wb2(n3)
        term3 = (n2**3 - 3.0 * n2 * (v2**2)) * self.f4_wb2(n3)

        return term1 + term2 + term3

    def evaluate_derivatives(self, wd: WeightedDensities) -> dict[str, np.ndarray]:
        """Evaluate exact analytical partial derivatives (dPhi / dn_alpha) for WBII functional."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        v1, v2 = wd.v1, wd.v2

        one_m_n3 = 1.0 - n3
        f2_val = self.f2_wb2(n3)
        df2_val = self.df2_wb2(n3)
        f4_val = self.f4_wb2(n3)
        df4_val = self.df4_wb2(n3)

        d_n0 = -np.log(one_m_n3)
        d_n1 = n2 * f2_val
        d_v1 = -v2 * f2_val

        d_n2 = n1 * f2_val + 3.0 * (n2**2 - v2**2) * f4_val
        d_v2 = -v1 * f2_val - 6.0 * n2 * v2 * f4_val

        d_n3 = (n0 / one_m_n3) + (n1 * n2 - v1 * v2) * df2_val + (n2**3 - 3.0 * n2 * (v2**2)) * df4_val

        return {
            "n0": d_n0,
            "n1": d_n1,
            "n2": d_n2,
            "n3": d_n3,
            "v1": d_v1,
            "v2": d_v2,
        }

    def compute_bulk_pressure(self, eta: float, sigma: float = 1.0) -> float:
        """Compute analytical bulk fluid pressure beta * p from Carnahan-Starling EoS."""
        rho_bulk = eta / ((math.pi / 6.0) * (sigma**3))
        p_cs = rho_bulk * (1.0 + eta + eta**2 - eta**3) / ((1.0 - eta) ** 3)
        return float(p_cs)
