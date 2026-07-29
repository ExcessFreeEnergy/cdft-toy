"""Original Rosenfeld (RF) FMT excess free energy density functional and analytical derivatives."""

import math

import numpy as np

from src.functionals.base import FMTFunctional
from src.weighted_densities import WeightedDensities


class RosenfeldFunctional(FMTFunctional):
    """Original Rosenfeld (1989) FMT hard-sphere excess free energy functional (Roth Sec. 4.1).

    The excess free energy density Phi_RF is given by:
        Phi_RF = -n0 * ln(1 - n3) + (n1*n2 - v1*v2)/(1 - n3) + (n2^3 - 3*n2*v2^2) / (24*pi*(1 - n3)^2)
    """

    FACTOR_24PI_INV = 1.0 / (24.0 * math.pi)
    FACTOR_12PI_INV = 1.0 / (12.0 * math.pi)
    FACTOR_8PI_INV = 1.0 / (8.0 * math.pi)
    FACTOR_4PI_INV = 1.0 / (4.0 * math.pi)

    def evaluate_phi(self, wd: WeightedDensities) -> np.ndarray:
        """Evaluate local excess free energy density Phi_RF(z)."""
        one_minus_n3 = np.maximum(1e-12, 1.0 - wd.n3)
        inv_n3 = 1.0 / one_minus_n3
        inv_n3_sq = inv_n3 * inv_n3

        phi1 = -wd.n0 * np.log(one_minus_n3)
        phi2 = (wd.n1 * wd.n2 - wd.v1 * wd.v2) * inv_n3
        phi3 = (wd.n2**3 - 3.0 * wd.n2 * (wd.v2**2)) * (
            self.FACTOR_24PI_INV * inv_n3_sq
        )

        return phi1 + phi2 + phi3

    def evaluate_derivatives(self, wd: WeightedDensities) -> dict[str, np.ndarray]:
        """Evaluate analytical partial derivatives (dPhi_RF / dn_alpha) for all 6 weight components."""
        one_minus_n3 = np.maximum(1e-12, 1.0 - wd.n3)
        inv_n3 = 1.0 / one_minus_n3
        inv_n3_sq = inv_n3 * inv_n3
        inv_n3_cube = inv_n3_sq * inv_n3

        # dPhi / dn0
        d_n0 = -np.log(one_minus_n3)

        # dPhi / dn1
        d_n1 = wd.n2 * inv_n3

        # dPhi / dv1
        d_v1 = -wd.v2 * inv_n3

        # dPhi / dn2
        d_n2 = (wd.n1 * inv_n3) + (wd.n2**2 - wd.v2**2) * (
            self.FACTOR_8PI_INV * inv_n3_sq
        )

        # dPhi / dv2
        d_v2 = -(wd.v1 * inv_n3) - (wd.n2 * wd.v2) * (self.FACTOR_4PI_INV * inv_n3_sq)

        # dPhi / dn3
        term1 = wd.n0 * inv_n3
        term2 = (wd.n1 * wd.n2 - wd.v1 * wd.v2) * inv_n3_sq
        term3 = (wd.n2**3 - 3.0 * wd.n2 * (wd.v2**2)) * (
            self.FACTOR_12PI_INV * inv_n3_cube
        )
        d_n3 = term1 + term2 + term3

        return {
            "n0": d_n0,
            "n1": d_n1,
            "n2": d_n2,
            "n3": d_n3,
            "v1": d_v1,
            "v2": d_v2,
        }

    def compute_bulk_pressure(self, eta: float, sigma: float = 1.0) -> float:
        """Compute Percus-Yevick (PY) compressibility bulk pressure beta * p_PY_c.

        Formula (Roth Eq. 15):
            beta * p_PY_c = rho_bulk / (1 - eta) + (n1*n2)/(1 - eta)^2 + n2^3 / (12*pi*(1 - eta)^3)
        """
        R = 0.5 * sigma
        v_sphere = (4.0 / 3.0) * math.pi * (R**3)
        rho_bulk = eta / v_sphere

        n0 = rho_bulk
        n1 = rho_bulk * R
        n2 = rho_bulk * 4.0 * math.pi * (R**2)
        n3 = eta

        one_minus_n3 = 1.0 - n3
        inv_n3 = 1.0 / one_minus_n3
        inv_n3_sq = inv_n3 * inv_n3
        inv_n3_cube = inv_n3_sq * inv_n3

        term1 = n0 * inv_n3
        term2 = (n1 * n2) * inv_n3_sq
        term3 = (n2**3) * (self.FACTOR_12PI_INV * inv_n3_cube)

        return float(term1 + term2 + term3)
