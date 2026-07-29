"""Tarazona Tensorial FMT excess free energy functional (Roth 2010 Sec. 4.4 Eq. 378).

Implements the White-Bear-Tensor (WB-Tensor) hard-sphere excess free energy density functional
with Tarazona's tensorial weight component w_m2 to eliminate divergence under tight confinement.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.functionals.base import FMTFunctional
from src.functionals.white_bear import WhiteBearFunctional
from src.weighted_densities import WeightedDensities


class WhiteBearTensorFunctional(FMTFunctional):
    """White-Bear Tensor (WB-Tensor) hard-sphere FMT functional (Roth Sec. 4.4 Eq. 378).

    Combines White-Bear equation of state prefactor f4_WB(n3) with Tarazona's tensorial
    weight component n_m2 to remove divergence in highly confined systems / solid phase.

    Phi_WB,T = -n0*ln(1-n3) + (n1*n2 - v1*v2)/(1-n3)
               + [n2^3 - 3*n2*v2^2 + 9*(v2^2*n_m2 - (3/8)*n_m2^3)] * f4_WB(n3)
    """

    def __init__(self, low_density_cutoff: float = 1e-3) -> None:
        self.wb = WhiteBearFunctional(low_density_cutoff=low_density_cutoff)

    def evaluate_phi(self, wd: WeightedDensities) -> np.ndarray:
        """Evaluate local excess free energy density Phi_WB,T(z)."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        v1, v2 = wd.v1, wd.v2
        n_m2 = wd.n_m2 if wd.n_m2 is not None else np.zeros_like(n3)

        one_m_n3 = 1.0 - n3
        term1 = -n0 * np.log(one_m_n3)
        term2 = (n1 * n2 - v1 * v2) / one_m_n3

        # Tarazona tensorial numerator term: n2^3 - 3*n2*v2^2 + 9*(v2^2*n_m2 - (3/8)*n_m2^3)
        tensor_num = (n2**3) - 3.0 * n2 * (v2**2) + 9.0 * ((v2**2) * n_m2 - 0.375 * (n_m2**3))
        term3 = tensor_num * self.wb.f4_wb(n3)

        return term1 + term2 + term3

    def evaluate_derivatives(self, wd: WeightedDensities) -> Dict[str, np.ndarray]:
        """Evaluate exact analytical partial derivatives (dPhi / dn_alpha) for WB-Tensor functional."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        v1, v2 = wd.v1, wd.v2
        n_m2 = wd.n_m2 if wd.n_m2 is not None else np.zeros_like(n3)

        one_m_n3 = 1.0 - n3
        f4_val = self.wb.f4_wb(n3)
        df4_val = self.wb.df4_wb(n3)

        d_n0 = -np.log(one_m_n3)
        d_n1 = n2 / one_m_n3
        d_v1 = -v2 / one_m_n3

        d_n2 = (n1 / one_m_n3) + 3.0 * (n2**2 - v2**2) * f4_val
        d_v2 = (-v1 / one_m_n3) + (-6.0 * n2 * v2 + 18.0 * v2 * n_m2) * f4_val
        d_n_m2 = 9.0 * (v2**2 - 1.125 * (n_m2**2)) * f4_val

        tensor_num = (n2**3) - 3.0 * n2 * (v2**2) + 9.0 * ((v2**2) * n_m2 - 0.375 * (n_m2**3))
        d_n3 = (n0 / one_m_n3) + (n1 * n2 - v1 * v2) / (one_m_n3**2) + tensor_num * df4_val

        return {
            "n0": d_n0,
            "n1": d_n1,
            "n2": d_n2,
            "n3": d_n3,
            "v1": d_v1,
            "v2": d_v2,
            "n_m2": d_n_m2,
        }

    def compute_bulk_pressure(self, eta: float, sigma: float = 1.0) -> float:
        """Compute analytical bulk fluid pressure beta * p from Carnahan-Starling EoS."""
        return self.wb.compute_bulk_pressure(eta, sigma=sigma)
