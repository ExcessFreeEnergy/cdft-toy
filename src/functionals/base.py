"""Abstract base class interface for Fundamental Measure Theory (FMT) free energy functionals."""

from abc import ABC, abstractmethod
import math
from typing import Dict

import numpy as np

from src.grid import Grid1D
from src.weighted_densities import WeightedDensities


class FMTFunctional(ABC):
    """Abstract base class for hard-sphere FMT excess free energy functionals."""

    @abstractmethod
    def evaluate_phi(self, wd: WeightedDensities) -> np.ndarray:
        """Evaluate local excess free energy density Phi(z) [length^-3].

        Args:
            wd: WeightedDensities container instance.

        Returns:
            1D array of local excess free energy density values matching spatial grid.
        """
        pass

    @abstractmethod
    def evaluate_derivatives(self, wd: WeightedDensities) -> Dict[str, np.ndarray]:
        """Evaluate analytical partial derivatives (dPhi / dn_alpha) for all weight components.

        Args:
            wd: WeightedDensities container instance.

        Returns:
            Dictionary mapping weight key ('n0','n1','n2','n3','v1','v2') to derivative array.
        """
        pass

    def compute_total_free_energy(self, grid: Grid1D, wd: WeightedDensities) -> float:
        """Compute spatial integral of total excess free energy F_ex = integral_0^Lz Phi(z) dz.

        Args:
            grid: Grid1D spatial domain instance.
            wd: WeightedDensities container instance.

        Returns:
            Total excess free energy (dimensionless, units of k_B * T).
        """
        phi = self.evaluate_phi(wd)
        return grid.integrate(phi)

    @abstractmethod
    def compute_bulk_pressure(self, eta: float, sigma: float = 1.0) -> float:
        """Compute analytical bulk fluid pressure beta * p for given packing fraction eta.

        Args:
            eta: Bulk packing fraction.
            sigma: Sphere diameter.

        Returns:
            Reduced bulk pressure beta * p [length^-3].
        """
        pass

    def bulk_excess_mu(self, eta: float, sigma: float = 1.0) -> float:
        """Compute exact analytical bulk excess chemical potential beta * mu_ex.

        Args:
            eta: Bulk packing fraction.
            sigma: Sphere diameter.

        Returns:
            Dimensionless excess chemical potential beta * mu_ex.
        """
        R = 0.5 * sigma
        n0_bulk = eta / ((math.pi / 6.0) * (sigma**3))
        n1_bulk = n0_bulk * R
        n2_bulk = n0_bulk * 4.0 * math.pi * (R**2)
        n3_bulk = eta

        wd_bulk = WeightedDensities(
            n0=np.array([n0_bulk]),
            n1=np.array([n1_bulk]),
            n2=np.array([n2_bulk]),
            n3=np.array([n3_bulk]),
            v1=np.array([0.0]),
            v2=np.array([0.0]),
        )
        der = self.evaluate_derivatives(wd_bulk)
        mu_ex = (
            der["n0"][0]
            + R * der["n1"][0]
            + 4.0 * math.pi * (R**2) * der["n2"][0]
            + (4.0 / 3.0) * math.pi * (R**3) * der["n3"][0]
        )
        return float(mu_ex)
