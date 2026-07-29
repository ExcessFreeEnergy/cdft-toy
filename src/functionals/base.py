"""Abstract base class interface for Fundamental Measure Theory (FMT) free energy functionals."""

from abc import ABC, abstractmethod
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
