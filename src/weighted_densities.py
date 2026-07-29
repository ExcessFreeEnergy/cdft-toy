"""Weighted density container and calculator for FMT classical density functional theory."""

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.convolutions import FFTConvolver1D
from src.grid import Grid1D


@dataclass
class WeightedDensities:
    """Container for 1D spatial weighted density arrays and physical metrics.

    Attributes:
        n0: Spatial density w_0 convolution array.
        n1: Spatial density w_1 convolution array.
        n2: Spatial density w_2 convolution array.
        n3: Local packing fraction w_3 convolution array.
        v1: Vector flux z-component v_1 convolution array.
        v2: Vector flux z-component v_2 convolution array.
    """

    n0: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    n3: np.ndarray
    v1: np.ndarray
    v2: np.ndarray

    @property
    def max_n3(self) -> float:
        """Maximum local packing fraction across spatial domain."""
        return float(np.max(self.n3))

    @property
    def min_n3(self) -> float:
        """Minimum local packing fraction across spatial domain."""
        return float(np.min(self.n3))

    @property
    def is_feasible(self) -> bool:
        """Boolean indicator whether local packing fraction satisfies n_3(z) < 1.0."""
        return self.max_n3 < (1.0 - 1e-12)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return dictionary mapping weight name to array."""
        return {
            "n0": self.n0,
            "n1": self.n1,
            "n2": self.n2,
            "n3": self.n3,
            "v1": self.v1,
            "v2": self.v2,
        }


class WeightedDensityCalculator:
    """High-level spatial weighted density calculator with physical feasibility checks.

    Args:
        grid: Grid1D spatial domain discretization instance.
        apply_endpoint_modification: Apply Section 8.4 endpoint weight modifications.
    """

    def __init__(self, grid: Grid1D, apply_endpoint_modification: bool = True) -> None:
        self.grid = grid
        self.convolver = FFTConvolver1D(grid, apply_endpoint_modification=apply_endpoint_modification)

    def compute(self, rho: np.ndarray) -> WeightedDensities:
        """Compute all 6 spatial weighted densities for a density profile rho(z).

        Args:
            rho: 1D spatial density profile array matching grid.z.

        Returns:
            WeightedDensities instance containing all 6 arrays and feasibility metrics.
        """
        raw_dict = self.convolver.compute_weighted_densities(rho)
        return WeightedDensities(
            n0=raw_dict["n0"],
            n1=raw_dict["n1"],
            n2=raw_dict["n2"],
            n3=raw_dict["n3"],
            v1=raw_dict["v1"],
            v2=raw_dict["v2"],
        )

    def validate_bulk_spt(self, eta: float) -> Dict[str, float]:
        """Compute Scaled Particle Theory (SPT) bulk reference values and relative error metrics.

        Args:
            eta: Bulk packing fraction.

        Returns:
            Dictionary containing exact SPT values and FFT convolution error metrics.
        """
        R = self.grid.params.radius
        rho_bulk = self.grid.params.rho_bulk_from_eta(eta, sigma=self.grid.params.sigma)

        # Uniform bulk density profile
        rho_uniform = np.full_like(self.grid.z, rho_bulk)
        wd = self.compute(rho_uniform)

        # Interior bulk mask (excluding wall cutoff boundary regions z in [2R, Lz - 2R])
        bulk_mask = (self.grid.z >= 2.0 * R) & (self.grid.z <= (self.grid.Lz - 2.0 * R))
        if not np.any(bulk_mask):
            bulk_mask = np.ones_like(self.grid.z, dtype=bool)

        spt_n3 = eta
        spt_n2 = rho_bulk * 4.0 * math.pi * (R**2)
        spt_n1 = rho_bulk * R
        spt_n0 = rho_bulk

        err_n3 = float(np.max(np.abs(wd.n3[bulk_mask] - spt_n3)))
        err_n2 = float(np.max(np.abs(wd.n2[bulk_mask] - spt_n2)))
        err_n1 = float(np.max(np.abs(wd.n1[bulk_mask] - spt_n1)))
        err_n0 = float(np.max(np.abs(wd.n0[bulk_mask] - spt_n0)))

        return {
            "spt_n3": spt_n3,
            "spt_n2": spt_n2,
            "spt_n1": spt_n1,
            "spt_n0": spt_n0,
            "max_err_n3": err_n3,
            "max_err_n2": err_n2,
            "max_err_n1": err_n1,
            "max_err_n0": err_n0,
        }
