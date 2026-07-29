"""Physical parameters and 1D spatial grid for FMT classical density functional theory."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalParameters:
    """Physical parameters for hard-sphere fluid system.

    Attributes:
        sigma: Sphere diameter (default: 1.0).
        eta: Bulk packing fraction (default: 0.4257).
        beta: Inverse thermal energy 1 / (k_B * T) (default: 1.0).
    """

    sigma: float = 1.0
    eta: float = 0.4257
    beta: float = 1.0

    @property
    def radius(self) -> float:
        """Sphere radius R = sigma / 2."""
        return 0.5 * self.sigma

    @property
    def volume(self) -> float:
        """Volume of a single sphere (4/3) * pi * R^3 = (pi/6) * sigma^3."""
        return (4.0 / 3.0) * math.pi * (self.radius**3)

    @property
    def rho_bulk(self) -> float:
        """Bulk number density rho_bulk = eta / volume = 6 * eta / (pi * sigma^3)."""
        return self.eta / self.volume

    @classmethod
    def eta_from_rho_bulk(cls, rho_bulk: float, sigma: float = 1.0) -> float:
        """Compute packing fraction eta from bulk number density rho_bulk."""
        vol = (math.pi / 6.0) * (sigma**3)
        return rho_bulk * vol

    @classmethod
    def rho_bulk_from_eta(cls, eta: float, sigma: float = 1.0) -> float:
        """Compute bulk number density rho_bulk from packing fraction eta."""
        vol = (math.pi / 6.0) * (sigma**3)
        return eta / vol


class Grid1D:
    """1D spatial grid and geometry utilities for planar FMT calculations.

    Args:
        params: PhysicalParameters instance.
        Lz: Domain length along z axis (default: 10.0 * sigma).
        dz: Spatial grid spacing (default: 0.005 * sigma, ensuring dz <= 0.01 * sigma).
    """

    def __init__(
        self,
        params: PhysicalParameters | None = None,
        Lz: float | None = None,
        dz: float | None = None,
    ) -> None:
        self.params = params if params is not None else PhysicalParameters()

        sigma = self.params.sigma
        self.Lz = Lz if Lz is not None else 10.0 * sigma
        self.dz = dz if dz is not None else 0.005 * sigma

        if self.dz > 0.01 * sigma + 1e-12:
            raise ValueError(
                f"Grid spacing dz={self.dz} exceeds maximum allowable spacing 0.01*sigma ({0.01 * sigma})."
            )
        if self.Lz <= 0:
            raise ValueError(f"Domain length Lz={self.Lz} must be positive.")

        # Construct 1D spatial coordinate array z in [0, Lz]
        self.num_points = round(self.Lz / self.dz) + 1
        self.z = np.linspace(0.0, self.Lz, self.num_points)
        # Update dz to exact grid step size
        self.dz = float(self.z[1] - self.z[0])

    def external_potential(
        self,
        wall_left: float | None = 0.0,
        wall_right: float | None = None,
    ) -> np.ndarray:
        """Compute external hard wall potential V_ext(z).

        For a sphere of radius R:
        - Forbidden region: center z < wall_left + R or z > wall_right - R (V_ext = np.inf).
        - Accessible region: V_ext = 0.0.

        Args:
            wall_left: Position of left hard wall (default: 0.0). If None, no left wall.
            wall_right: Position of right hard wall (default: None). If None, no right wall.

        Returns:
            1D array of external potential values (0.0 or np.inf).
        """
        R = self.params.radius
        v_ext = np.zeros_like(self.z, dtype=float)

        if wall_left is not None:
            forbidden_left = self.z < (wall_left + R - 1e-12)
            v_ext[forbidden_left] = np.inf

        if wall_right is not None:
            forbidden_right = self.z > (wall_right - R + 1e-12)
            v_ext[forbidden_right] = np.inf

        return v_ext

    def is_accessible(
        self,
        wall_left: float | None = 0.0,
        wall_right: float | None = None,
    ) -> np.ndarray:
        """Boolean mask indicating accessible regions for particle centers (V_ext == 0)."""
        v_ext = self.external_potential(wall_left, wall_right)
        return np.isfinite(v_ext)

    def initial_density_profile(
        self,
        rho_bulk: float | None = None,
        wall_left: float | None = 0.0,
        wall_right: float | None = None,
    ) -> np.ndarray:
        """Construct initial guess for density profile rho(z) = rho_bulk * exp(-beta * V_ext(z)).

        Density is zero in forbidden wall overlap regions and rho_bulk in accessible regions.
        """
        rho_b = rho_bulk if rho_bulk is not None else self.params.rho_bulk
        v_ext = self.external_potential(wall_left, wall_right)

        rho = np.zeros_like(self.z, dtype=float)
        accessible = np.isfinite(v_ext)
        rho[accessible] = rho_b * np.exp(-self.params.beta * v_ext[accessible])
        return rho

    def integrate(self, f: np.ndarray) -> float:
        """Compute spatial integral integral_0^Lz f(z) dz using trapezoidal rule."""
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(f, self.z))
        else:
            return float(np.trapz(f, self.z))
