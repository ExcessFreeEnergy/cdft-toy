"""Dimensional Crossover & Confinement Collapse Visualizer Engine (Roth 2010 Sec. 4.2 & 4.4).

Analyzes and visualizes free energy density divergence spikes in scalar FMT functionals (RF, WB, WBII)
versus bounded stability in Tarazona's tensorial functional (WB-Tensor) under extreme spatial confinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.functionals import FMTFunctional, functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver
from src.weighted_densities import WeightedDensityCalculator


@dataclass
class ConfinementSweepMetrics:
    """Container for confinement sweep results at a single pore width."""

    pore_width: float
    max_rho: float
    max_phi: float
    free_energy_ex: float
    converged: bool
    iterations: int


class CrossoverAnalyzer:
    """Engine for dimensional crossover analysis and extreme confinement stability testing."""

    @staticmethod
    def generate_zero_d_gaussian(grid: Grid1D, alpha_width: float = 0.05, center_z: float | None = None) -> np.ndarray:
        """Generate a 1-particle normalized Gaussian zero-D cavity density profile.

        Formula:
            rho(z) = (1 / (sqrt(2*pi)*alpha)) * exp(-(z - z_0)^2 / (2*alpha^2))

        Args:
            grid: Grid1D spatial domain discretization.
            alpha_width: Gaussian width parameter in units of sigma (default 0.05).
            center_z: Center position z_0 (default Lz / 2).

        Returns:
            Normalized 1D density profile array matching grid.z.
        """
        z0 = center_z if center_z is not None else 0.5 * grid.Lz
        alpha = max(1e-4, float(alpha_width))

        # Gaussian shape centered at z0
        gaussian = (1.0 / (math.sqrt(2.0 * math.pi) * alpha)) * np.exp(-((grid.z - z0) ** 2) / (2.0 * (alpha**2)))

        # Normalize integral to exactly 1.0 particle
        integral = grid.integrate(gaussian)
        if integral > 1e-12:
            gaussian /= integral

        return gaussian

    @classmethod
    def evaluate_zero_d_divergence(
        self,
        grid: Grid1D,
        alpha_widths: list[float],
        functional: FMTFunctional,
        calc: WeightedDensityCalculator | None = None,
    ) -> list[float]:
        """Evaluate peak local free energy density max Phi(z) for synthetic zero-D Gaussian distributions.

        Args:
            grid: Grid1D spatial domain discretization.
            alpha_widths: List of Gaussian width parameters alpha_width.
            functional: FMT functional instance.
            calc: Optional WeightedDensityCalculator instance.

        Returns:
            List of peak local free energy density values max Phi(z) for each alpha_width.
        """
        if calc is None:
            calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

        max_phi_list = []
        for alpha in alpha_widths:
            rho = self.generate_zero_d_gaussian(grid, alpha_width=alpha)
            wd = calc.compute(rho)
            phi = functional.evaluate_phi(wd)

            # Mask out any non-finite overflow values for clean metric tracking
            phi_finite = np.nan_to_num(phi, nan=1e10, posinf=1e10, neginf=-1e10)
            max_phi_list.append(float(np.max(phi_finite)))

        return max_phi_list

    @classmethod
    def evaluate_zero_d_divergence_all(
        self,
        grid: Grid1D,
        alpha_widths: list[float],
        functionals: list[str] | None = None,
        calc: WeightedDensityCalculator | None = None,
    ) -> dict[str, list[float]]:
        """Evaluate peak local free energy density max Phi(z) for synthetic zero-D Gaussians across all specified functionals.

        Args:
            grid: Grid1D spatial domain discretization.
            alpha_widths: List of Gaussian width parameters alpha_width.
            functionals: List of functional names (default ["RF", "WB", "WBII", "WB-Tensor"]).
            calc: Optional WeightedDensityCalculator instance.

        Returns:
            Dictionary mapping functional name to list of peak local free energy density values max Phi(z).
        """
        if functionals is None:
            functionals = ["RF", "WB", "WBII", "WB-Tensor"]

        if calc is None:
            calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

        res: dict[str, list[float]] = {}
        for f_name in functionals:
            func = functional_factory(f_name)
            res[f_name] = self.evaluate_zero_d_divergence(grid, alpha_widths, func, calc)

        return res

    @staticmethod
    def sweep_pore_confinement_direct(
        widths: list[float],
        eta_bulk: float = 0.35,
        functionals: list[str] | None = None,
        dz: float = 0.008,
    ) -> dict[str, list[float]]:
        """Evaluate peak free energy density max Phi(z) directly across slit-pore widths without iterative solver overhead.

        Args:
            widths: List of slit-pore widths Lz.
            eta_bulk: Bulk packing fraction.
            functionals: List of functional names.
            dz: Spatial resolution.

        Returns:
            Dictionary mapping functional name to list of max Phi(z) values.
        """
        if functionals is None:
            functionals = ["RF", "WB", "WBII", "WB-Tensor"]

        results: dict[str, list[float]] = {f_name: [] for f_name in functionals}
        params = PhysicalParameters(eta=eta_bulk)

        for w in widths:
            grid = Grid1D(params=params, Lz=w, dz=dz)
            calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)
            v_ext = grid.external_potential(wall_left=0.0, wall_right=w)
            rho = params.rho_bulk * np.exp(-np.clip(v_ext, -700.0, 700.0))
            wd = calc.compute(rho)

            for f_name in functionals:
                func = functional_factory(f_name)
                phi = func.evaluate_phi(wd)
                max_phi = float(np.max(np.nan_to_num(phi, nan=1e10, posinf=1e10)))
                results[f_name].append(max_phi)

        return results

    @staticmethod
    def sweep_pore_confinement(
        widths: list[float],
        eta_bulk: float = 0.35,
        functionals: list[str] | None = None,
        dz: float = 0.005,
    ) -> dict[str, list[ConfinementSweepMetrics]]:
        """Sweep slit-pore widths Lz in [0.2 sigma, 3.0 sigma] and compute confinement metrics for specified functionals.

        Args:
            widths: List of slit-pore widths Lz.
            eta_bulk: Bulk packing fraction (default 0.35).
            functionals: List of functional names (default ["RF", "WB", "WBII", "WB-Tensor"]).
            dz: Spatial resolution (default 0.005).

        Returns:
            Dictionary mapping functional name to list of ConfinementSweepMetrics across pore widths.
        """
        if functionals is None:
            functionals = ["RF", "WB", "WBII", "WB-Tensor"]

        results: dict[str, list[ConfinementSweepMetrics]] = {f_name: [] for f_name in functionals}
        params = PhysicalParameters(eta=eta_bulk)

        for w in widths:
            grid = Grid1D(params=params, Lz=w, dz=dz)

            for f_name in functionals:
                func = functional_factory(f_name)
                solver = RothPicardSolver(grid, functional=func, alpha_init=0.03, wall_left=0.0, wall_right=w)
                res = solver.solve(max_iter=150, tol=1e-5)

                calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)
                wd = calc.compute(res.rho)
                phi = func.evaluate_phi(wd)

                max_rho = float(np.max(res.rho))
                max_phi = float(np.max(np.nan_to_num(phi, nan=1e10, posinf=1e10)))
                f_ex = func.compute_total_free_energy(grid, wd)

                metrics = ConfinementSweepMetrics(
                    pore_width=w,
                    max_rho=max_rho,
                    max_phi=max_phi,
                    free_energy_ex=f_ex,
                    converged=res.converged,
                    iterations=res.iterations,
                )
                results[f_name].append(metrics)

        return results
