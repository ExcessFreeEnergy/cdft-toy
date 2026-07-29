"""Fixed-step Picard iteration solver for Classical Density Functional Theory."""

import math
from typing import Optional, Tuple, Union

import numpy as np

from src.functionals.base import FMTFunctional
from src.grid import Grid1D
from src.solvers.base import DFTSolver, SolverResult
from src.weighted_densities import WeightedDensityCalculator


class FixedPicardSolver(DFTSolver):
    """Fixed-step Picard solver for FMT hard-sphere density functional theory (Roth Sec. 8.1)."""

    def __init__(
        self,
        grid: Grid1D,
        functional: Optional[Union[str, FMTFunctional]] = None,
        functional_name: Optional[str] = None,
        alpha: float = 0.02,
        wall_left: Optional[float] = 0.0,
        wall_right: Optional[float] = None,
    ) -> None:
        from src.functionals import FMTFunctional, RosenfeldFunctional, functional_factory

        self.grid = grid
        if isinstance(functional, str):
            self.functional = functional_factory(functional)
        elif functional_name is not None:
            self.functional = functional_factory(functional_name)
        elif isinstance(functional, FMTFunctional):
            self.functional = functional
        else:
            self.functional = RosenfeldFunctional()

        self.alpha = float(alpha)
        self.wall_left = wall_left
        self.wall_right = wall_right

        self.calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)
        self.convolver = self.calc.convolver
        self.v_ext = grid.external_potential(wall_left=wall_left, wall_right=wall_right)
        self.accessible = grid.is_accessible(wall_left=wall_left, wall_right=wall_right)

        # Precompute constant bulk direct correlation scalar c1_bulk
        self.c1_bulk = self.compute_c1_bulk()

    def compute_c1_bulk(self) -> float:
        """Compute exact theoretical bulk direct correlation scalar c1_bulk."""
        eta = self.grid.params.eta
        rho_bulk = self.grid.params.rho_bulk
        R = self.grid.params.radius

        # Create uniform bulk weighted density container
        rho_uniform = np.full_like(self.grid.z, rho_bulk)
        wd_bulk = self.calc.compute(rho_uniform)
        df_dn_bulk = self.functional.evaluate_derivatives(wd_bulk)

        # Inspect interior bulk values
        idx_mid = len(self.grid.z) // 2
        d0 = df_dn_bulk["n0"][idx_mid]
        d1 = df_dn_bulk["n1"][idx_mid]
        d2 = df_dn_bulk["n2"][idx_mid]
        d3 = df_dn_bulk["n3"][idx_mid]

        v_sphere = (4.0 / 3.0) * math.pi * (R**3)
        s_sphere = 4.0 * math.pi * (R**2)

        c1_bulk = -(d0 * 1.0 + d1 * R + d2 * s_sphere + d3 * v_sphere)
        return float(c1_bulk)

    def compute_c1(self, rho: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute spatial one-body direct correlation function c^(1)(z) and bulk correlation c1_bulk.

        Formula (Roth Eq. 28):
            c^(1)(z) = - sum_alpha integral (dPhi / dn_alpha)(z') w_alpha(z' - z) dz'
        """
        wd = self.calc.compute(rho)
        df_dn_dict = self.functional.evaluate_derivatives(wd)
        c1_conv_dict = self.convolver.compute_direct_correlation_convolutions(df_dn_dict)

        # Sum convolution components: c^(1)(z) = - sum_alpha conv_alpha(z)
        c1 = np.zeros_like(self.grid.z, dtype=float)
        for key, conv_arr in c1_conv_dict.items():
            c1 -= conv_arr

        return c1, self.c1_bulk

    def solve_step(self, rho: np.ndarray, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Execute a single Picard iteration step.

        Args:
            rho: Current density profile array.
            alpha: Optional mixing parameter override.

        Returns:
            Tuple of (updated rho array, c1(z) array, residual norm).
        """
        mix_alpha = alpha if alpha is not None else self.alpha
        c1, c1_bulk = self.compute_c1(rho)

        # Target density profile: rho_target = rho_bulk * exp(-beta*V_ext + c1 - c1_bulk)
        rho_target = np.zeros_like(self.grid.z, dtype=float)
        delta_c1 = c1[self.accessible] - c1_bulk

        # Protect against exponential numerical overflow during early iterations
        delta_c1_clipped = np.clip(delta_c1, -10.0, 3.5)
        rho_target[self.accessible] = self.grid.params.rho_bulk * np.exp(delta_c1_clipped)

        # Calculate L2 residual norm normalized by domain length Lz
        diff_sq = (rho_target - rho) ** 2
        residual = float(np.sqrt(self.grid.integrate(diff_sq) / self.grid.Lz))

        # Picard relaxation update: rho_next = (1 - alpha) * rho + alpha * rho_target
        rho_next = (1.0 - mix_alpha) * rho + mix_alpha * rho_target
        rho_next = np.maximum(0.0, rho_next)  # Guarantee non-negativity

        return rho_next, c1, residual

    def solve(
        self,
        rho_init: Optional[np.ndarray] = None,
        max_iter: int = 2000,
        tol: float = 1e-7,
        alpha: Optional[float] = None,
    ) -> SolverResult:
        """Execute iterative relaxation solver until convergence or max_iter."""
        mix_alpha = alpha if alpha is not None else self.alpha

        if rho_init is None:
            rho_current = self.grid.initial_density_profile(
                wall_left=self.wall_left, wall_right=self.wall_right
            )
        else:
            rho_current = rho_init.copy()

        history_residual = []
        c1_last = np.zeros_like(self.grid.z)
        converged = False

        for k in range(1, max_iter + 1):
            rho_next, c1_last, res = self.solve_step(rho_current, alpha=mix_alpha)
            history_residual.append(res)
            rho_current = rho_next

            if res < tol:
                converged = True
                break

        return SolverResult(
            converged=converged,
            iterations=len(history_residual),
            residual=history_residual[-1] if history_residual else 0.0,
            rho=rho_current,
            c1=c1_last,
            c1_bulk=self.c1_bulk,
            history_residual=history_residual,
        )
