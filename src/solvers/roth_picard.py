"""Roth's adaptive line-search Picard solver for Classical Density Functional Theory."""

import math
from typing import Optional, Tuple, Union

import numpy as np

from src.functionals.base import FMTFunctional
from src.grid import Grid1D
from src.solvers.base import DFTSolver, SolverResult
from src.weighted_densities import WeightedDensityCalculator


class RothPicardSolver(DFTSolver):
    """Roth's adaptive line-search Picard solver for FMT hard-sphere cDFT (Roth Sec. 8.1 Eq. 29).

    Calculates optimal scalar mixing parameter alpha_opt at each iteration to minimize residual growth,
    enabling fast, robust convergence at high packing fractions (eta >= 0.45).

    Args:
        grid: Grid1D spatial domain discretization instance.
        functional: FMTFunctional excess free energy functional instance.
        alpha_init: Initial fallback mixing parameter (default 0.03).
        alpha_min: Minimum allowable mixing parameter (default 0.001).
        alpha_max: Maximum allowable mixing parameter (default 0.50).
        wall_left: Position of left hard wall (default 0.0).
        wall_right: Position of right hard wall (default None).
    """

    def __init__(
        self,
        grid: Grid1D,
        functional: Optional[Union[str, FMTFunctional]] = None,
        functional_name: Optional[str] = None,
        alpha_init: float = 0.03,
        alpha_min: float = 0.001,
        alpha_max: float = 0.10,
        wall_left: Optional[float] = 0.0,
        wall_right: Optional[float] = None,
    ) -> None:
        from src.functionals import RosenfeldFunctional, functional_factory

        self.grid = grid
        if isinstance(functional, str):
            self.functional = functional_factory(functional)
        elif functional_name is not None:
            self.functional = functional_factory(functional_name)
        elif isinstance(functional, FMTFunctional):
            self.functional = functional
        else:
            self.functional = RosenfeldFunctional()

        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
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
        """Compute spatial one-body direct correlation function c^(1)(z) and bulk correlation c1_bulk."""
        wd = self.calc.compute(rho)
        df_dn_dict = self.functional.evaluate_derivatives(wd)
        c1_conv_dict = self.convolver.compute_direct_correlation_convolutions(df_dn_dict)

        # Sum convolution components: c^(1)(z) = - sum_alpha conv_alpha(z)
        c1 = np.zeros_like(self.grid.z, dtype=float)
        for key, conv_arr in c1_conv_dict.items():
            c1 -= conv_arr

        return c1, self.c1_bulk

    def compute_alpha_opt(self, delta_rho_in: np.ndarray, delta_rho_out: np.ndarray) -> float:
        """Compute Roth's optimal line-search mixing parameter alpha_opt (Roth Eq. 29).

        Formula:
            alpha_opt = - integral(delta_rho_in * delta_rho_out) / integral(delta_rho_out^2)
        """
        num = self.grid.integrate(delta_rho_in * delta_rho_out)
        den = self.grid.integrate(delta_rho_out**2)

        if den <= 1e-15:
            alpha = self.alpha_init
        else:
            alpha = -num / den

        # Safety bounds clamping
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, alpha))
        return float(alpha_clamped)

    def solve_step(
        self,
        rho_current: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fixed-step fallback implementation of solve_step for DFTSolver interface compliance."""
        mix_alpha = alpha if alpha is not None else self.alpha_init
        c1, c1_bulk = self.compute_c1(rho_current)

        rho_target = np.zeros_like(self.grid.z, dtype=float)
        delta_c1 = np.clip(c1[self.accessible] - c1_bulk, -10.0, 3.5)
        rho_target[self.accessible] = self.grid.params.rho_bulk * np.exp(delta_c1)

        diff_sq = (rho_target - rho_current) ** 2
        residual = float(np.sqrt(self.grid.integrate(diff_sq) / self.grid.Lz))

        rho_next = (1.0 - mix_alpha) * rho_current + mix_alpha * rho_target
        rho_next = np.maximum(0.0, rho_next)

        return rho_next, c1, residual

    def solve_step_adaptive(
        self,
        rho_current: np.ndarray,
        rho_prev: Optional[np.ndarray] = None,
        rho_target_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Execute a single adaptive Roth line-search Picard step.

        Returns:
            Tuple of (rho_next, rho_target, c1, residual_norm, alpha_used).
        """
        c1, c1_bulk = self.compute_c1(rho_current)

        rho_target = np.zeros_like(self.grid.z, dtype=float)
        delta_c1 = np.clip(c1[self.accessible] - c1_bulk, -10.0, 3.5)
        rho_target[self.accessible] = self.grid.params.rho_bulk * np.exp(delta_c1)

        diff_sq = (rho_target - rho_current) ** 2
        residual = float(np.sqrt(self.grid.integrate(diff_sq) / self.grid.Lz))

        # Calculate optimal mixing alpha_opt if previous iteration history is available
        if rho_prev is None or rho_target_prev is None:
            alpha_opt = self.alpha_init
        else:
            delta_rho_in = rho_current - rho_prev
            delta_rho_target = rho_target - rho_target_prev
            delta_rho_out = delta_rho_target - delta_rho_in
            alpha_opt = self.compute_alpha_opt(delta_rho_in, delta_rho_out)

        # Physical feasibility back-tracking: ensure max n_3 < 1.0
        alpha_used = alpha_opt
        while True:
            rho_next = (1.0 - alpha_used) * rho_current + alpha_used * rho_target
            rho_next = np.maximum(0.0, rho_next)

            wd_next = self.calc.compute(rho_next)
            if wd_next.is_feasible or alpha_used <= self.alpha_min:
                break
            alpha_used /= 2.0

        return rho_next, rho_target, c1, residual, alpha_used

    def solve(
        self,
        rho_init: Optional[np.ndarray] = None,
        max_iter: int = 2000,
        tol: float = 1e-7,
        alpha: Optional[float] = None,
    ) -> SolverResult:
        """Execute adaptive line-search relaxation solver until convergence or max_iter."""
        if rho_init is None:
            rho_current = self.grid.initial_density_profile(
                wall_left=self.wall_left, wall_right=self.wall_right
            )
        else:
            rho_current = rho_init.copy()

        rho_prev = None
        rho_target_prev = None
        history_residual = []
        c1_last = np.zeros_like(self.grid.z)
        converged = False

        for k in range(1, max_iter + 1):
            rho_next, rho_target_cur, c1_last, res, alpha_used = self.solve_step_adaptive(
                rho_current, rho_prev, rho_target_prev
            )

            history_residual.append(res)

            if res < tol:
                converged = True
                rho_current = rho_next
                break

            rho_prev = rho_current.copy()
            rho_target_prev = rho_target_cur.copy()
            rho_current = rho_next

        return SolverResult(
            converged=converged,
            iterations=len(history_residual),
            residual=history_residual[-1] if history_residual else 0.0,
            rho=rho_current,
            c1=c1_last,
            c1_bulk=self.c1_bulk,
            history_residual=history_residual,
        )
