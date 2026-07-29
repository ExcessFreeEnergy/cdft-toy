"""Unit tests for RothPicardSolver and adaptive line-search mixing scheme (Step 06)."""

import numpy as np
import pytest

from src.grid import Grid1D, PhysicalParameters
from src.solvers.picard import FixedPicardSolver
from src.solvers.roth_picard import RothPicardSolver


def test_alpha_opt_calculation():
    """Verify compute_alpha_opt inner product ratio calculation."""
    grid = Grid1D(Lz=5.0, dz=0.005)
    solver = RothPicardSolver(grid, alpha_min=0.001, alpha_max=0.50)

    # Test vectors where - (in . out) / (out . out) = 0.10
    delta_in = np.ones_like(grid.z)
    delta_out = -10.0 * np.ones_like(grid.z)

    alpha_opt = solver.compute_alpha_opt(delta_in, delta_out)
    assert pytest.approx(alpha_opt) == 0.10


def test_roth_picard_high_density_benchmark_convergence():
    """Verify robust convergence for Roth (2010) benchmark packing fractions eta = 0.4257 and 0.4783."""
    for eta_test in [0.4257, 0.4783]:
        params = PhysicalParameters(eta=eta_test)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)

        solver = RothPicardSolver(grid, alpha_init=0.03, wall_left=0.0)
        result = solver.solve(max_iter=2000, tol=1e-6)

        assert result.converged is True
        assert result.residual < 1e-6
        assert result.iterations < 2000
        assert np.all(result.rho >= 0.0)

        # Verify contact density elevation at wall z = R
        idx_contact = np.searchsorted(grid.z, params.radius)
        assert result.rho[idx_contact] > params.rho_bulk * 2.0


def test_roth_picard_speedup_over_fixed_picard():
    """Verify that Roth adaptive line-search solver converges faster than fixed Picard solver."""
    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    solver_fixed = FixedPicardSolver(grid, alpha=0.03, wall_left=0.0)
    res_fixed = solver_fixed.solve(max_iter=1500, tol=1e-6)

    solver_roth = RothPicardSolver(grid, alpha_init=0.03, wall_left=0.0)
    res_roth = solver_roth.solve(max_iter=1500, tol=1e-6)

    assert res_roth.converged is True
    assert res_roth.iterations < res_fixed.iterations
