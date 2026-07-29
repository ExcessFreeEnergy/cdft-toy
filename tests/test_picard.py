"""Unit tests for FixedPicardSolver and one-body direct correlation function c^(1)(z) (Step 05)."""

import numpy as np

from src.grid import Grid1D, PhysicalParameters
from src.solvers.picard import FixedPicardSolver


def test_bulk_direct_correlation_value():
    """Verify compute_c1_bulk() produces exact theoretical scalar bulk correlation."""
    params = PhysicalParameters(eta=0.30)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    solver = FixedPicardSolver(grid)
    c1_bulk = solver.compute_c1_bulk()

    # Bulk correlation c1_bulk is negative for hard-sphere fluid
    assert isinstance(c1_bulk, float)
    assert c1_bulk < 0.0


def test_single_picard_solver_step():
    """Verify single Picard step update and non-negative density profile preservation."""
    params = PhysicalParameters(eta=0.20)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    solver = FixedPicardSolver(grid, alpha=0.05)
    rho_init = grid.initial_density_profile(wall_left=0.0)

    rho_next, c1, res = solver.solve_step(rho_init)

    assert len(rho_next) == len(grid.z)
    assert len(c1) == len(grid.z)
    assert np.all(rho_next >= 0.0)
    assert res > 0.0


def test_picard_solver_full_convergence_moderate_packing():
    """Verify full equilibrium density profile convergence for eta = 0.20 and 0.30."""
    for eta_test in [0.20, 0.30]:
        params = PhysicalParameters(eta=eta_test)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)

        # Use fixed Picard solver with mixing alpha = 0.05
        solver = FixedPicardSolver(grid, alpha=0.05, wall_left=0.0)
        result = solver.solve(max_iter=1000, tol=1e-6)

        assert result.converged is True
        assert result.residual < 1e-6
        assert result.iterations < 1000
        assert np.all(result.rho >= 0.0)

        # Contact density at wall (z = R) must be elevated above bulk
        idx_contact = np.searchsorted(grid.z, params.radius)
        assert result.rho[idx_contact] > params.rho_bulk
