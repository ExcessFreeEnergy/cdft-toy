"""Unit tests for RosenfeldFunctional excess free energy density and analytical derivatives (Step 04)."""

import numpy as np
import pytest

from src.functionals.rosenfeld import RosenfeldFunctional
from src.grid import Grid1D, PhysicalParameters
from src.weighted_densities import WeightedDensities, WeightedDensityCalculator


def test_finite_difference_partial_derivatives():
    """Verify analytical partial derivatives (dPhi/dn_alpha) against central finite differences."""
    params = PhysicalParameters(eta=0.4257)
    grid = Grid1D(params=params, Lz=5.0, dz=0.005)
    calc = WeightedDensityCalculator(grid)

    rho = grid.initial_density_profile(wall_left=0.0)
    wd = calc.compute(rho)

    func = RosenfeldFunctional()
    analytical_derivatives = func.evaluate_derivatives(wd)

    eps = 1e-7
    wd_dict = wd.to_dict()

    for key in ["n0", "n1", "n2", "n3", "v1", "v2"]:
        # Compute forward step
        wd_plus_dict = {k: v.copy() for k, v in wd_dict.items()}
        wd_plus_dict[key] += eps
        wd_plus = WeightedDensities(**wd_plus_dict)
        phi_plus = func.evaluate_phi(wd_plus)

        # Compute backward step
        wd_minus_dict = {k: v.copy() for k, v in wd_dict.items()}
        wd_minus_dict[key] -= eps
        wd_minus = WeightedDensities(**wd_minus_dict)
        phi_minus = func.evaluate_phi(wd_minus)

        # Central finite difference approximation
        fd_derivative = (phi_plus - phi_minus) / (2.0 * eps)
        analytical_derivative = analytical_derivatives[key]

        # Ignore unphysical boundary region where n3 ~ 1 or 0 division edge cases occurs
        valid_mask = np.isfinite(fd_derivative) & np.isfinite(analytical_derivative)
        assert np.allclose(
            analytical_derivative[valid_mask],
            fd_derivative[valid_mask],
            rtol=1e-5,
            atol=1e-6,
        )


def test_bulk_percus_yevick_pressure_agreement():
    """Verify that (dPhi/dn3) in bulk fluid matches exact Percus-Yevick compressibility pressure."""
    func = RosenfeldFunctional()

    for eta_test in [0.1, 0.25, 0.4257, 0.4783]:
        params = PhysicalParameters(eta=eta_test)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)
        calc = WeightedDensityCalculator(grid)

        rho_bulk = np.full_like(grid.z, params.rho_bulk)
        wd = calc.compute(rho_bulk)

        derivatives = func.evaluate_derivatives(wd)

        # Inspect interior bulk region
        bulk_idx = grid.num_points // 2
        computed_d_n3 = derivatives["n3"][bulk_idx]

        expected_py_pressure = func.compute_bulk_pressure(eta=eta_test, sigma=params.sigma)
        assert pytest.approx(computed_d_n3, rel=1e-3) == expected_py_pressure


def test_total_excess_free_energy_integration():
    """Test spatial integration of excess free energy F_ex = integral Phi(z) dz."""
    params = PhysicalParameters(eta=0.4257)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)
    calc = WeightedDensityCalculator(grid)

    rho = grid.initial_density_profile(wall_left=0.0)
    wd = calc.compute(rho)

    func = RosenfeldFunctional()
    f_ex = func.compute_total_free_energy(grid, wd)

    assert isinstance(f_ex, float)
    assert np.isfinite(f_ex)
