"""Unit tests for WeightedDensityCalculator and WeightedDensities (Step 03)."""

import numpy as np
import pytest

from src.grid import Grid1D, PhysicalParameters
from src.weighted_densities import WeightedDensityCalculator


def test_weighted_density_calculator_computation():
    """Verify 6-component weighted density computation and dataclass properties."""
    params = PhysicalParameters(eta=0.4257)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    calc = WeightedDensityCalculator(grid)
    rho_init = grid.initial_density_profile(wall_left=0.0)

    wd = calc.compute(rho_init)

    # Check 6 components present
    assert hasattr(wd, "n0")
    assert hasattr(wd, "n1")
    assert hasattr(wd, "n2")
    assert hasattr(wd, "n3")
    assert hasattr(wd, "v1")
    assert hasattr(wd, "v2")

    # Check array lengths match grid.z
    assert len(wd.n3) == len(grid.z)

    # Check dataclass properties
    assert pytest.approx(wd.max_n3) == np.max(wd.n3)
    assert pytest.approx(wd.min_n3) == np.min(wd.n3)
    assert wd.is_feasible is True


def test_physical_feasibility_check():
    """Verify physical feasibility assertion (is_feasible = False when max_n3 >= 1.0)."""
    grid = Grid1D(Lz=5.0, dz=0.005)
    calc = WeightedDensityCalculator(grid)

    # Valid profile: rho = 0.5
    rho_valid = np.full_like(grid.z, 0.5)
    wd_valid = calc.compute(rho_valid)
    assert wd_valid.is_feasible is True

    # Over-packed unphysical profile: rho = 10.0 => n3 > 1.0
    rho_unphysical = np.full_like(grid.z, 10.0)
    wd_unphysical = calc.compute(rho_unphysical)
    assert wd_unphysical.max_n3 >= 1.0
    assert wd_unphysical.is_feasible is False


def test_spt_bulk_validation_benchmark_packing_fractions():
    """Verify bulk SPT error metrics across benchmark packing fractions."""
    for eta_test in [0.1, 0.4257, 0.4783]:
        params = PhysicalParameters(eta=eta_test)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)

        calc = WeightedDensityCalculator(grid)
        spt_metrics = calc.validate_bulk_spt(eta=eta_test)

        assert spt_metrics["spt_n3"] == eta_test
        assert spt_metrics["max_err_n3"] < 1e-3
        assert spt_metrics["max_err_n2"] < 1e-3
        assert spt_metrics["max_err_n1"] < 1e-3
        assert spt_metrics["max_err_n0"] < 1e-3
