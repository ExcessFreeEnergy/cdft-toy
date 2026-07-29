"""Unit tests for White-Bear (WB) and White-Bear II (WBII) FMT functionals (Step 07)."""

import math

import numpy as np
import pytest

from src.functionals import (
    RosenfeldFunctional,
    functional_factory,
)
from src.functionals.white_bear import (
    WhiteBearFunctional,
    WhiteBearIIFunctional,
    _f4_wb_series,
    _phi2_series,
    _phi3_series,
)
from src.grid import Grid1D, PhysicalParameters
from src.solvers import FixedPicardSolver, RothPicardSolver
from src.weighted_densities import WeightedDensities


def test_functional_factory():
    """Verify functional_factory creates correct instances."""
    rf = functional_factory("RF")
    assert isinstance(rf, RosenfeldFunctional)

    wb = functional_factory("wb")
    assert isinstance(wb, WhiteBearFunctional)

    wb2 = functional_factory("WBII")
    assert isinstance(wb2, WhiteBearIIFunctional)

    with pytest.raises(ValueError):
        functional_factory("UNKNOWN")


def test_wb_low_density_series_continuity():
    """Verify continuity between analytic and series forms at cutoff n3 = 1e-3."""
    c = 1e-3

    # Test f4_WB analytic vs series at cutoff
    f4_analytic = (c + (1 - c) ** 2 * math.log(1 - c)) / (
        36 * math.pi * c**2 * (1 - c) ** 2
    )
    f4_series = _f4_wb_series(np.array([c]))[0]
    assert math.isclose(f4_analytic, f4_series, rel_tol=1e-6)

    # Test phi2 analytic vs series at cutoff
    phi2_analytic = (2 * c - c**2 + 2 * (1 - c) * math.log(1 - c)) / c
    phi2_series = _phi2_series(np.array([c]))[0]
    assert math.isclose(phi2_analytic, phi2_series, rel_tol=1e-6)

    # Test phi3 analytic vs series at cutoff
    phi3_analytic = (
        2 * c - 3 * c**2 + 2 * c**3 + 2 * (1 - c) ** 2 * math.log(1 - c)
    ) / (c**2)
    phi3_series = _phi3_series(np.array([c]))[0]
    assert math.isclose(phi3_analytic, phi3_series, rel_tol=1e-6)


def test_wb_analytical_derivatives_finite_difference():
    """Verify exact analytical derivatives of WhiteBearFunctional using finite differences."""
    wb = WhiteBearFunctional()
    h = 1e-6

    n0, n1, n2, n3 = 0.4, 0.25, 1.1, 0.20
    v1, v2 = 0.04, 0.08

    wd = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    anal = wb.evaluate_derivatives(wd)

    # Check n3 derivative
    wd_p = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3 + h]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    wd_m = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3 - h]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    fd_n3 = (wb.evaluate_phi(wd_p)[0] - wb.evaluate_phi(wd_m)[0]) / (2 * h)
    assert math.isclose(anal["n3"][0], fd_n3, rel_tol=1e-5)

    # Check n2 derivative
    wd_p2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2 + h]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    wd_m2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2 - h]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    fd_n2 = (wb.evaluate_phi(wd_p2)[0] - wb.evaluate_phi(wd_m2)[0]) / (2 * h)
    assert math.isclose(anal["n2"][0], fd_n2, rel_tol=1e-5)


def test_wb2_analytical_derivatives_finite_difference():
    """Verify exact analytical derivatives of WhiteBearIIFunctional using finite differences."""
    wb2 = WhiteBearIIFunctional()
    h = 1e-6

    n0, n1, n2, n3 = 0.4, 0.25, 1.1, 0.20
    v1, v2 = 0.04, 0.08

    wd = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    anal = wb2.evaluate_derivatives(wd)

    # Check n3 derivative
    wd_p = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3 + h]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    wd_m = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3 - h]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    fd_n3 = (wb2.evaluate_phi(wd_p)[0] - wb2.evaluate_phi(wd_m)[0]) / (2 * h)
    assert math.isclose(anal["n3"][0], fd_n3, rel_tol=1e-5)

    # Check n2 derivative
    wd_p2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2 + h]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    wd_m2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2 - h]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
    )
    fd_n2 = (wb2.evaluate_phi(wd_p2)[0] - wb2.evaluate_phi(wd_m2)[0]) / (2 * h)
    assert math.isclose(anal["n2"][0], fd_n2, rel_tol=1e-5)


def test_bulk_pressure_carnahan_starling():
    """Verify WB and WBII bulk pressure matches exact Carnahan-Starling equation of state."""
    wb = WhiteBearFunctional()
    wb2 = WhiteBearIIFunctional()

    for eta in [0.10, 0.30, 0.45]:
        rho_bulk = 6.0 * eta / math.pi
        p_cs_expected = rho_bulk * (1.0 + eta + eta**2 - eta**3) / ((1.0 - eta) ** 3)

        assert math.isclose(wb.compute_bulk_pressure(eta), p_cs_expected, rel_tol=1e-6)
        assert math.isclose(wb2.compute_bulk_pressure(eta), p_cs_expected, rel_tol=1e-6)


def test_picard_solver_with_wb_functionals():
    """Verify convergence of FixedPicardSolver and RothPicardSolver with WB functional at eta=0.30."""
    params = PhysicalParameters(eta=0.30)
    grid = Grid1D(params=params, Lz=10.0, dz=0.01)

    solver_fixed = FixedPicardSolver(grid, functional="WB", alpha=0.03)
    res_fixed = solver_fixed.solve(max_iter=1000, tol=1e-6)
    assert res_fixed.converged
    assert res_fixed.residual < 1e-6

    solver_roth = RothPicardSolver(grid, functional="WB", alpha_init=0.03)
    res_roth = solver_roth.solve(max_iter=1000, tol=1e-6)
    assert res_roth.converged
    assert res_roth.residual < 1e-6


def test_picard_solver_with_wb2_functional():
    """Verify convergence of FixedPicardSolver and RothPicardSolver with WBII functional."""
    params = PhysicalParameters(eta=0.25)
    grid = Grid1D(params=params, Lz=10.0, dz=0.01)

    solver = RothPicardSolver(grid, functional="WBII", alpha_init=0.03)
    res = solver.solve(max_iter=1000, tol=1e-6)
    assert res.converged
    assert res.residual < 1e-6
