"""Unit tests for thermodynamic observables and sum-rule validation (Step 09)."""

import math

import numpy as np
import pytest

from src.diagnostics import SumRuleDiagnostics
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver


def test_contact_theorem_convergence():
    """Verify contact theorem sum-rule rho(z = R+) == beta * P_bulk with grid convergence."""
    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=10.0, dz=0.002)
    functional = functional_factory("WB")

    solver = RothPicardSolver(grid, functional=functional, alpha_init=0.03)
    res = solver.solve(max_iter=2000, tol=1e-7)

    assert res.converged

    rho_c, p_bulk, err_rel, passed = SumRuleDiagnostics.verify_contact_theorem(grid, res.rho, functional, tol_rel=0.01)
    assert passed
    assert err_rel < 0.01  # < 1.0% relative error for dz=0.002
    assert math.isclose(rho_c, p_bulk, rel_tol=0.01)


def test_surface_tension_and_bulk_route():
    """Verify surface tension spatial integration and bulk-route formula calculation."""
    params = PhysicalParameters(eta=0.30)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)
    functional = functional_factory("WBII")

    solver = RothPicardSolver(grid, functional=functional, alpha_init=0.03)
    res = solver.solve(max_iter=2000, tol=1e-7)
    assert res.converged

    gamma_spatial = SumRuleDiagnostics.compute_surface_tension(grid, res.rho, functional)
    gamma_bulk_route = SumRuleDiagnostics.compute_bulk_route_surface_tension(functional, eta=0.30, sigma=1.0)

    assert isinstance(gamma_spatial, float)
    assert isinstance(gamma_bulk_route, float)
    assert gamma_bulk_route > 0.0


def test_gibbs_adsorption_sum_rule():
    """Verify Gibbs adsorption theorem -d(gamma)/d(mu) == Gamma_ex to within numerical derivative tolerance."""
    params = PhysicalParameters(eta=0.30)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)
    functional = functional_factory("WB")

    minus_dgamma_dmu, gamma_ex, err_rel = SumRuleDiagnostics.verify_gibbs_adsorption(
        grid, functional, eta=0.30, delta_eta=0.005
    )

    assert math.isclose(minus_dgamma_dmu, gamma_ex, rel_tol=0.10)  # Agree to within 10% for numerical finite-difference
    assert err_rel < 0.10


def test_diagnostics_evaluate_all():
    """Verify SumRuleDiagnostics.evaluate_all returns complete metrics."""
    params = PhysicalParameters(eta=0.25)
    grid = Grid1D(params=params, Lz=8.0, dz=0.005)
    functional = functional_factory("WB-TENSOR")

    solver = RothPicardSolver(grid, functional=functional, alpha_init=0.03)
    res = solver.solve(max_iter=1000, tol=1e-6)

    metrics = SumRuleDiagnostics.evaluate_all(grid, res.rho, functional)
    assert metrics.contact_passed
    assert metrics.contact_error_rel < 0.03
    assert metrics.surface_tension_bulk_route > 0.0
