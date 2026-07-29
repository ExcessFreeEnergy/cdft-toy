"""Unit tests for Tarazona Tensorial FMT functional (Step 08)."""

import math

import numpy as np
import pytest

from src.functionals import functional_factory
from src.functionals.tarazona_tensor import WhiteBearTensorFunctional
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver
from src.weighted_densities import WeightedDensities, WeightedDensityCalculator
from src.weights import PlanarWeights


def test_w_m2_analytical_integral():
    """Verify analytical integral int_{-R}^R w_m2(z) dz == 0."""
    pw = PlanarWeights(radius=0.5)
    integrals = pw.evaluate_analytical_integrals()
    assert math.isclose(integrals["n_m2_integral"], 0.0, abs_tol=1e-12)

    # Numerical Simpson integral check
    z_w, weights = pw.get_grid_and_weights(dz=0.001, apply_endpoint_modification=False)
    num_integral = np.trapezoid(weights["n_m2"], x=z_w)
    assert math.isclose(num_integral, 0.0, abs_tol=1e-3)


def test_w_m2_parity_even():
    """Verify w_m2 is an even function under z -> -z."""
    pw = PlanarWeights(radius=0.5)
    z_w, weights = pw.get_grid_and_weights(dz=0.005, apply_endpoint_modification=False)
    w_m2 = weights["n_m2"]

    # Check w_m2(z) == w_m2(-z)
    assert np.allclose(w_m2, w_m2[::-1], atol=1e-12)
    assert not PlanarWeights.PARITY_IS_VECTOR["n_m2"]


def test_functional_factory_wb_tensor():
    """Verify functional_factory creates WhiteBearTensorFunctional."""
    wbt = functional_factory("WB-TENSOR")
    assert isinstance(wbt, WhiteBearTensorFunctional)

    wbt2 = functional_factory("WBT")
    assert isinstance(wbt2, WhiteBearTensorFunctional)


def test_wb_tensor_analytical_derivatives_finite_difference():
    """Verify exact analytical derivatives of WhiteBearTensorFunctional using finite differences."""
    wbt = WhiteBearTensorFunctional()
    h = 1e-6

    n0, n1, n2, n3 = 0.4, 0.25, 1.1, 0.20
    v1, v2 = 0.04, 0.08
    n_m2 = 0.05

    wd = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
        n_m2=np.array([n_m2]),
    )
    anal = wbt.evaluate_derivatives(wd)

    # Check n_m2 derivative
    wd_p_m2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
        n_m2=np.array([n_m2 + h]),
    )
    wd_m_m2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2]),
        n_m2=np.array([n_m2 - h]),
    )
    fd_m2 = (wbt.evaluate_phi(wd_p_m2)[0] - wbt.evaluate_phi(wd_m_m2)[0]) / (2 * h)
    assert math.isclose(anal["n_m2"][0], fd_m2, rel_tol=1e-5)

    # Check v2 derivative
    wd_p_v2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2 + h]),
        n_m2=np.array([n_m2]),
    )
    wd_m_v2 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3]),
        v1=np.array([v1]),
        v2=np.array([v2 - h]),
        n_m2=np.array([n_m2]),
    )
    fd_v2 = (wbt.evaluate_phi(wd_p_v2)[0] - wbt.evaluate_phi(wd_m_v2)[0]) / (2 * h)
    assert math.isclose(anal["v2"][0], fd_v2, rel_tol=1e-5)

    # Check n3 derivative
    wd_p_n3 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3 + h]),
        v1=np.array([v1]),
        v2=np.array([v2]),
        n_m2=np.array([n_m2]),
    )
    wd_m_n3 = WeightedDensities(
        n0=np.array([n0]),
        n1=np.array([n1]),
        n2=np.array([n2]),
        n3=np.array([n3 - h]),
        v1=np.array([v1]),
        v2=np.array([v2]),
        n_m2=np.array([n_m2]),
    )
    fd_n3 = (wbt.evaluate_phi(wd_p_n3)[0] - wbt.evaluate_phi(wd_m_n3)[0]) / (2 * h)
    assert math.isclose(anal["n3"][0], fd_n3, rel_tol=1e-5)


def test_wb_tensor_solver_convergence_tight_confinement():
    """Verify convergence of RothPicardSolver under tight slit-pore confinement (Lz = 2.0 sigma)."""
    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=2.0, dz=0.005)

    solver = RothPicardSolver(grid, functional="WB-TENSOR", alpha_init=0.03, wall_left=0.0, wall_right=2.0)
    res = solver.solve(max_iter=1000, tol=1e-6)

    assert res.converged
    assert res.residual < 1e-6
    assert np.all(np.isfinite(res.rho))
