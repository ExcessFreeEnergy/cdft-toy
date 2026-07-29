"""Unit tests for Dimensional Crossover & Confinement Engine (Step 10)."""

import math

import numpy as np

from src.crossover import CrossoverAnalyzer
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters


def test_zero_d_gaussian_normalization():
    """Verify generate_zero_d_gaussian creates a 1-particle normalized density profile."""
    params = PhysicalParameters(eta=0.30)
    grid = Grid1D(params=params, Lz=4.0, dz=0.005)

    rho_g = CrossoverAnalyzer.generate_zero_d_gaussian(grid, alpha_width=0.05)
    integral = grid.integrate(rho_g)

    assert math.isclose(integral, 1.0, rel_tol=1e-3)
    assert np.all(rho_g >= 0.0)


def test_zero_d_free_energy_evaluation():
    """Verify zero-D cavity free energy evaluation for WB-Tensor vs scalar functionals."""
    params = PhysicalParameters(eta=0.30)
    grid = Grid1D(params=params, Lz=4.0, dz=0.005)

    wbt = functional_factory("WB-TENSOR")
    rf = functional_factory("RF")

    alphas = [0.10, 0.05, 0.03]
    phi_wbt = CrossoverAnalyzer.evaluate_zero_d_divergence(grid, alphas, wbt)
    phi_rf = CrossoverAnalyzer.evaluate_zero_d_divergence(grid, alphas, rf)

    assert len(phi_wbt) == 3
    assert len(phi_rf) == 3
    assert np.all(np.isfinite(phi_wbt))


def test_sweep_pore_confinement():
    """Verify sweep_pore_confinement returns metrics for requested functionals across pore widths."""
    widths = [0.8, 1.5, 2.5]
    results = CrossoverAnalyzer.sweep_pore_confinement(widths, eta_bulk=0.30, functionals=["WB", "WB-Tensor"])

    assert "WB" in results
    assert "WB-Tensor" in results
    assert len(results["WB"]) == 3

    for m in results["WB-Tensor"]:
        assert m.converged
        assert np.isfinite(m.max_rho)
        assert np.isfinite(m.max_phi)
