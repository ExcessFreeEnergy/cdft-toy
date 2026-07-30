"""End-to-End Regression Test Suite & Performance Benchmarks (Step 12)."""

import time

import numpy as np

from src.diagnostics import SumRuleDiagnostics
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver
from src.weighted_densities import WeightedDensityCalculator


def test_monte_carlo_figure1_contact_densities():
    """Verify computed wall contact density rho(R+) against theoretical bulk pressure for Fig 1 benchmarks."""
    benchmark_etas = [0.4257, 0.4783]  # Roth (2010) Fig. 1a and Fig. 1b

    for eta_val in benchmark_etas:
        params = PhysicalParameters(eta=eta_val)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)
        calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

        for f_name in ["RF", "WB", "WBII", "WB-Tensor"]:
            func = functional_factory(f_name)
            solver = RothPicardSolver(grid, functional=func, alpha_init=0.03, wall_left=0.0, wall_right=None)
            res = solver.solve(max_iter=1000, tol=1e-6)

            assert res.converged, f"Solver failed to converge for {f_name} at eta={eta_val}"

            diag = SumRuleDiagnostics.evaluate_all(grid, res.rho, func, calc)

            # Contact theorem verification: rho(R+) agrees with exact bulk pressure within < 5%
            assert diag.contact_error_rel < 0.05


def test_fft_convolution_performance_benchmark():
    """Verify 1000 forward FFT convolution passes execute in under 1.0 second on a 1000-point grid."""
    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=10.0, dz=0.010)  # N = 1001 grid points
    calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

    v_ext = np.where(grid.z < params.radius - 1e-12, 1e10, 0.0)
    rho = params.rho_bulk * np.exp(-v_ext)

    start_t = time.perf_counter()
    for _ in range(1000):
        _wd = calc.compute(rho)
    elapsed_t = time.perf_counter() - start_t

    assert elapsed_t < 1.0, f"1000 FFT convolutions took {elapsed_t:.4f}s (expected < 1.0s)"


def test_picard_solver_performance_benchmark():
    """Verify 100 Roth Picard solver iterations execute in under 5.0 seconds on a 1000-point grid."""
    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=10.0, dz=0.010)  # N = 1001 grid points
    func = functional_factory("WB-Tensor")
    solver = RothPicardSolver(grid, functional=func, alpha_init=0.03, wall_left=0.0, wall_right=None)

    rho_prev = params.rho_bulk * np.exp(-solver.v_ext)
    rho_current = rho_prev.copy()
    c1, c1_bulk = solver.compute_c1(rho_current)
    c1_ex = c1 - c1_bulk
    mu_ex = func.bulk_excess_mu(params.eta, sigma=params.sigma)
    arg = np.clip(mu_ex + c1_ex - solver.v_ext, -700.0, 700.0)
    rho_target_prev = params.rho_bulk * np.exp(arg) * 0.0 + 1.0

    start_t = time.perf_counter()
    for _ in range(100):
        rho_next, rho_target_cur, _, _, _ = solver.solve_step_adaptive(rho_current, rho_prev, rho_target_prev)
        rho_prev = rho_current
        rho_target_prev = rho_target_cur
        rho_current = rho_next
    elapsed_t = time.perf_counter() - start_t

    assert elapsed_t < 5.0, f"100 Picard solver steps took {elapsed_t:.4f}s (expected < 5.0s)"
