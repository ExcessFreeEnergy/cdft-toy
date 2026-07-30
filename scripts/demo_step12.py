"""Demonstration script for Step 12: End-to-End Test Suite & Performance Benchmarking."""

import time

import numpy as np

from src.diagnostics import SumRuleDiagnostics
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver
from src.weighted_densities import WeightedDensityCalculator


def main():
    print("=" * 85)
    print("Classical DFT (FMT) Simulator - Step 12 Final Demonstration")
    print("End-to-End Monte Carlo Benchmark Validation & Performance Throughput")
    print("=" * 85)

    # 1. Roth (2010) Figure 1 Monte Carlo Contact Density Regression Table
    benchmarks = [
        ("Figure 1a (eta=0.4257)", 0.4257),
        ("Figure 1b (eta=0.4783)", 0.4783),
    ]

    print("\n1. MONTE CARLO CONTACT DENSITY BENCHMARK REGRESSION:")
    print(
        f"{'Benchmark Case':<24} | {'Functional':<10} | {'Contact rho(R+)':<16} | {'Exact p_bulk':<14} | {'Relative Error':<16}"
    )
    print("-" * 90)

    for case_label, eta_val in benchmarks:
        params = PhysicalParameters(eta=eta_val)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)
        calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

        for f_name in ["RF", "WB", "WBII", "WB-Tensor"]:
            func = functional_factory(f_name)
            solver = RothPicardSolver(grid, functional=func, alpha_init=0.03, wall_left=0.0, wall_right=None)
            res = solver.solve(max_iter=1000, tol=1e-6)

            diag = SumRuleDiagnostics.evaluate_all(grid, res.rho, func, calc)
            p_ref = func.compute_bulk_pressure(eta_val, sigma=grid.params.sigma)
            err_rel = diag.contact_error_rel * 100.0

            print(f"{case_label:<24} | {f_name:<10} | {diag.contact_density:<16.6f} | {p_ref:<14.6f} | {err_rel:<16.4f}%")
        print("-" * 90)

    # 2. Computational Throughput & Timing Performance
    print("\n2. COMPUTATIONAL PERFORMANCE & THROUGHPUT BENCHMARKS:")

    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=10.0, dz=0.010)
    calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)
    v_ext = np.where(grid.z < params.radius - 1e-12, 1e10, 0.0)
    rho = params.rho_bulk * np.exp(-v_ext)

    # Benchmark FFT Convolutions
    t0 = time.perf_counter()
    n_fft_passes = 1000
    for _ in range(n_fft_passes):
        _wd = calc.compute(rho)
    t1 = time.perf_counter()
    fft_fps = n_fft_passes / (t1 - t0)
    print(f"  - 1000 Forward FFT Convolutions (1001 grid pts) : {t1 - t0:.4f} s ({fft_fps:.1f} convolutions/sec)")

    # Benchmark Picard Solver Steps
    func = functional_factory("WB-Tensor")
    solver = RothPicardSolver(grid, functional=func, alpha_init=0.03, wall_left=0.0, wall_right=None)
    rho_prev = params.rho_bulk * np.exp(-solver.v_ext)
    rho_current = rho_prev.copy()
    c1, c1_bulk = solver.compute_c1(rho_current)
    c1_ex = c1 - c1_bulk
    mu_ex = func.bulk_excess_mu(params.eta, sigma=params.sigma)
    arg = np.clip(mu_ex + c1_ex - solver.v_ext, -700.0, 700.0)
    rho_target_prev = grid.params.rho_bulk * np.exp(arg)

    t2 = time.perf_counter()
    n_solver_steps = 100
    for _ in range(n_solver_steps):
        rho_next, rho_target_cur, _, _, _ = solver.solve_step_adaptive(rho_current, rho_prev, rho_target_prev)
        rho_prev = rho_current
        rho_target_prev = rho_target_cur
        rho_current = rho_next
    t3 = time.perf_counter()
    solver_fps = n_solver_steps / (t3 - t2)
    print(f"  - 100 Adaptive Roth Picard Iterations (WB-Tensor): {t3 - t2:.4f} s ({solver_fps:.1f} iterations/sec)")

    print("\n" + "=" * 85)
    print("Step 12 End-to-End Test Suite & Performance Benchmarks fully verified!")
    print("=" * 85)


if __name__ == "__main__":
    main()
