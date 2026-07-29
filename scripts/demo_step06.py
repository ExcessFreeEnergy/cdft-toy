"""Demonstration script for Step 06: Roth's Adaptive Line-Search Mixing Scheme."""

import numpy as np

from src.grid import Grid1D, PhysicalParameters
from src.solvers import FixedPicardSolver, RothPicardSolver


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 06 Demonstration")
    print("=" * 75)

    # 1. High-Density Benchmark Cases (Roth 2010 Figs 1a & 1b)
    for eta_test, fig_name in [(0.4257, "Fig 1a"), (0.4783, "Fig 1b")]:
        params = PhysicalParameters(eta=eta_test)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)

        solver = RothPicardSolver(grid, alpha_init=0.03, wall_left=0.0)
        result = solver.solve(max_iter=2000, tol=1e-6)

        idx_contact = np.searchsorted(grid.z, params.radius)

        print(f"\n1. HIGH-DENSITY BENCHMARK CONVERGENCE ({fig_name}, eta = {eta_test}):")
        print(f"   Converged Status            : {result.converged}")
        print(f"   Iterations Executed (k)    : {result.iterations}")
        print(f"   Final Residual Norm (R)     : {result.residual:.6e}")
        print(f"   Bulk Density (rho_bulk)     : {params.rho_bulk:.6f}")
        print(f"   Rosenfeld Wall Contact Density: {result.rho[idx_contact]:.6f}")

    # 2. Convergence Speedup Comparison: Fixed Picard vs Roth Adaptive Line Search
    print("\n2. SOLVER SPEEDUP COMPARISON (eta = 0.35):")
    params_cmp = PhysicalParameters(eta=0.35)
    grid_cmp = Grid1D(params=params_cmp, Lz=10.0, dz=0.005)

    solver_fixed = FixedPicardSolver(grid_cmp, alpha=0.03, wall_left=0.0)
    res_fixed = solver_fixed.solve(max_iter=2000, tol=1e-6)

    solver_roth = RothPicardSolver(grid_cmp, alpha_init=0.03, wall_left=0.0)
    res_roth = solver_roth.solve(max_iter=2000, tol=1e-6)

    print(f"   Fixed Picard Iterations     : {res_fixed.iterations}")
    print(f"   Roth Adaptive Iterations    : {res_roth.iterations}")
    print(f"   Speedup Factor              : {res_fixed.iterations / res_roth.iterations:.2f}x faster")

    print("\nStep 06 Roth Adaptive Line-Search Solver fully functioning!")
    print("=" * 75)


if __name__ == "__main__":
    main()
