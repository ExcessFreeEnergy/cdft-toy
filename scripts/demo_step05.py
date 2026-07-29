"""Demonstration script for Step 05: One-Body Direct Correlation Function & Fixed-Step Solver."""

import numpy as np

from src.grid import Grid1D, PhysicalParameters
from src.solvers import FixedPicardSolver


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 05 Demonstration")
    print("=" * 75)

    # 1. System Setup
    eta = 0.25
    params = PhysicalParameters(eta=eta)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    solver = FixedPicardSolver(grid, alpha=0.03, wall_left=0.0)

    print("\n1. ONE-BODY DIRECT CORRELATION FUNCTION c^(1)(z) & BULK CORRELATION:")
    rho_init = grid.initial_density_profile(wall_left=0.0)
    c1, c1_bulk = solver.compute_c1(rho_init)

    print(f"   Theoretical Bulk c1_bulk    : {c1_bulk:.6f}")
    print(f"   Computed c1(z) at z=R (wall): {c1[np.searchsorted(grid.z, 0.5)]:.6f}")
    print(f"   Computed c1(z) in Bulk      : {c1[grid.num_points // 2]:.6f}")

    # 2. Iterative Fixed Picard Solver Convergence
    print("\n2. SOLVING EQUILIBRIUM DENSITY PROFILE (Picard Relaxation):")
    result = solver.solve(max_iter=1000, tol=1e-6)

    idx_contact = np.searchsorted(grid.z, params.radius)

    print(f"   Converged Status            : {result.converged}")
    print(f"   Iterations Executed (k)    : {result.iterations}")
    print(f"   Final Residual Norm (R)     : {result.residual:.6e}")
    print(f"   Bulk Density (rho_bulk)     : {params.rho_bulk:.6f}")
    print(f"   Contact Density (rho_wall)  : {result.rho[idx_contact]:.6f}")
    print(f"   Contact Density Accumulation: {result.rho[idx_contact] / params.rho_bulk:.2f}x bulk")

    print("\nStep 05 One-Body Direct Correlation & Picard Solver fully functioning!")
    print("=" * 75)


if __name__ == "__main__":
    main()
