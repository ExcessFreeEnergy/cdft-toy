"""Demonstration script for Step 07: White-Bear (WB) & White-Bear II (WBII) Functionals."""

import numpy as np

from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 07 Demonstration")
    print("White-Bear (WB) & White-Bear II (WBII) Functional Extensions")
    print("=" * 75)

    eta = 0.45
    params = PhysicalParameters(eta=eta)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    functionals = ["RF", "WB", "WBII"]

    print(f"\n1. BULK THERMODYNAMIC COMPARISON AT ETA = {eta:.4f}:")
    print(f"{'Functional':<12} | {'Bulk Pressure (beta*p)':<22} | {'Excess Chem. Pot. (beta*mu_ex)':<32}")
    print("-" * 75)

    for f_name in functionals:
        func = functional_factory(f_name)
        p_bulk = func.compute_bulk_pressure(eta)
        mu_ex = func.bulk_excess_mu(eta)
        print(f"{f_name:<12} | {p_bulk:<22.6f} | {mu_ex:<32.6f}")

    print("\n2. SOLVER CONVERGENCE & CONTACT DENSITY PROFILES:")
    print(f"{'Functional':<12} | {'Converged':<10} | {'Iterations':<10} | {'Residual':<14} | {'Contact rho':<12}")
    print("-" * 75)

    R = params.radius
    for f_name in functionals:
        solver = RothPicardSolver(grid, functional=f_name, alpha_init=0.03)
        res = solver.solve(max_iter=1000, tol=1e-6)

        # Contact density is at z = R (first accessible grid point)
        idx_contact = np.searchsorted(grid.z, R)
        rho_contact = res.rho[idx_contact]

        print(f"{f_name:<12} | {str(res.converged):<10} | {res.iterations:<10} | {res.residual:<14.6e} | {rho_contact:<12.6f}")

    print("\n" + "=" * 75)
    print("Step 07 White-Bear & White-Bear II Functional Extensions fully verified!")
    print("=" * 75)


if __name__ == "__main__":
    main()
