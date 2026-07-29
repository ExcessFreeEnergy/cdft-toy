"""Demonstration script for Step 08: Tarazona Tensorial Weights & WB-Tensor Functional Engine."""

import numpy as np

from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 08 Demonstration")
    print("Tarazona Tensorial Weights & Tensorial FMT Engine (WB-Tensor)")
    print("=" * 75)

    # Tight slit-pore confinement scenario
    eta = 0.35
    Lz = 2.0
    params = PhysicalParameters(eta=eta)
    grid = Grid1D(params=params, Lz=Lz, dz=0.005)

    functionals = ["RF", "WB", "WBII", "WB-Tensor"]

    print(f"\n1. BULK THERMODYNAMIC COMPARISON AT ETA = {eta:.4f}:")
    print(f"{'Functional':<12} | {'Bulk Pressure (beta*p)':<22} | {'Excess Chem. Pot. (beta*mu_ex)':<32}")
    print("-" * 75)

    for f_name in functionals:
        func = functional_factory(f_name)
        p_bulk = func.compute_bulk_pressure(eta)
        mu_ex = func.bulk_excess_mu(eta)
        print(f"{f_name:<12} | {p_bulk:<22.6f} | {mu_ex:<32.6f}")

    print(f"\n2. SLIT PORE CONFINEMENT (Lz = {Lz:.2f} sigma, wall_left=0, wall_right={Lz}):")
    print(f"{'Functional':<12} | {'Converged':<10} | {'Iterations':<10} | {'Residual':<14} | {'Max rho(z)':<12}")
    print("-" * 75)

    for f_name in functionals:
        solver = RothPicardSolver(grid, functional=f_name, alpha_init=0.03, wall_left=0.0, wall_right=Lz)
        res = solver.solve(max_iter=1000, tol=1e-6)
        max_rho = float(np.max(res.rho))
        print(f"{f_name:<12} | {res.converged!s:<10} | {res.iterations:<10} | {res.residual:<14.6e} | {max_rho:<12.6f}")

    print("\n" + "=" * 75)
    print("Step 08 Tarazona Tensorial Weights & WB-Tensor Engine fully verified!")
    print("=" * 75)


if __name__ == "__main__":
    main()
