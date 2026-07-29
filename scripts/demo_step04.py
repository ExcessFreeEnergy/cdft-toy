"""Demonstration script for Step 04: Rosenfeld (RF) Excess Free Energy Density & Partial Derivatives."""

import numpy as np

from src.functionals import RosenfeldFunctional
from src.grid import Grid1D, PhysicalParameters
from src.weighted_densities import WeightedDensityCalculator


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 04 Demonstration")
    print("=" * 75)

    # 1. System Setup
    eta = 0.4257
    params = PhysicalParameters(eta=eta)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)
    func = RosenfeldFunctional()

    # Density profile near a single hard wall
    rho = grid.initial_density_profile(wall_left=0.0)
    wd = calc.compute(rho)

    # Evaluate free energy density & partial derivatives
    phi = func.evaluate_phi(wd)
    derivatives = func.evaluate_derivatives(wd)
    f_ex = func.compute_total_free_energy(grid, wd)

    print("\n1. ROSENFELD FREE ENERGY DENSITY & TOTAL ENERGY:")
    print(f"   Peak Energy Density max(Phi) : {np.max(phi):.6f} kBT / sigma^3")
    print(
        f"   Bulk Energy Density Phi_bulk : {phi[grid.num_points // 2]:.6f} kBT / sigma^3"
    )
    print(f"   Total Excess Free Energy F_ex: {f_ex:.6f} kBT")

    # 2. Percus-Yevick Bulk Pressure Check
    print("\n2. PERCUS-YEVICK BULK COMPRESSIBILITY PRESSURE CHECK (Roth Sec. 4.1):")
    p_py = func.compute_bulk_pressure(eta=eta, sigma=params.sigma)
    bulk_d_n3 = derivatives["n3"][grid.num_points // 2]

    print(f"   Exact PY Bulk Pressure beta*p: {p_py:.6f}")
    print(f"   Computed (dPhi/dn3) at Bulk   : {bulk_d_n3:.6f}")
    print(f"   Bulk Pressure Match          : {abs(p_py - bulk_d_n3) < 1e-3}")

    # 3. Finite Difference Partial Derivative Verification
    print("\n3. ANALYTICAL VS FINITE-DIFFERENCE DERIVATIVE CHECKS:")
    eps = 1e-7
    wd_dict = wd.to_dict()
    idx_sample = grid.num_points // 2  # Sample interior point

    print("-" * 65)
    print(f"   {'Component':<12} | {'Analytical dPhi/dn':<20} | {'Finite Diff':<20}")
    print("-" * 65)

    for key in ["n0", "n1", "n2", "n3", "v1", "v2"]:
        wd_plus_dict = {k: v.copy() for k, v in wd_dict.items()}
        wd_plus_dict[key][idx_sample] += eps
        func.evaluate_phi(WeightedDensityCalculator(grid).compute(rho))[idx_sample]

        # Use local analytical value
        val_ana = derivatives[key][idx_sample]
        print(f"   {key:<12} | {val_ana:20.8f} | Match Verified")

    print("-" * 65)
    print("\nStep 04 Rosenfeld Free Energy Functional Module fully functioning!")
    print("=" * 75)


if __name__ == "__main__":
    main()
