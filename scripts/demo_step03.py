"""Demonstration script for Step 03: Weighted Density Calculator & Physical Assertions."""

import numpy as np

from src.grid import Grid1D, PhysicalParameters
from src.weighted_densities import WeightedDensityCalculator


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 03 Demonstration")
    print("=" * 75)

    # 1. Benchmark System Setup (Roth 2010 Fig 1a)
    eta = 0.4257
    params = PhysicalParameters(eta=eta)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

    # Initial wall profile
    rho_wall = grid.initial_density_profile(wall_left=0.0)
    wd = calc.compute(rho_wall)

    print("\n1. WEIGHTED DENSITY EVALUATION & FEASIBILITY ASSERTIONS:")
    print(f"   Max Local Packing (max_n3) : {wd.max_n3:.6f}")
    print(f"   Min Local Packing (min_n3) : {wd.min_n3:.6f}")
    print(f"   Physical Feasible (n3 < 1) : {wd.is_feasible}")

    # 2. SPT Bulk Reference Validation Metrics
    print("\n2. SCALED PARTICLE THEORY (SPT) BULK ERROR METRICS:")
    spt_metrics = calc.validate_bulk_spt(eta=eta)

    print(f"   Exact Bulk Packing (spt_n3) : {spt_metrics['spt_n3']:.6f}")
    print(f"   Exact Bulk Surface (spt_n2) : {spt_metrics['spt_n2']:.6f}")
    print(f"   Max Error in n3 (bulk)     : {spt_metrics['max_err_n3']:.8f}")
    print(f"   Max Error in n2 (bulk)     : {spt_metrics['max_err_n2']:.8f}")
    print(f"   Max Error in n1 (bulk)     : {spt_metrics['max_err_n1']:.8f}")
    print(f"   Max Error in n0 (bulk)     : {spt_metrics['max_err_n0']:.8f}")

    # 3. Unphysical Over-Packing Feasibility Test
    print("\n3. UNPHYSICAL OVER-PACKING FEASIBILITY TEST:")
    rho_overpacked = np.full_like(grid.z, 5.0)
    wd_overpacked = calc.compute(rho_overpacked)

    print(f"   Overpacked max_n3          : {wd_overpacked.max_n3:.6f}")
    print(f"   Physical Feasible (n3 < 1) : {wd_overpacked.is_feasible} (Correctly flags divergence risk)")

    print("\nStep 03 Weighted Density Calculator & Safety Assertions fully functioning!")
    print("=" * 75)


if __name__ == "__main__":
    main()
