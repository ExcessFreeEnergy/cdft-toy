"""Demonstration script for Step 01: Physical Parameters, 1D Spatial Grid, and Hard Wall Potential."""

import numpy as np

from src.grid import Grid1D, PhysicalParameters


def main():
    print("=" * 70)
    print("Classical DFT (FMT) Simulator - Step 01 Demonstration")
    print("=" * 70)

    # 1. Benchmark Physical Parameters from Roth (2010) Fig. 1
    eta1 = 0.4257
    eta2 = 0.4783

    params1 = PhysicalParameters(eta=eta1)
    params2 = PhysicalParameters(eta=eta2)

    print("\n1. PHYSICAL PARAMETERS & DENSITY CONVERSIONS:")
    print(f"   Sphere Diameter (sigma) : {params1.sigma:.4f}")
    print(f"   Sphere Radius (R)        : {params1.radius:.4f}")
    print(f"   Sphere Volume (V_sphere) : {params1.volume:.6f}")
    print("-" * 50)
    print(f"   Benchmark 1 (Roth Fig 1a) -> eta = {params1.eta:.4f} => rho_bulk = {params1.rho_bulk:.6f}")
    print(f"   Benchmark 2 (Roth Fig 1b) -> eta = {params2.eta:.4f} => rho_bulk = {params2.rho_bulk:.6f}")

    # 2. Domain Discretization & Grid Setup
    Lz = 10.0  # 10 * sigma
    dz = 0.005  # 0.005 * sigma

    grid = Grid1D(params=params1, Lz=Lz, dz=dz)

    print("\n2. 1D SPATIAL GRID DISCRETIZATION:")
    print(f"   Domain Length (Lz)  : [0.0, {grid.Lz:.2f}] sigma")
    print(f"   Grid Resolution (dz): {grid.dz:.4f} sigma (satisfies dz <= 0.01 sigma)")
    print(f"   Total Grid Points   : {grid.num_points}")

    # 3. External Potential & Initial Density Profile
    v_ext = grid.external_potential(wall_left=0.0)
    rho_init = grid.initial_density_profile(wall_left=0.0)

    print("\n3. EXTERNAL HARD WALL POTENTIAL & DENSITY PROFILE NEAR WALL (z = 0):")
    print("   Showing spatial sampling across the hard wall boundary (R = 0.5):")
    print("-" * 60)
    print(f"   {'z / sigma':>12} | {'V_ext(z)':>14} | {'rho_init(z)':>14} | Status")
    print("-" * 60)

    # Pick sample indices around the hard wall boundary z = R = 0.5
    sample_z = [0.0, 0.25, 0.49, 0.495, 0.50, 0.505, 0.51, 1.0, 2.0, 5.0]
    for z_val in sample_z:
        idx = round(z_val / grid.dz)
        z_actual = grid.z[idx]
        v_val = v_ext[idx]
        rho_val = rho_init[idx]

        v_str = "inf" if np.isinf(v_val) else f"{v_val:.4f}"
        status = "Forbidden (Overlap)" if np.isinf(v_val) else "Accessible (Bulk)"

        print(f"   {z_actual:12.4f} | {v_str:>14} | {rho_val:14.6f} | {status}")

    print("-" * 60)

    # 4. Numerical Quadrature
    total_particles = grid.integrate(rho_init)
    accessible_length = grid.Lz - params1.radius
    expected_particles = accessible_length * params1.rho_bulk

    print("\n4. DOMAIN QUADRATURE & INTEGRATION:")
    print(f"   Accessible Domain Length : {accessible_length:.4f} sigma")
    print(f"   Integrated Particle Count: {total_particles:.6f}")
    print(f"   Expected Particle Count  : {expected_particles:.6f}")

    print("\nStep 01 Grid & Physics Module initialized and functioning correctly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
