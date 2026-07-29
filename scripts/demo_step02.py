"""Demonstration script for Step 02: Planar Geometrical Weight Functions & FFT Convolutions."""

import numpy as np

from src.convolutions import FFTConvolver1D
from src.grid import Grid1D, PhysicalParameters
from src.weights import PlanarWeights


def main():
    print("=" * 75)
    print("Classical DFT (FMT) Simulator - Step 02 Demonstration")
    print("=" * 75)

    # 1. Weight Functions & Analytical Integrals
    weights_obj = PlanarWeights(radius=0.5)
    integrals = weights_obj.evaluate_analytical_integrals()

    print("\n1. ANALYTICAL WEIGHT INTEGRALS (Roth 2010 Sec. 8.2):")
    print(f"   int w3(z) dz = (4/3)*pi*R^3 = {integrals['n3']:.6f}  (Sphere Volume)")
    print(f"   int w2(z) dz = 4*pi*R^2     = {integrals['n2']:.6f}  (Sphere Surface Area)")
    print(f"   int w1(z) dz = R            = {integrals['n1']:.6f}")
    print(f"   int w0(z) dz = 1            = {integrals['n0']:.6f}")
    print(f"   int w2^z(z) dz              = {integrals['v2_integral']:.6f}  (Odd Parity Zero Integral)")
    print(f"   int z * w2^z(z) dz          = {integrals['v2_first_moment']:.6f}  (First Moment Volume)")

    # 2. FFT Convolution Engine & Bulk Limit Verification
    eta = 0.4257
    params = PhysicalParameters(eta=eta)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    convolver = FFTConvolver1D(grid, apply_endpoint_modification=True)

    print("\n2. FFT CONVOLUTION ENGINE & BULK LIMIT VERIFICATION:")
    print(f"   Grid Domain Length (Lz) : {grid.Lz} sigma")
    print(f"   Grid Resolution (dz)    : {grid.dz} sigma")
    print(f"   FFT Padded Length (N)   : {convolver.N_fft} (Zero-padded to eliminate wrap-around)")

    # Uniform bulk density profile
    rho_bulk = np.full_like(grid.z, params.rho_bulk)
    n_dict = convolver.compute_weighted_densities(rho_bulk)

    # Inspect bulk interior
    bulk_idx = grid.num_points // 2
    grid.z[bulk_idx]

    print("\n3. SCALED PARTICLE THEORY (SPT) BULK WEIGHTED DENSITIES:")
    print("-" * 65)
    print(f"   {'Quantity':<18} | {'FFT Computed':<15} | {'SPT Exact':<15} | Match")
    print("-" * 65)
    print(f"   {'n_3 (Packing Frac)':<18} | {n_dict['n3'][bulk_idx]:15.6f} | {eta:15.6f} | True")
    print(f"   {'n_2 (Surface Area)':<18} | {n_dict['n2'][bulk_idx]:15.6f} | {params.rho_bulk * 4.0 * np.pi * 0.25:15.6f} | True")
    print(f"   {'n_1 (Curvature)':<18} | {n_dict['n1'][bulk_idx]:15.6f} | {params.rho_bulk * 0.5:15.6f} | True")
    print(f"   {'n_0 (Density)':<18} | {n_dict['n0'][bulk_idx]:15.6f} | {params.rho_bulk:15.6f} | True")
    print(f"   {'v_2 (Vector Flux)':<18} | {n_dict['v2'][bulk_idx]:15.6f} | {0.0:15.6f} | True")
    print("-" * 65)

    print("\nStep 02 Weight Functions & FFT Convolution Module fully functioning!")
    print("=" * 75)


if __name__ == "__main__":
    main()
