"""Unit tests for FFTConvolver1D (Step 02)."""

import math
import numpy as np
import pytest

from src.convolutions import FFTConvolver1D
from src.grid import Grid1D, PhysicalParameters


def test_bulk_limit_weighted_densities():
    """Verify that a uniform bulk density profile produces exact Scaled Particle Theory (SPT) bulk values.

    Bulk values (Roth Sec. 3):
    - n_3 -> eta
    - n_2 -> rho_bulk * 4*pi*R^2
    - n_1 -> rho_bulk * R
    - n_0 -> rho_bulk
    - v_1, v_2 -> 0
    """
    eta_bulk = 0.4257
    params = PhysicalParameters(eta=eta_bulk)
    # Use domain well away from boundaries to inspect bulk interior
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)

    convolver = FFTConvolver1D(grid, apply_endpoint_modification=True)

    # Uniform bulk density profile across domain
    rho_uniform = np.full_like(grid.z, params.rho_bulk)
    n_dict = convolver.compute_weighted_densities(rho_uniform)

    # Inspect bulk interior region z in [2*R, Lz - 2*R] to avoid boundary cutoff
    R = params.radius
    bulk_mask = (grid.z >= 2.0 * R) & (grid.z <= (grid.Lz - 2.0 * R))

    expected_n3 = eta_bulk
    expected_n2 = params.rho_bulk * 4.0 * math.pi * (R**2)
    expected_n1 = params.rho_bulk * R
    expected_n0 = params.rho_bulk

    assert np.allclose(n_dict["n3"][bulk_mask], expected_n3, rtol=1e-3)
    assert np.allclose(n_dict["n2"][bulk_mask], expected_n2, rtol=1e-3)
    assert np.allclose(n_dict["n1"][bulk_mask], expected_n1, rtol=1e-3)
    assert np.allclose(n_dict["n0"][bulk_mask], expected_n0, rtol=1e-3)

    # Vector components must vanish in uniform bulk
    assert np.allclose(n_dict["v1"][bulk_mask], 0.0, atol=1e-4)
    assert np.allclose(n_dict["v2"][bulk_mask], 0.0, atol=1e-4)


def test_vector_parity_sign_inversion():
    """Verify parity sign inversion in direct correlation derivative convolutions."""
    grid = Grid1D(Lz=5.0, dz=0.005)
    convolver = FFTConvolver1D(grid)

    # Asymmetric test profile
    df_dn = np.zeros_like(grid.z)
    df_dn[grid.z >= 2.0] = 1.0

    df_dict = {"v2": df_dn, "n2": df_dn}
    c1_conv = convolver.compute_direct_correlation_convolutions(df_dict)

    # Scalar component convolution has no parity flip
    n2_conv_normal = convolver._convolve_raw(df_dn, "n2", parity_flip=False)
    assert np.allclose(c1_conv["n2"], n2_conv_normal)

    # Vector component convolution MUST have parity sign flip applied
    v2_conv_normal = convolver._convolve_raw(df_dn, "v2", parity_flip=False)
    assert np.allclose(c1_conv["v2"], -v2_conv_normal)


def test_zero_padding_no_boundary_wraparound():
    """Verify that zero padding prevents circular wrap-around across domain boundaries."""
    grid = Grid1D(Lz=5.0, dz=0.005)
    convolver = FFTConvolver1D(grid)

    # Localized delta-like density peak at z = 2.5
    rho_spike = np.zeros_like(grid.z)
    idx_mid = len(grid.z) // 2
    rho_spike[idx_mid] = 1.0 / grid.dz

    n_dict = convolver.compute_weighted_densities(rho_spike)

    # Check boundaries z=0 and z=Lz: zero-padding must ensure n_alpha is strictly 0 at far edges
    assert pytest.approx(n_dict["n3"][0]) == 0.0
    assert pytest.approx(n_dict["n3"][-1]) == 0.0
