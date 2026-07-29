"""Unit tests for PhysicalParameters and Grid1D discretization (Step 01)."""

import math

import numpy as np
import pytest

from src.grid import Grid1D, PhysicalParameters


def test_physical_parameters_defaults():
    """Verify default physical parameters and derived geometric properties."""
    params = PhysicalParameters()
    assert params.sigma == 1.0
    assert params.radius == 0.5
    assert params.beta == 1.0
    assert params.eta == 0.4257

    # Sphere volume V = (pi/6) * sigma^3
    expected_volume = (math.pi / 6.0) * (1.0**3)
    assert pytest.approx(params.volume) == expected_volume

    # Bulk number density for eta = 0.4257 (Roth 2010 Fig 1a benchmark)
    expected_rho_bulk = 0.4257 / expected_volume
    assert pytest.approx(params.rho_bulk) == expected_rho_bulk
    assert pytest.approx(params.rho_bulk, abs=1e-4) == 0.8130


def test_roth_2010_benchmark_packing_fractions():
    """Verify density conversions for benchmark packing fractions in Roth (2010)."""
    # Fig 1a: eta = 0.4257
    params_a = PhysicalParameters(eta=0.4257)
    assert pytest.approx(params_a.rho_bulk, abs=1e-4) == 0.8130

    # Fig 1b: eta = 0.4783
    params_b = PhysicalParameters(eta=0.4783)
    assert pytest.approx(params_b.rho_bulk, abs=1e-4) == 0.9135


def test_eta_rho_bulk_roundtrip():
    """Test static conversion methods between packing fraction eta and bulk density rho_bulk."""
    for eta_test in [0.1, 0.25, 0.4257, 0.4783, 0.5]:
        rho_b = PhysicalParameters.rho_bulk_from_eta(eta_test, sigma=1.0)
        eta_back = PhysicalParameters.eta_from_rho_bulk(rho_b, sigma=1.0)
        assert pytest.approx(eta_back) == eta_test


def test_grid_discretization_and_resolution():
    """Test 1D grid setup and verify spacing constraints."""
    grid = Grid1D(Lz=10.0, dz=0.005)
    assert grid.Lz == 10.0
    assert grid.dz <= 0.01
    assert grid.z[0] == 0.0
    assert pytest.approx(grid.z[-1]) == 10.0
    assert len(grid.z) == 2001

    # Exception raised if dz > 0.01 * sigma
    with pytest.raises(ValueError, match="exceeds maximum allowable spacing"):
        Grid1D(dz=0.02)


def test_external_potential_single_wall():
    """Test hard wall potential boundaries for a single wall at z = 0."""
    grid = Grid1D(Lz=10.0, dz=0.005)
    v_ext = grid.external_potential(wall_left=0.0, wall_right=None)

    # For z < R (0.5), V_ext = inf
    forbidden_mask = grid.z < 0.5
    assert np.all(np.isinf(v_ext[forbidden_mask]))

    # For z >= R (0.5), V_ext = 0
    accessible_mask = grid.z >= 0.5
    assert np.all(v_ext[accessible_mask] == 0.0)


def test_external_potential_slit_pore():
    """Test hard wall potential boundaries for a slit pore between z = 0 and z = 5.0."""
    grid = Grid1D(Lz=5.0, dz=0.005)
    v_ext = grid.external_potential(wall_left=0.0, wall_right=5.0)

    R = 0.5
    # Forbidden near left wall (z < 0.5) and right wall (z > 4.5)
    forbidden_mask = (grid.z < R) | (grid.z > (5.0 - R))
    accessible_mask = (grid.z >= R) & (grid.z <= (5.0 - R))

    assert np.all(np.isinf(v_ext[forbidden_mask]))
    assert np.all(v_ext[accessible_mask] == 0.0)


def test_initial_density_profile():
    """Verify initial density profile mask and zeroing in forbidden region."""
    params = PhysicalParameters(eta=0.4257)
    grid = Grid1D(params=params, Lz=10.0, dz=0.005)
    rho_init = grid.initial_density_profile(wall_left=0.0, wall_right=10.0)

    R = params.radius  # 0.5
    # Forbidden region: density must be strictly 0.0
    forbidden_indices = grid.z < R
    assert np.all(rho_init[forbidden_indices] == 0.0)

    # Accessible region: density must be equal to rho_bulk
    accessible_indices = (grid.z >= R) & (grid.z <= 10.0 - R)
    assert np.allclose(rho_init[accessible_indices], params.rho_bulk)


def test_grid_integration():
    """Test spatial numerical integration over the 1D grid."""
    grid = Grid1D(Lz=10.0, dz=0.005)

    # Integral of constant 1.0 over [0, Lz] is Lz
    f_const = np.ones_like(grid.z)
    integral_const = grid.integrate(f_const)
    assert pytest.approx(integral_const) == 10.0

    # Integral of sin(z) over [0, pi] is 2.0
    grid_pi = Grid1D(Lz=math.pi, dz=0.001)
    f_sin = np.sin(grid_pi.z)
    integral_sin = grid_pi.integrate(f_sin)
    assert pytest.approx(integral_sin, abs=1e-5) == 2.0
