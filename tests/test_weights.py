"""Unit tests for 1D planar weight functions (Step 02)."""

import math

import numpy as np
import pytest

from src.weights import PlanarWeights


def test_analytical_integrals_defaults():
    """Verify analytical integrals of PlanarWeights against exact physical formulas."""
    weights = PlanarWeights(radius=0.5)
    integrals = weights.evaluate_analytical_integrals()

    # Sphere volume V = (4/3)*pi*R^3 = (pi/6) for R=0.5
    expected_v = (4.0 / 3.0) * math.pi * (0.5**3)
    expected_s = 4.0 * math.pi * (0.5**2)

    assert pytest.approx(integrals["n3"]) == expected_v
    assert pytest.approx(integrals["n2"]) == expected_s
    assert pytest.approx(integrals["n1"]) == 0.5
    assert pytest.approx(integrals["n0"]) == 1.0
    assert pytest.approx(integrals["v2_integral"]) == 0.0
    assert pytest.approx(integrals["v2_first_moment"]) == expected_v


def test_weight_discretization_and_parity():
    """Verify weight function array symmetry (even for scalars, odd for vectors)."""
    weights = PlanarWeights(radius=0.5)
    z_w, w_dict = weights.get_grid_and_weights(dz=0.001, apply_endpoint_modification=False)

    # Check symmetry across origin z = 0
    center_idx = len(z_w) // 2
    assert pytest.approx(z_w[center_idx]) == 0.0

    for key in ["n0", "n1", "n2", "n3"]:
        w_arr = w_dict[key]
        # Scalar weights must be strictly even: w(-z) = w(z)
        assert np.allclose(w_arr, np.flip(w_arr))

    for key in ["v1", "v2"]:
        w_arr = w_dict[key]
        # Vector weights must be strictly odd: w(-z) = -w(z)
        assert np.allclose(w_arr, -np.flip(w_arr))
        # Center value at z=0 must be 0
        assert pytest.approx(w_arr[center_idx]) == 0.0


def test_numerical_quadrature_integrals():
    """Verify discrete numerical integrals of weights match analytical values."""
    weights = PlanarWeights(radius=0.5)
    dz = 0.0005
    z_w, w_dict = weights.get_grid_and_weights(dz=dz, apply_endpoint_modification=True)

    # Trapezoidal integration using dz
    int_n3 = float(np.sum(w_dict["n3"]) * dz)
    int_n2 = float(np.sum(w_dict["n2"]) * dz)
    int_n1 = float(np.sum(w_dict["n1"]) * dz)
    int_n0 = float(np.sum(w_dict["n0"]) * dz)
    int_v2_moment = float(np.sum(z_w * w_dict["v2"]) * dz)

    analytical = weights.evaluate_analytical_integrals()

    assert pytest.approx(int_n3, rel=1e-4) == analytical["n3"]
    assert pytest.approx(int_n2, rel=1e-4) == analytical["n2"]
    assert pytest.approx(int_n1, rel=1e-4) == analytical["n1"]
    assert pytest.approx(int_n0, rel=1e-4) == analytical["n0"]
    assert pytest.approx(int_v2_moment, rel=1e-4) == analytical["v2_first_moment"]
