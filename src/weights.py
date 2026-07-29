"""1D Planar geometrical weight functions for FMT classical density functional theory."""

import math
from typing import Dict, Tuple

import numpy as np


class PlanarWeights:
    """1D Planar fundamental measure theory (FMT) weight functions (Roth 2010 Sec. 8.2 & 8.4).

    Attributes:
        radius: Sphere radius R = sigma / 2.
    """

    # Parity dictionary: True for odd vector components, False for even scalar components
    PARITY_IS_VECTOR: Dict[str, bool] = {
        "n0": False,
        "n1": False,
        "n2": False,
        "n3": False,
        "v1": True,
        "v2": True,
        "n_m2": False,
    }

    def __init__(self, radius: float = 0.5) -> None:
        self.radius = float(radius)

    def evaluate_analytical_integrals(self) -> Dict[str, float]:
        """Return exact analytical 1D integrals over [-R, R] for verification.

        - int_{-R}^R w_3(z) dz = (4/3)*pi*R^3 = V_sphere
        - int_{-R}^R w_2(z) dz = 4*pi*R^2 = S_sphere
        - int_{-R}^R w_1(z) dz = R
        - int_{-R}^R w_0(z) dz = 1
        - int_{-R}^R w_2^z(z) dz = 0
        - int_{-R}^R z * w_2^z(z) dz = (4/3)*pi*R^3 = V_sphere
        - int_{-R}^R w_{m2}(z) dz = 0
        """
        R = self.radius
        v_sphere = (4.0 / 3.0) * math.pi * (R**3)
        s_sphere = 4.0 * math.pi * (R**2)

        return {
            "n3": v_sphere,
            "n2": s_sphere,
            "n1": R,
            "n0": 1.0,
            "v2_integral": 0.0,
            "v2_first_moment": v_sphere,
            "n_m2_integral": 0.0,
        }

    def get_grid_and_weights(
        self, dz: float, apply_endpoint_modification: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Evaluate weight function arrays on 1D grid z_w in [-R, R].

        Args:
            dz: Spatial grid resolution.
            apply_endpoint_modification: Apply Section 8.4 endpoint modifications
                (3/8, 7/6, 23/24 factors) for O(dz^4) Fourier space Simpson accuracy.

        Returns:
            Tuple of (z_w grid array, dictionary of weight arrays).
        """
        R = self.radius
        M = int(round(R / dz))
        z_w = np.linspace(-M * dz, M * dz, 2 * M + 1)

        abs_z = np.abs(z_w)
        mask = abs_z <= (R + 1e-12)

        # 1. Scalar weight functions (Roth Sec. 8.2 & 8.4)
        w3 = np.zeros_like(z_w, dtype=float)
        w2 = np.zeros_like(z_w, dtype=float)

        w3[mask] = math.pi * (R**2 - z_w[mask] ** 2)
        w2[mask] = 2.0 * math.pi * R

        w1 = w2 / (4.0 * math.pi * R)
        w0 = w2 / (4.0 * math.pi * (R**2))

        # 2. Vector weight functions (Roth Sec. 8.2)
        v2 = np.zeros_like(z_w, dtype=float)
        v2[mask] = 2.0 * math.pi * z_w[mask]
        v1 = v2 / (4.0 * math.pi * R)

        # 3. Tensorial scalar component weight function w_m2 (Roth Sec. 4.4 Eq. 19)
        wm2 = np.zeros_like(z_w, dtype=float)
        wm2[mask] = 2.0 * math.pi * R * (((z_w[mask] ** 2) / (R**2)) - (1.0 / 3.0))

        weights = {"n0": w0, "n1": w1, "n2": w2, "n3": w3, "v1": v1, "v2": v2, "n_m2": wm2}

        # 3. Section 8.4 Endpoint Weight Modifications (High-Order Simpson Quadrature)
        if apply_endpoint_modification and M >= 3:
            for key in weights:
                w_arr = weights[key]
                # Boundary endpoints at x = +-R (index 0 and -1)
                w_arr[0] *= 3.0 / 8.0
                w_arr[-1] *= 3.0 / 8.0

                # Index 1 and -2 (x = +-(R - dz))
                w_arr[1] *= 7.0 / 6.0
                w_arr[-2] *= 7.0 / 6.0

                # Index 2 and -3 (x = +-(R - 2dz))
                w_arr[2] *= 23.0 / 24.0
                w_arr[-3] *= 23.0 / 24.0

        return z_w, weights
