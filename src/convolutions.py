"""FFT-based 1D convolution engine for weighted densities and direct correlation functions."""

from typing import Dict, Optional

import numpy as np
import scipy.fft as fft

from src.grid import Grid1D
from src.weights import PlanarWeights


class FFTConvolver1D:
    """Zero-padded FFT convolution engine for planar FMT density functional theory.

    Args:
        grid: Grid1D spatial domain discretization instance.
        apply_endpoint_modification: Apply Section 8.4 endpoint weight modifications.
    """

    def __init__(self, grid: Grid1D, apply_endpoint_modification: bool = True) -> None:
        self.grid = grid
        self.apply_endpoint_modification = apply_endpoint_modification

        self.weights_obj = PlanarWeights(radius=grid.params.radius)
        self.z_w, self.weights_dict = self.weights_obj.get_grid_and_weights(
            dz=grid.dz, apply_endpoint_modification=apply_endpoint_modification
        )

        self.N_grid = len(grid.z)
        self.N_w = len(self.z_w)
        self.center_idx = self.N_w // 2  # Index corresponding to z = 0 in z_w

        # Compute optimal zero-padded length for FFT convolution (prevent periodic wrap-around)
        min_len = self.N_grid + self.N_w - 1
        self.N_fft = int(fft.next_fast_len(min_len))

        # Pre-compute FFT transforms of padded weight functions
        self.fft_weights: Dict[str, np.ndarray] = {}
        for key, w_arr in self.weights_dict.items():
            # Pad weight array such that origin z=0 is at index 0 of padded array
            padded_w = np.zeros(self.N_fft, dtype=float)
            padded_w[: self.N_w] = np.roll(w_arr, -self.center_idx)
            self.fft_weights[key] = fft.fft(padded_w, n=self.N_fft)

    def _convolve_raw(self, f: np.ndarray, key: str, parity_flip: bool = False) -> np.ndarray:
        """Core zero-padded FFT spatial convolution (f * w)(z) = integral f(z') w(z - z') dz'."""
        padded_f = np.zeros(self.N_fft, dtype=float)
        padded_f[: self.N_grid] = f

        fft_f = fft.fft(padded_f, n=self.N_fft)
        fft_w = self.fft_weights[key]

        # Invert sign if parity flip requested (for odd vector weights during c^(1) derivatives)
        if parity_flip:
            fft_w = -fft_w

        conv_full = np.real(fft.ifft(fft_f * fft_w)) * self.grid.dz
        # Extract slice corresponding to spatial grid z
        return conv_full[: self.N_grid]

    def compute_weighted_densities(self, rho: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all 6 spatial weighted densities n_0(z), n_1(z), n_2(z), n_3(z), v_1(z), v_2(z).

        Args:
            rho: 1D spatial density profile array matching grid.z.

        Returns:
            Dictionary mapping weight key to 1D weighted density array.
        """
        result: Dict[str, np.ndarray] = {}
        for key in self.weights_dict:
            result[key] = self._convolve_raw(rho, key, parity_flip=False)
        return result

    def compute_direct_correlation_convolutions(
        self, df_dn_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute convolution integrals integral (dPhi/dn_alpha)(z') w_alpha(z' - z) dz'.

        Applies odd parity sign flip for vector weight components (v1, v2).

        Args:
            df_dn_dict: Dictionary mapping weight key to (dPhi/dn_alpha) profile array.

        Returns:
            Dictionary mapping weight key to convolution integral array.
        """
        result: Dict[str, np.ndarray] = {}
        for key, df_dn in df_dn_dict.items():
            is_vector = PlanarWeights.PARITY_IS_VECTOR.get(key, False)
            result[key] = self._convolve_raw(df_dn, key, parity_flip=is_vector)
        return result
