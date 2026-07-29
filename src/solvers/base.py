"""Abstract base class interface and result dataclass for cDFT solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SolverResult:
    """Container storing cDFT solver convergence status and results.

    Attributes:
        converged: True if residual norm dropped below tolerance.
        iterations: Number of iterations executed.
        residual: Final L2 residual norm.
        rho: Final equilibrium spatial density profile array.
        c1: One-body direct correlation function array c^(1)(z).
        c1_bulk: Constant bulk direct correlation value.
        history_residual: List of residual norms across iteration history.
    """

    converged: bool
    iterations: int
    residual: float
    rho: np.ndarray
    c1: np.ndarray
    c1_bulk: float
    history_residual: list[float]


class DFTSolver(ABC):
    """Abstract base class for Classical Density Functional Theory solvers."""

    @abstractmethod
    def compute_c1(self, rho: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute spatial one-body direct correlation function c^(1)(z) and bulk correlation c1_bulk.

        Args:
            rho: 1D spatial density profile array.

        Returns:
            Tuple of (c1(z) array, c1_bulk scalar).
        """

    @abstractmethod
    def solve_step(self, rho: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Execute a single Picard iteration step.

        Args:
            rho: Current density profile array.
            alpha: Mixing parameter in (0, 1].

        Returns:
            Tuple of (updated rho array, c1(z) array, residual norm).
        """

    @abstractmethod
    def solve(
        self,
        rho_init: np.ndarray,
        max_iter: int = 2000,
        tol: float = 1e-7,
        alpha: float = 0.02,
    ) -> SolverResult:
        """Execute iterative relaxation solver until convergence or max_iter.

        Args:
            rho_init: Initial density profile guess.
            max_iter: Maximum number of iterations.
            tol: Residual norm tolerance threshold.
            alpha: Picard mixing parameter.

        Returns:
            SolverResult container.
        """
