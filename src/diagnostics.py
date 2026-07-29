"""Thermodynamic observables and sum-rule validation routines (Roth 2010 Sec. 5.1).

Computes wall surface tension, contact density extrapolation, bulk-route surface tension,
excess adsorption, and verifies contact theorem and Gibbs adsorption sum-rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.functionals.base import FMTFunctional
from src.grid import Grid1D
from src.weighted_densities import WeightedDensityCalculator


@dataclass
class SumRuleResult:
    """Container for sum-rule accuracy metrics and thermodynamic observables."""

    contact_density: float
    bulk_pressure: float
    contact_error_rel: float
    contact_passed: bool
    surface_tension_spatial: float
    surface_tension_bulk_route: float
    excess_adsorption: float
    gibbs_dgamma_dmu: float | None = None
    gibbs_error_rel: float | None = None


class SumRuleDiagnostics:
    """Thermodynamic observables and sum-rule validation engine (Roth Sec. 5.1)."""

    @staticmethod
    def extrapolate_contact_density(grid: Grid1D, rho: np.ndarray, wall_position: float = 0.0) -> float:
        """Extrapolate density profile to wall contact z = wall_position + R using 3-point fit.

        Args:
            grid: Grid1D spatial domain discretization.
            rho: Density profile array matching grid.z.
            wall_position: Position of hard wall (default 0.0).

        Returns:
            Extrapolated contact density rho(R+).
        """
        R = grid.params.radius
        z_contact = wall_position + R

        # Find first accessible grid index z >= z_contact
        idx_c = int(np.searchsorted(grid.z, z_contact - 1e-12))
        if idx_c >= len(grid.z) - 2:
            return float(rho[idx_c])

        z0, z1, z2 = grid.z[idx_c], grid.z[idx_c + 1], grid.z[idx_c + 2]
        y0, y1, y2 = rho[idx_c], rho[idx_c + 1], rho[idx_c + 2]

        # Lagrange 2nd-degree polynomial interpolation/extrapolation to z_contact
        L0 = ((z_contact - z1) * (z_contact - z2)) / ((z0 - z1) * (z0 - z2))
        L1 = ((z_contact - z0) * (z_contact - z2)) / ((z1 - z0) * (z1 - z2))
        L2 = ((z_contact - z0) * (z_contact - z1)) / ((z2 - z0) * (z2 - z1))

        rho_contact = y0 * L0 + y1 * L1 + y2 * L2
        return float(max(0.0, rho_contact))

    @classmethod
    def verify_contact_theorem(
        self,
        grid: Grid1D,
        rho: np.ndarray,
        functional: FMTFunctional,
        tol_rel: float = 0.03,
    ) -> tuple[float, float, float, bool]:
        """Verify contact theorem sum-rule: rho(z = R+) == beta * P_bulk (Roth Eq. 26).

        Args:
            grid: Grid1D spatial domain discretization.
            rho: Density profile array matching grid.z.
            functional: FMT functional instance.
            tol_rel: Relative tolerance threshold for pass (default 0.03 = 3.0%).

        Returns:
            Tuple of (rho_contact, p_bulk, rel_error, passed_boolean).
        """
        rho_contact = self.extrapolate_contact_density(grid, rho)
        p_bulk = functional.compute_bulk_pressure(grid.params.eta, sigma=grid.params.sigma)
        err_rel = float(abs(rho_contact - p_bulk) / p_bulk)
        passed = err_rel <= tol_rel
        return rho_contact, p_bulk, err_rel, passed

    @staticmethod
    def compute_bulk_route_surface_tension(functional: FMTFunctional, eta: float, sigma: float = 1.0) -> float:
        """Compute analytical bulk-route wall surface tension beta * gamma_bulk = (dPhi/dn2)_bulk (Roth Eq. 447).

        Args:
            functional: FMT functional instance.
            eta: Bulk packing fraction.
            sigma: Sphere diameter.

        Returns:
            Dimensionless bulk-route surface tension beta * gamma_bulk.
        """
        R = 0.5 * sigma
        rho_bulk = eta / ((math.pi / 6.0) * (sigma**3))
        n0_b = rho_bulk
        n1_b = rho_bulk * R
        n2_b = rho_bulk * 4.0 * math.pi * (R**2)
        n3_b = eta

        from src.weighted_densities import WeightedDensities

        wd_bulk = WeightedDensities(
            n0=np.array([n0_b]),
            n1=np.array([n1_b]),
            n2=np.array([n2_b]),
            n3=np.array([n3_b]),
            v1=np.array([0.0]),
            v2=np.array([0.0]),
            n_m2=np.array([0.0]),
        )
        der_bulk = functional.evaluate_derivatives(wd_bulk)
        return float(der_bulk["n2"][0])

    @classmethod
    def compute_surface_tension(
        self,
        grid: Grid1D,
        rho: np.ndarray,
        functional: FMTFunctional,
        calc: WeightedDensityCalculator | None = None,
    ) -> float:
        """Compute wall surface tension beta * gamma via spatial grand-potential integration (Roth Eq. 28).

        Formula (for volume starting at z=0):
            beta * gamma = integral_0^Lz [ omega(z) + beta * P_bulk ] dz
        where:
            omega(z) = Phi(z) + rho(z) * [ ln(rho(z)/rho_bulk) - 1 - beta*mu_ex ]  (for z >= R)
            omega(z) = Phi(z)                                                    (for z < R)

        Args:
            grid: Grid1D spatial domain discretization.
            rho: Density profile array matching grid.z.
            functional: FMT functional instance.
            calc: Optional WeightedDensityCalculator instance.

        Returns:
            Dimensionless wall surface tension beta * gamma.
        """
        if calc is None:
            calc = WeightedDensityCalculator(grid, apply_endpoint_modification=True)

        wd = calc.compute(rho)
        phi = functional.evaluate_phi(wd)

        eta = grid.params.eta
        rho_bulk = grid.params.rho_bulk
        sigma = grid.params.sigma
        p_bulk = functional.compute_bulk_pressure(eta, sigma=sigma)
        mu_ex = functional.bulk_excess_mu(eta, sigma=sigma)

        R = grid.params.radius
        accessible = grid.z >= (R - 1e-12)

        omega = phi.copy()
        rho_acc = rho[accessible]
        rho_safe = np.maximum(1e-15, rho_acc)
        ideal_chem = rho_acc * (np.log(rho_safe / rho_bulk) - 1.0 - mu_ex)
        omega[accessible] += ideal_chem

        integrand = omega + p_bulk
        gamma = grid.integrate(integrand)
        return float(gamma)

    @staticmethod
    def compute_excess_adsorption(grid: Grid1D, rho: np.ndarray) -> float:
        """Compute excess adsorption Gamma = integral_0^Lz (rho(z) - rho_bulk) dz (Roth Eq. 27).

        Args:
            grid: Grid1D spatial domain discretization.
            rho: Density profile array matching grid.z.

        Returns:
            Dimensionless excess adsorption Gamma.
        """
        rho_bulk = grid.params.rho_bulk
        diff = rho - rho_bulk
        return float(grid.integrate(diff))

    @classmethod
    def verify_gibbs_adsorption(
        self,
        grid: Grid1D,
        functional: FMTFunctional,
        eta: float,
        delta_eta: float = 0.005,
    ) -> tuple[float, float, float]:
        """Verify Gibbs adsorption theorem: Gamma == -d(gamma)/d(mu) (Roth Eq. 27).

        Args:
            grid: Grid1D spatial domain discretization.
            functional: FMT functional instance.
            eta: Central bulk packing fraction.
            delta_eta: Packing fraction step size for finite-difference d(gamma)/d(mu).

        Returns:
            Tuple of (-dgamma/dmu, excess_adsorption_Gamma, relative_error).
        """
        from src.solvers import RothPicardSolver

        def _solve_and_get_gamma_mu(
            e: float,
        ) -> tuple[float, float, float, Grid1D, np.ndarray]:
            p = grid.params.__class__(eta=e)
            g = grid.__class__(params=p, Lz=grid.Lz, dz=grid.dz)
            solver = RothPicardSolver(g, functional=functional, alpha_init=0.03)
            res = solver.solve(max_iter=3000, tol=1e-7)
            c = WeightedDensityCalculator(g, apply_endpoint_modification=True)

            gamma = self.compute_surface_tension(g, res.rho, functional, calc=c)
            mu_tot = math.log(p.rho_bulk) + functional.bulk_excess_mu(e, sigma=p.sigma)
            gamma_ex = self.compute_excess_adsorption(g, res.rho)
            return gamma, mu_tot, gamma_ex, g, res.rho

        _gamma_mid, _mu_mid, gamma_ex_mid, _, _ = _solve_and_get_gamma_mu(eta)
        gamma_p, mu_p, _, _, _ = _solve_and_get_gamma_mu(eta + delta_eta)
        gamma_m, mu_m, _, _, _ = _solve_and_get_gamma_mu(eta - delta_eta)

        minus_dgamma_dmu = -(gamma_p - gamma_m) / (mu_p - mu_m)
        err_rel = float(abs(minus_dgamma_dmu - gamma_ex_mid) / max(1e-10, abs(gamma_ex_mid)))
        return minus_dgamma_dmu, gamma_ex_mid, err_rel

    @classmethod
    def evaluate_all(
        self,
        grid: Grid1D,
        rho: np.ndarray,
        functional: FMTFunctional,
        calc: WeightedDensityCalculator | None = None,
    ) -> SumRuleResult:
        """Execute full suite of thermodynamic sum-rule diagnostics.

        Args:
            grid: Grid1D spatial domain discretization.
            rho: Density profile array matching grid.z.
            functional: FMT functional instance.
            calc: Optional WeightedDensityCalculator instance.

        Returns:
            SumRuleResult dataclass containing all metrics.
        """
        rho_c, p_bulk, err_c, passed_c = self.verify_contact_theorem(grid, rho, functional)
        gamma_spatial = self.compute_surface_tension(grid, rho, functional, calc=calc)
        gamma_bulk = self.compute_bulk_route_surface_tension(functional, grid.params.eta, sigma=grid.params.sigma)
        gamma_excess = self.compute_excess_adsorption(grid, rho)

        return SumRuleResult(
            contact_density=rho_c,
            bulk_pressure=p_bulk,
            contact_error_rel=err_c,
            contact_passed=passed_c,
            surface_tension_spatial=gamma_spatial,
            surface_tension_bulk_route=gamma_bulk,
            excess_adsorption=gamma_excess,
        )
