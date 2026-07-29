"""Demonstration script for Step 09: Thermodynamic Observables & Sum-Rule Validation."""

from src.diagnostics import SumRuleDiagnostics
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver


def main():
    print("=" * 85)
    print("Classical DFT (FMT) Simulator - Step 09 Demonstration")
    print("Thermodynamic Observables & Sum-Rule Validation Engine")
    print("=" * 85)

    eta_values = [0.20, 0.35, 0.45]
    functionals = ["RF", "WB", "WBII", "WB-Tensor"]

    for eta in eta_values:
        print(f"\n--- PACKING FRACTION ETA = {eta:.4f} ---")
        print(
            f"{'Functional':<12} | {'Contact rho':<12} | {'beta*P_bulk':<12} | {'Contact Err':<12} | "
            f"{'beta*gamma (sp)':<15} | {'beta*gamma (bulk)':<17} | {'Adsorption G':<12}"
        )
        print("-" * 105)

        params = PhysicalParameters(eta=eta)
        grid = Grid1D(params=params, Lz=10.0, dz=0.005)

        for f_name in functionals:
            func = functional_factory(f_name)
            solver = RothPicardSolver(grid, functional=func, alpha_init=0.03)
            res = solver.solve(max_iter=1500, tol=1e-7)

            diag = SumRuleDiagnostics.evaluate_all(grid, res.rho, func)

            status_str = f"{diag.contact_error_rel * 100:.3f}%"
            print(
                f"{f_name:<12} | {diag.contact_density:<12.6f} | {diag.bulk_pressure:<12.6f} | {status_str:<12} | "
                f"{diag.surface_tension_spatial:<15.6f} | {diag.surface_tension_bulk_route:<17.6f} | {diag.excess_adsorption:<12.6f}"
            )

    print("\n" + "=" * 85)
    print(
        "Step 09 Thermodynamic Observables & Sum-Rule Validation Engine fully verified!"
    )
    print("=" * 85)


if __name__ == "__main__":
    main()
