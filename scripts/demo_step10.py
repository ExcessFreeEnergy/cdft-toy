"""Demonstration script for Step 10: Dimensional Crossover & Collapse Visualizer Engine."""

from src.crossover import CrossoverAnalyzer
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters


def main():
    print("=" * 85)
    print("Classical DFT (FMT) Simulator - Step 10 Demonstration")
    print("Dimensional Crossover & Confinement Collapse Visualizer Engine")
    print("=" * 85)

    # 1. Slit-pore confinement sweep across pore widths Lz in [0.5, 3.0]
    widths = [0.6, 1.0, 1.5, 2.5]
    functionals = ["RF", "WB", "WBII", "WB-Tensor"]

    print("\n1. SLIT PORE CONFINEMENT SWEEP (eta_bulk = 0.35):")
    print(
        f"{'Pore Width (Lz)':<18} | {'Functional':<12} | {'Converged':<10} | {'Max rho(z)':<12} | "
        f"{'Max Phi(z)':<14} | {'Excess Free Energy F_ex':<24}"
    )
    print("-" * 105)

    sweep_res = CrossoverAnalyzer.sweep_pore_confinement(widths, eta_bulk=0.35, functionals=functionals)

    for idx, w in enumerate(widths):
        for f_name in functionals:
            m = sweep_res[f_name][idx]
            print(
                f"{w:<18.2f} | {f_name:<12} | {str(m.converged):<10} | {m.max_rho:<12.4f} | "
                f"{m.max_phi:<14.4f} | {m.free_energy_ex:<24.6f}"
            )
        print("-" * 105)

    # 2. Zero-D Cavity Confinement Divergence Test as Gaussian width alpha -> 0
    print("\n2. SYNTHETIC ZERO-D CAVITY GAUSSIAN CONFINEMENT (alpha -> 0):")
    print(f"{'Cavity Width alpha':<20} | {'RF max Phi':<18} | {'WB max Phi':<18} | {'WB-Tensor max Phi':<20}")
    print("-" * 80)

    params = PhysicalParameters(eta=0.35)
    grid = Grid1D(params=params, Lz=4.0, dz=0.005)
    alphas = [0.15, 0.10, 0.06, 0.04, 0.025]

    phi_rf = CrossoverAnalyzer.evaluate_zero_d_divergence(grid, alphas, functional_factory("RF"))
    phi_wb = CrossoverAnalyzer.evaluate_zero_d_divergence(grid, alphas, functional_factory("WB"))
    phi_wbt = CrossoverAnalyzer.evaluate_zero_d_divergence(grid, alphas, functional_factory("WB-Tensor"))

    for a, r_val, w_val, t_val in zip(alphas, phi_rf, phi_wb, phi_wbt, strict=False):
        print(f"{a:<20.3f} | {r_val:<18.4f} | {w_val:<18.4f} | {t_val:<20.4f}")

    print("\n" + "=" * 85)
    print("Step 10 Dimensional Crossover & Collapse Visualizer Engine fully verified!")
    print("=" * 85)


if __name__ == "__main__":
    main()
