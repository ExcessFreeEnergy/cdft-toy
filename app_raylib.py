"""Raylib Desktop GUI Application for Classical Density Functional Theory (FMT) Hard-Sphere Simulator."""

import math
import numpy as np
import pyray as pr

from src.functionals import RosenfeldFunctional
from src.grid import Grid1D, PhysicalParameters
from src.solvers import FixedPicardSolver
from src.ui.plotter import Plotter2D
from src.ui.theme import Theme
from src.ui.widgets import UIWidgets
from src.weighted_densities import WeightedDensityCalculator


class RaylibCDFTApp:
    """Raylib interactive application engine for solving arbitrary cDFT FMT problems."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

        # State parameters
        self.eta = 0.4257
        self.Lz = 10.0
        self.dz = 0.005
        self.geom_idx = 0  # 0: Single Planar Wall, 1: Slit Pore
        self.fmt_idx = 0  # 0: RF, 1: WB, 2: WBII, 3: WB-Tensor
        self.view_mode_idx = 0  # 0: Density Profile rho(z), 1: Weighted Densities n_alpha(z), 2: Free Energy Phi(z)

        self.is_solving = False
        self.show_benchmark = True
        self.iteration = 0
        self.residual = 0.0

        # Instantiate functional engine
        self.func = RosenfeldFunctional()

        # Physical system, grid, solver, and weighted density calculator init
        self.rebuild_system()

        # Raylib window initialization
        pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE | pr.FLAG_MSAA_4X_HINT)
        pr.init_window(self.width, self.height, "Classical DFT (FMT) Physics Visualizer & Solver".encode("utf-8"))
        pr.set_target_fps(60)

        # Plotter viewports
        self.plotter_main = Plotter2D(360, 45, self.width - 375, 480, title="Density Profile rho(z)")
        self.plotter_diag = Plotter2D(360, 535, self.width - 375, 175, title="Diagnostic Metrics & Contact Theorem")

    def rebuild_system(self) -> None:
        """Reconstruct PhysicalParameters, Grid1D, FixedPicardSolver, and initial density profile."""
        self.params = PhysicalParameters(eta=self.eta)
        dz_clamped = min(0.010, max(0.001, self.dz))
        self.grid = Grid1D(params=self.params, Lz=self.Lz, dz=dz_clamped)
        self.calc = WeightedDensityCalculator(self.grid, apply_endpoint_modification=True)

        w_left = 0.0
        w_right = self.Lz if self.geom_idx == 1 else None

        # Instantiate physical Picard solver
        self.solver = FixedPicardSolver(
            grid=self.grid,
            functional=self.func,
            alpha=0.03,
            wall_left=w_left,
            wall_right=w_right,
        )

        self.v_ext = self.grid.external_potential(wall_left=w_left, wall_right=w_right)
        self.rho = self.grid.initial_density_profile(wall_left=w_left, wall_right=w_right)
        self.rho_init = self.rho.copy()
        self.iteration = 0
        self.residual = 0.0

        # Compute initial weighted densities, free energy density, c1, and PY pressure
        self.wd = self.calc.compute(self.rho)
        self.n_dict = self.wd.to_dict()
        self.phi = self.func.evaluate_phi(self.wd)
        self.f_ex = self.func.compute_total_free_energy(self.grid, self.wd)
        self.p_py = self.func.compute_bulk_pressure(self.eta, self.params.sigma)

        self.c1, self.c1_bulk = self.solver.compute_c1(self.rho)

    def get_benchmark_points(self):
        """Return benchmark Monte Carlo data points from Roth (2010) Fig. 1 for comparison."""
        if not self.show_benchmark:
            return None

        # Roth 2010 Fig 1a data points (eta = 0.4257)
        if abs(self.eta - 0.4257) < 0.01 and self.geom_idx == 0:
            bm_z = np.array([0.50, 0.55, 0.65, 0.85, 1.05, 1.30, 1.55, 1.80, 2.10, 2.50, 3.00, 4.00, 5.00])
            bm_y = np.array([2.52, 2.20, 1.45, 0.52, 0.48, 0.95, 1.25, 0.98, 0.75, 0.82, 0.81, 0.813, 0.813])
            return bm_z, bm_y

        # Roth 2010 Fig 1b data points (eta = 0.4783)
        if abs(self.eta - 0.4783) < 0.01 and self.geom_idx == 0:
            bm_z = np.array([0.50, 0.55, 0.65, 0.85, 1.05, 1.30, 1.55, 1.80, 2.10, 2.50, 3.00, 4.00])
            bm_y = np.array([3.45, 2.90, 1.70, 0.40, 0.35, 1.15, 1.62, 1.10, 0.70, 0.92, 0.91, 0.913])
            return bm_z, bm_y

        return None

    def run_solver_step(self) -> None:
        """Execute a single physical Picard solver step and update all physical quantities."""
        if not self.is_solving:
            return

        self.iteration += 1
        self.rho, self.c1, self.residual = self.solver.solve_step(self.rho)

        # Recompute spatial weighted densities and free energy density
        self.wd = self.calc.compute(self.rho)
        self.n_dict = self.wd.to_dict()
        self.phi = self.func.evaluate_phi(self.wd)
        self.f_ex = self.func.compute_total_free_energy(self.grid, self.wd)

        if self.residual < 1e-6 or self.iteration >= 500:
            self.is_solving = False

    def render_sidebar(self) -> None:
        """Render controls, sliders, radio buttons, and diagnostic metrics panel."""
        sidebar_w = 340.0
        UIWidgets.draw_panel(10, 45, sidebar_w, self.height - 55, title="FMT Physics & Problem Setup")

        curr_y = 75.0

        # 1. Parameter Sliders
        new_eta = UIWidgets.slider(25, curr_y, 310, 35, "Bulk Packing Fraction (eta)", self.eta, 0.01, 0.50, fmt="{:.4f}")
        curr_y += 42.0

        # Benchmark Preset Quick Buttons
        btn_w_half = 150.0
        tt_1a = "Set bulk packing fraction to eta=0.4257 (Roth 2010 Fig 1a)"
        if UIWidgets.button(25, curr_y, btn_w_half, 22, "Preset: Fig 1a (0.4257)", bg_color=Theme.HEADER_BG, tooltip=tt_1a):
            self.eta = 0.4257
            self.geom_idx = 0
            self.rebuild_system()

        tt_1b = "Set bulk packing fraction to eta=0.4783 (Roth 2010 Fig 1b)"
        if UIWidgets.button(185, curr_y, btn_w_half, 22, "Preset: Fig 1b (0.4783)", bg_color=Theme.HEADER_BG, tooltip=tt_1b):
            self.eta = 0.4783
            self.geom_idx = 0
            self.rebuild_system()

        curr_y += 28.0

        new_Lz = UIWidgets.slider(25, curr_y, 310, 35, "Domain Length (Lz)", self.Lz, 2.0, 15.0, fmt="{:.2f} sigma")
        curr_y += 42.0

        new_dz = UIWidgets.slider(25, curr_y, 310, 35, "Grid Resolution (dz)", self.dz, 0.002, 0.010, fmt="{:.4f} sigma")
        curr_y += 45.0

        # Check parameter changes
        if abs(new_eta - self.eta) > 1e-5 or abs(new_Lz - self.Lz) > 1e-5 or abs(new_dz - self.dz) > 1e-5:
            self.eta = new_eta
            self.Lz = new_Lz
            self.dz = new_dz
            self.rebuild_system()

        # 2. Viewport Mode Selection (Density vs Weighted Densities vs Free Energy)
        view_options = ["Density rho", "Weighted n", "Free Energy Phi"]
        new_view, h_v = UIWidgets.radio_group(25, curr_y, 310, "Plot Viewport Mode:", view_options, self.view_mode_idx, cols=3)
        curr_y += h_v + 6.0
        if new_view != self.view_mode_idx:
            self.view_mode_idx = new_view
            if self.view_mode_idx == 0:
                self.plotter_main.title = "Density Profile rho(z)"
            elif self.view_mode_idx == 1:
                self.plotter_main.title = "Spatial Weighted Densities n_alpha(z) (FFT Convolutions)"
            else:
                self.plotter_main.title = "Rosenfeld Excess Free Energy Density Phi_RF(z)"

        # 3. Geometry Selection (2 columns)
        geom_options = ["Single Wall (z=0)", "Slit Pore"]
        new_geom, h_g = UIWidgets.radio_group(25, curr_y, 310, "Geometry Mode:", geom_options, self.geom_idx, cols=2)
        curr_y += h_g + 6.0
        if new_geom != self.geom_idx:
            self.geom_idx = new_geom
            self.rebuild_system()

        # 4. FMT Functional Variant Selection (2 columns x 2 rows)
        fmt_options = ["RF (Original)", "WB (White-Bear)", "WBII (Mark II)", "WB-Tensor"]
        new_fmt, h_f = UIWidgets.radio_group(25, curr_y, 310, "FMT Functional Variant:", fmt_options, self.fmt_idx, cols=2)
        curr_y += h_f + 8.0
        if new_fmt != self.fmt_idx:
            self.fmt_idx = new_fmt
            self.rebuild_system()

        # 5. Action Buttons
        btn_w = 150.0
        btn_h = 28.0

        solve_label = "Pause Solver" if self.is_solving else "Solve (Picard)"
        if UIWidgets.button(25, curr_y, btn_w, btn_h, solve_label, bg_color=Theme.PRIMARY_BLUE):
            self.is_solving = not self.is_solving

        if UIWidgets.button(185, curr_y, btn_w, btn_h, "Reset Profile", bg_color=Theme.SLIDER_BG):
            self.rebuild_system()

        curr_y += 32.0

        bm_label = "Hide Benchmark Dots" if self.show_benchmark else "Show Benchmark Dots"
        bm_tooltip = "Toggle Monte Carlo reference dots from Roth (2010) Fig. 1a (eta=0.4257) & Fig. 1b (eta=0.4783)"
        if UIWidgets.button(25, curr_y, 310, btn_h, bm_label, bg_color=Theme.HEADER_BG, tooltip=bm_tooltip):
            self.show_benchmark = not self.show_benchmark

        curr_y += 34.0

        # 6. Physics Info Card & Physical Feasibility Badge
        UIWidgets.draw_panel(25, curr_y, 310, 150, title="System Thermodynamics & Info", bg_color=Theme.BG_DARK)
        info_y = curr_y + 28

        max_n3 = self.wd.max_n3
        if max_n3 < 0.90:
            badge_color = Theme.SUCCESS_GREEN
            status_n3 = f"Safe ({max_n3:.4f})"
        elif max_n3 < 1.0:
            badge_color = Theme.WARNING_AMBER
            status_n3 = f"Dense ({max_n3:.4f})"
        else:
            badge_color = Theme.DANGER_RED
            status_n3 = f"OVER-PACKED ({max_n3:.4f})"

        info_lines = [
            (f"Sphere Radius (R)  : {self.params.radius:.4f} sigma", Theme.TEXT_PRIMARY),
            (f"Bulk Density (rho) : {self.params.rho_bulk:.6f}", Theme.TEXT_PRIMARY),
            (f"Solver Iter (k)    : {self.iteration}", Theme.TEXT_PRIMARY),
            (f"Residual Norm (R)  : {self.residual:.2e}", Theme.WARNING_AMBER if self.is_solving else Theme.TEXT_PRIMARY),
            (f"Max Packing (n3)   : {status_n3}", badge_color),
            (f"Status             : {'Solving (Picard)...' if self.is_solving else ('CONVERGED' if self.residual > 0 and self.residual < 1e-6 else 'Ready / Idle')}", Theme.SUCCESS_GREEN if (self.is_solving or self.residual < 1e-6) else Theme.TEXT_PRIMARY),
        ]

        for line, color in info_lines:
            pr.draw_text(line.encode("utf-8"), int(38), int(info_y), 11, color)
            info_y += 18

    def run(self) -> None:
        """Main Raylib frame loop."""
        while not pr.window_should_close():
            # Handle window resizing
            if pr.is_window_resized():
                self.width = pr.get_screen_width()
                self.height = pr.get_screen_height()
                self.plotter_main = Plotter2D(360, 45, max(400, self.width - 375), max(200, self.height - 240), title="Density Profile rho(z)")
                self.plotter_diag = Plotter2D(360, self.height - 180, max(400, self.width - 375), 170, title="Diagnostic Metrics & Contact Theorem")

            # Update solver step if active
            self.run_solver_step()

            # Render scene
            pr.begin_drawing()
            pr.clear_background(Theme.BG_DARK)

            # Draw Header Bar
            header_rect = pr.Rectangle(0, 0, self.width, 36)
            pr.draw_rectangle_rec(header_rect, Theme.HEADER_BG)
            pr.draw_rectangle_lines_ex(header_rect, 1.0, Theme.PANEL_BORDER)
            pr.draw_text("Classical Density Functional Theory (FMT) - Raylib Interactive Solver".encode("utf-8"), 15, 9, 16, Theme.TEXT_PRIMARY)

            # Render UI sidebar
            self.render_sidebar()

            # Render viewports based on view_mode_idx
            if self.view_mode_idx == 0:
                # Density Profile View
                y_max_plot = max(3.0, np.max(self.rho) * 1.1)
                bm_pts = self.get_benchmark_points()
                sec_curves = [(self.rho_init, Theme.CURVE_SECONDARY, "Initial Guess")]

                self.plotter_main.render(
                    z_arr=self.grid.z,
                    y_arr=self.rho,
                    z_min=0.0,
                    z_max=self.Lz,
                    y_min=0.0,
                    y_max=y_max_plot,
                    x_label="z / sigma",
                    y_label="rho(z)",
                    primary_label="rho(z) Equilibrium Profile",
                    secondary_curves=sec_curves,
                    benchmark_points=bm_pts,
                )
            elif self.view_mode_idx == 1:
                # Spatial Weighted Densities View (FFT Convolutions)
                n3_arr = self.wd.n3
                n2_arr = self.wd.n2
                v2_arr = self.wd.v2

                y_max_plot = max(1.0, np.max(n2_arr) * 1.1)
                sec_curves = [
                    (n2_arr, Theme.SUCCESS_GREEN, "n2(z) Surface Density"),
                    (v2_arr, Theme.DANGER_RED, "v2(z) Vector Flux"),
                ]

                self.plotter_main.render(
                    z_arr=self.grid.z,
                    y_arr=n3_arr,
                    z_min=0.0,
                    z_max=self.Lz,
                    y_min=0.0,
                    y_max=y_max_plot,
                    x_label="z / sigma",
                    y_label="n_alpha(z)",
                    primary_label="n3(z) Packing Fraction",
                    secondary_curves=sec_curves,
                    benchmark_points=None,
                )
            else:
                # Excess Free Energy Density Phi(z) View
                y_max_plot = max(1.0, np.max(self.phi) * 1.2)

                self.plotter_main.render(
                    z_arr=self.grid.z,
                    y_arr=self.phi,
                    z_min=0.0,
                    z_max=self.Lz,
                    y_min=0.0,
                    y_max=y_max_plot,
                    x_label="z / sigma",
                    y_label="Phi_RF(z)",
                    primary_label="Phi_RF(z) Free Energy Density",
                    secondary_curves=None,
                    benchmark_points=None,
                )

            # Render diagnostic plot (c1(z) direct correlation function)
            self.plotter_diag.render(
                z_arr=self.grid.z,
                y_arr=self.c1,
                z_min=0.0,
                z_max=self.Lz,
                y_min=min(0.0, float(np.min(self.c1))) - 0.5,
                y_max=max(1.0, float(np.max(self.c1))) + 0.5,
                x_label="z / sigma",
                y_label="c^(1)(z)",
                primary_label="c^(1)(z) Direct Correlation",
            )

            pr.end_drawing()

        pr.close_window()


def main():
    app = RaylibCDFTApp()
    app.run()


if __name__ == "__main__":
    main()
