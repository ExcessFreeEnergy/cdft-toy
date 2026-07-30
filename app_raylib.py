"""Raylib Desktop GUI Application for Classical Density Functional Theory (FMT) Hard-Sphere Simulator."""

import numpy as np
import pyray as pr

from src.crossover import CrossoverAnalyzer
from src.diagnostics import SumRuleDiagnostics
from src.functionals import functional_factory
from src.grid import Grid1D, PhysicalParameters
from src.solvers import RothPicardSolver
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
        self.fmt_names = ["RF", "WB", "WBII", "WB-Tensor"]  # Available functional names
        self.view_mode_idx = 0  # 0: Density Profile rho(z), 1: Weighted Densities n_alpha(z), 2: Free Energy Phi(z)

        self.is_solving = False
        self.show_benchmark = True
        self.iteration = 0
        self.residual = 0.0
        self.alpha_used = 0.03
        self.residual_history: list[float] = []
        self.diag_view_mode = 0  # 0: Direct Correlation c1(z), 1: Residual History R(k)

        # Instantiate functional engine from current selection
        fmt_name = self.fmt_names[self.fmt_idx] if self.fmt_idx < len(self.fmt_names) else "RF"
        self.func = functional_factory(fmt_name)

        # Physical system, grid, solver, and weighted density calculator init
        self.rebuild_system()

        # Raylib window initialization
        pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE | pr.FLAG_MSAA_4X_HINT)
        pr.init_window(self.width, self.height, b"Classical DFT (FMT) Physics Visualizer & Solver")
        pr.set_target_fps(60)

        # Automatically default size of app to 90% height and width of monitor
        mon = pr.get_current_monitor()
        mon_w = pr.get_monitor_width(mon)
        mon_h = pr.get_monitor_height(mon)
        if mon_w > 0 and mon_h > 0:
            target_w = int(mon_w * 0.90)
            target_h = int(mon_h * 0.90)
            pr.set_window_size(target_w, target_h)
            self.width = target_w
            self.height = target_h

        # Plotter viewports scaled to window dimensions
        self.plotter_main = Plotter2D(
            360,
            45,
            max(400, self.width - 375),
            max(200, self.height - 240),
            title="Density Profile rho(z)",
        )
        self.plotter_diag = Plotter2D(
            360,
            self.height - 180,
            max(400, self.width - 375),
            170,
            title="Diagnostic Metrics & Contact Theorem",
        )

    def rebuild_system(self) -> None:
        """Reconstruct PhysicalParameters, Grid1D, RothPicardSolver, and initial density profile."""
        # Recreate functional from current selection
        fmt_name = self.fmt_names[self.fmt_idx] if self.fmt_idx < len(self.fmt_names) else "RF"
        self.func = functional_factory(fmt_name)

        self.params = PhysicalParameters(eta=self.eta)
        dz_clamped = min(0.010, max(0.001, self.dz))
        self.grid = Grid1D(params=self.params, Lz=self.Lz, dz=dz_clamped)
        self.calc = WeightedDensityCalculator(self.grid, apply_endpoint_modification=True)

        w_left = 0.0
        w_right = self.Lz if self.geom_idx == 1 else None

        # Instantiate physical Roth adaptive Picard solver
        self.solver = RothPicardSolver(
            grid=self.grid,
            functional=self.func,
            alpha_init=0.03,
            wall_left=w_left,
            wall_right=w_right,
        )

        self.v_ext = self.grid.external_potential(wall_left=w_left, wall_right=w_right)
        self.rho = self.grid.initial_density_profile(wall_left=w_left, wall_right=w_right)
        self.rho_init = self.rho.copy()

        self.rho_prev = None
        self.rho_target_prev = None

        self.iteration = 0
        self.residual = 0.0
        self.alpha_used = 0.03

        # Compute initial weighted densities, free energy density, and c1
        self.wd = self.calc.compute(self.rho)
        self.n_dict = self.wd.to_dict()
        self.phi = self.func.evaluate_phi(self.wd)
        self.f_ex = self.func.compute_total_free_energy(self.grid, self.wd)

        self.c1, self.c1_bulk = self.solver.compute_c1(self.rho)

    def get_benchmark_points(self):
        """Return benchmark Monte Carlo data points from Roth (2010) Fig. 1 for comparison."""
        if not self.show_benchmark:
            return None

        # Roth 2010 Fig 1a data points (eta = 0.4257)
        if abs(self.eta - 0.4257) < 0.01 and self.geom_idx == 0:
            bm_z = np.array(
                [
                    0.50,
                    0.55,
                    0.65,
                    0.85,
                    1.05,
                    1.30,
                    1.55,
                    1.80,
                    2.10,
                    2.50,
                    3.00,
                    4.00,
                    5.00,
                ]
            )
            bm_y = np.array(
                [
                    2.52,
                    2.20,
                    1.45,
                    0.52,
                    0.48,
                    0.95,
                    1.25,
                    0.98,
                    0.75,
                    0.82,
                    0.81,
                    0.813,
                    0.813,
                ]
            )
            return bm_z, bm_y

        # Roth 2010 Fig 1b data points (eta = 0.4783)
        if abs(self.eta - 0.4783) < 0.01 and self.geom_idx == 0:
            bm_z = np.array([0.50, 0.55, 0.65, 0.85, 1.05, 1.30, 1.55, 1.80, 2.10, 2.50, 3.00, 4.00])
            bm_y = np.array(
                [
                    3.45,
                    2.90,
                    1.70,
                    0.40,
                    0.35,
                    1.15,
                    1.62,
                    1.10,
                    0.70,
                    0.92,
                    0.91,
                    0.913,
                ]
            )
            return bm_z, bm_y

        return None

    def execute_single_step(self) -> None:
        """Execute exactly one adaptive Roth Picard solver step for single-step debugging."""
        self.iteration += 1
        rho_next, rho_target_cur, self.c1, self.residual, self.alpha_used = self.solver.solve_step_adaptive(
            self.rho, self.rho_prev, self.rho_target_prev
        )

        self.rho_prev = self.rho.copy()
        self.rho_target_prev = rho_target_cur.copy()
        self.rho = rho_next
        self.residual_history.append(float(self.residual))

        # Recompute spatial weighted densities and free energy density
        self.wd = self.calc.compute(self.rho)
        self.n_dict = self.wd.to_dict()
        self.phi = self.func.evaluate_phi(self.wd)
        self.f_ex = self.func.compute_total_free_energy(self.grid, self.wd)

        if self.residual < 1e-6 or self.iteration >= 2000:
            self.is_solving = False

    def run_solver_step(self) -> None:
        """Execute a single adaptive Roth Picard solver step and update all physical quantities."""
        if not self.is_solving:
            return

        self.execute_single_step()

    def render_sidebar(self) -> None:
        """Render controls, sliders, radio buttons, and diagnostic metrics panel."""
        sidebar_w = 340.0
        UIWidgets.draw_panel(10, 45, sidebar_w, self.height - 55, title="FMT Physics & Problem Setup")

        curr_y = 75.0

        # 1. Parameter Sliders
        new_eta = UIWidgets.slider(
            25,
            curr_y,
            310,
            35,
            "Bulk Packing Fraction (eta)",
            self.eta,
            0.01,
            0.50,
            fmt="{:.4f}",
        )
        curr_y += 42.0

        # Benchmark Preset Quick Buttons
        btn_w_half = 150.0
        tt_1a = "Set bulk packing fraction to eta=0.4257 (Roth 2010 Fig 1a)"
        if UIWidgets.button(
            25,
            curr_y,
            btn_w_half,
            22,
            "Preset: Fig 1a (0.4257)",
            bg_color=Theme.HEADER_BG,
            tooltip=tt_1a,
        ):
            self.eta = 0.4257
            new_eta = 0.4257
            self.geom_idx = 0
            self.rebuild_system()

        tt_1b = "Set bulk packing fraction to eta=0.4783 (Roth 2010 Fig 1b)"
        if UIWidgets.button(
            185,
            curr_y,
            btn_w_half,
            22,
            "Preset: Fig 1b (0.4783)",
            bg_color=Theme.HEADER_BG,
            tooltip=tt_1b,
        ):
            self.eta = 0.4783
            new_eta = 0.4783
            self.geom_idx = 0
            self.rebuild_system()

        curr_y += 28.0

        new_Lz = UIWidgets.slider(
            25,
            curr_y,
            310,
            35,
            "Domain Length (Lz)",
            self.Lz,
            0.10,
            15.0,
            fmt="{:.2f} sigma",
        )
        curr_y += 42.0

        new_dz = UIWidgets.slider(
            25,
            curr_y,
            310,
            35,
            "Grid Resolution (dz)",
            self.dz,
            0.002,
            0.010,
            fmt="{:.4f} sigma",
        )
        curr_y += 45.0

        # Check parameter changes
        if abs(new_eta - self.eta) > 1e-5 or abs(new_Lz - self.Lz) > 1e-5 or abs(new_dz - self.dz) > 1e-5:
            self.eta = new_eta
            self.Lz = new_Lz
            self.dz = new_dz
            self.rebuild_system()

        # 2. Viewport Mode Selection (Density vs Weighted Densities vs Free Energy vs Crossover Suite)
        view_options = ["Density rho", "Weighted n", "Free Energy Phi", "Crossover Suite"]
        new_view, h_v = UIWidgets.radio_group(
            25,
            curr_y,
            310,
            "Plot Viewport Mode:",
            view_options,
            self.view_mode_idx,
            cols=2,
        )
        curr_y += h_v + 6.0
        if new_view != self.view_mode_idx:
            self.view_mode_idx = new_view
            if self.view_mode_idx == 0:
                self.plotter_main.title = "Density Profile rho(z)"
            elif self.view_mode_idx == 1:
                self.plotter_main.title = "Spatial Weighted Densities n_alpha(z) (FFT Convolutions)"
            elif self.view_mode_idx == 2:
                fmt_label = self.fmt_names[self.fmt_idx] if self.fmt_idx < len(self.fmt_names) else "RF"
                self.plotter_main.title = f"Excess Free Energy Density Phi_{fmt_label}(z)"
            else:
                self.plotter_main.title = "Dimensional Crossover & Zero-D Cavity Free Energy Stability"

        # 3. Geometry Selection (2 columns)
        geom_options = ["Single Wall (z=0)", "Slit Pore"]
        new_geom, h_g = UIWidgets.radio_group(25, curr_y, 310, "Geometry Mode:", geom_options, self.geom_idx, cols=2)
        curr_y += h_g + 6.0
        if new_geom != self.geom_idx:
            self.geom_idx = new_geom
            self.rebuild_system()

        # 4. FMT Functional Variant Selection (2 columns x 2 rows)
        fmt_options = [
            "RF (Original)",
            "WB (White-Bear)",
            "WBII (Mark II)",
            "WB-Tensor",
        ]
        new_fmt, h_f = UIWidgets.radio_group(
            25,
            curr_y,
            310,
            "FMT Functional Variant:",
            fmt_options,
            self.fmt_idx,
            cols=2,
        )
        curr_y += h_f + 8.0
        if new_fmt != self.fmt_idx:
            self.fmt_idx = new_fmt
            self.rebuild_system()

        # 5. Action Buttons (3-column layout)
        btn_w3 = 98.0
        btn_h = 28.0

        solve_label = "Pause" if self.is_solving else "Solve"
        if UIWidgets.button(25, curr_y, btn_w3, btn_h, solve_label, bg_color=Theme.PRIMARY_BLUE):
            self.is_solving = not self.is_solving

        if UIWidgets.button(
            131,
            curr_y,
            btn_w3,
            btn_h,
            "Step 1 Iter",
            bg_color=Theme.HEADER_BG,
            tooltip="Advance Picard solver by exactly 1 iteration",
        ):
            self.execute_single_step()

        if UIWidgets.button(237, curr_y, btn_w3, btn_h, "Reset", bg_color=Theme.SLIDER_BG):
            self.rebuild_system()

        curr_y += 32.0

        bm_label = "Hide Benchmark Dots" if self.show_benchmark else "Show Benchmark Dots"
        bm_tooltip = "Toggle Monte Carlo reference dots from Roth (2010) Fig. 1a (eta=0.4257) & Fig. 1b (eta=0.4783)"
        if UIWidgets.button(
            25,
            curr_y,
            150.0,
            btn_h,
            bm_label,
            bg_color=Theme.HEADER_BG,
            tooltip=bm_tooltip,
        ):
            self.show_benchmark = not self.show_benchmark

        diag_label = "Show R(k) History" if self.diag_view_mode == 0 else "Show c^(1)(z)"
        if UIWidgets.button(
            185.0,
            curr_y,
            150.0,
            btn_h,
            diag_label,
            bg_color=Theme.HEADER_BG,
            tooltip="Toggle lower diagnostic plot between c1(z) profile and residual convergence history R(k)",
        ):
            self.diag_view_mode = 1 - self.diag_view_mode
            if self.diag_view_mode == 0:
                self.plotter_diag.title = "Diagnostic Metrics & Direct Correlation c^(1)(z)"
            else:
                self.plotter_diag.title = "Solver Residual Convergence History log10 R(k)"

        curr_y += 34.0

        # 6. Physics Info Card & Physical Feasibility Badge
        UIWidgets.draw_panel(
            25,
            curr_y,
            310,
            240,
            title="System Thermodynamics & Info",
            bg_color=Theme.BG_DARK,
        )
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

        fmt_label = self.fmt_names[self.fmt_idx] if self.fmt_idx < len(self.fmt_names) else "RF"
        p_bulk = self.func.compute_bulk_pressure(self.eta, self.params.sigma)

        # Compute live sum-rule diagnostics
        rho_c = SumRuleDiagnostics.extrapolate_contact_density(self.grid, self.rho)
        gamma_sp = SumRuleDiagnostics.compute_surface_tension(self.grid, self.rho, self.func, self.calc)
        gamma_bulk = SumRuleDiagnostics.compute_bulk_route_surface_tension(self.func, self.eta, self.params.sigma)
        gamma_ex = SumRuleDiagnostics.compute_excess_adsorption(self.grid, self.rho)
        err_contact = abs(rho_c - p_bulk) / p_bulk * 100.0

        status_str = (
            "Solving (Roth Picard)..." if self.is_solving else ("CONVERGED" if 0 < self.residual < 1e-6 else "Ready / Idle")
        )

        info_lines = [
            (f"Active Functional  : {fmt_label}", Theme.PRIMARY_BLUE),
            (
                f"Sphere Radius (R)  : {self.params.radius:.4f} sigma",
                Theme.TEXT_PRIMARY,
            ),
            (f"Bulk Density (rho) : {self.params.rho_bulk:.6f}", Theme.TEXT_PRIMARY),
            (f"Bulk Pressure (bp) : {p_bulk:.4f}", Theme.TEXT_PRIMARY),
            (
                f"Contact rho(R+)    : {rho_c:.4f} ({err_contact:.2f}% err)",
                Theme.SUCCESS_GREEN if err_contact < 0.1 else Theme.TEXT_PRIMARY,
            ),
            (
                f"Surface Tension bg : {gamma_sp:.4f} (bulk: {gamma_bulk:.4f})",
                Theme.TEXT_PRIMARY,
            ),
            (f"Excess Adsorption G: {gamma_ex:.4f}", Theme.TEXT_PRIMARY),
            (f"Solver Iter (k)    : {self.iteration}", Theme.TEXT_PRIMARY),
            (f"Alpha Opt (alpha)  : {self.alpha_used:.4f}", Theme.PRIMARY_BLUE),
            (
                f"Residual Norm (R)  : {self.residual:.2e}",
                Theme.WARNING_AMBER if self.is_solving else Theme.TEXT_PRIMARY,
            ),
            (f"Max Packing (n3)   : {status_n3}", badge_color),
            (
                f"Status             : {status_str}",
                Theme.SUCCESS_GREEN if (self.is_solving or self.residual < 1e-6) else Theme.TEXT_PRIMARY,
            ),
        ]

        for line, color in info_lines:
            pr.draw_text(line.encode("utf-8"), 38, int(info_y), 11, color)
            info_y += 18

    def run(self) -> None:
        """Main Raylib frame loop."""
        while not pr.window_should_close():
            # Handle window resizing
            if pr.is_window_resized():
                self.width = pr.get_screen_width()
                self.height = pr.get_screen_height()
                self.plotter_main = Plotter2D(
                    360,
                    45,
                    max(400, self.width - 375),
                    max(200, self.height - 240),
                    title="Density Profile rho(z)",
                )
                self.plotter_diag = Plotter2D(
                    360,
                    self.height - 180,
                    max(400, self.width - 375),
                    170,
                    title="Diagnostic Metrics & Contact Theorem",
                )

            # Update solver step if active
            self.run_solver_step()

            # Render scene
            pr.begin_drawing()
            pr.clear_background(Theme.BG_DARK)

            # Draw Header Bar
            header_rect = pr.Rectangle(0, 0, self.width, 36)
            pr.draw_rectangle_rec(header_rect, Theme.HEADER_BG)
            pr.draw_rectangle_lines_ex(header_rect, 1.0, Theme.PANEL_BORDER)
            pr.draw_text(
                b"Classical Density Functional Theory (FMT) - Raylib Interactive Solver",
                15,
                9,
                16,
                Theme.TEXT_PRIMARY,
            )

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
            elif self.view_mode_idx == 2:
                # Excess Free Energy Density Phi(z) View
                y_max_plot = max(1.0, np.max(self.phi) * 1.2)
                active_fmt = self.fmt_names[self.fmt_idx] if self.fmt_idx < len(self.fmt_names) else "RF"

                self.plotter_main.render(
                    z_arr=self.grid.z,
                    y_arr=self.phi,
                    z_min=0.0,
                    z_max=self.Lz,
                    y_min=0.0,
                    y_max=y_max_plot,
                    x_label="z / sigma",
                    y_label="Phi(z)",
                    primary_label=f"Phi(z) Free Energy Density ({active_fmt})",
                    secondary_curves=None,
                    benchmark_points=None,
                )
            else:
                # Crossover Suite View: Zero-D Cavity Free Energy Stability
                alphas = np.linspace(0.02, 0.20, 20)
                phi_wb_tensor = CrossoverAnalyzer.evaluate_zero_d_divergence(
                    self.grid, alphas, functional_factory("WB-Tensor"), self.calc
                )
                phi_rf = CrossoverAnalyzer.evaluate_zero_d_divergence(self.grid, alphas, functional_factory("RF"), self.calc)
                phi_wb = CrossoverAnalyzer.evaluate_zero_d_divergence(self.grid, alphas, functional_factory("WB"), self.calc)

                sec_curves = [
                    (np.array(phi_rf), Theme.DANGER_RED, "RF Divergence Spike"),
                    (np.array(phi_wb), Theme.WARNING_AMBER, "WB Scalar Divergence"),
                ]
                y_max_crossover = max(10.0, float(np.max(phi_wb_tensor)) * 1.5)

                self.plotter_main.render(
                    z_arr=alphas,
                    y_arr=np.array(phi_wb_tensor),
                    z_min=0.02,
                    z_max=0.20,
                    y_min=0.0,
                    y_max=y_max_crossover,
                    x_label="alpha / sigma (Cavity Width)",
                    y_label="max Phi(z)",
                    primary_label="WB-Tensor (Bounded Free Energy)",
                    secondary_curves=sec_curves,
                    benchmark_points=None,
                )

            # Render diagnostic plot (c1(z) direct correlation or residual history log10 R(k))
            if self.diag_view_mode == 0:
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
            else:
                if self.residual_history:
                    k_arr = np.arange(1, len(self.residual_history) + 1, dtype=float)
                    log_res = np.log10(np.clip(self.residual_history, 1e-12, 1.0))
                    k_max = max(10.0, float(len(self.residual_history)))
                    y_min_log = min(-6.0, float(np.min(log_res)) - 0.5)
                    y_max_log = max(0.0, float(np.max(log_res)) + 0.5)
                else:
                    k_arr = np.array([1.0])
                    log_res = np.array([0.0])
                    k_max = 10.0
                    y_min_log = -6.0
                    y_max_log = 0.0

                self.plotter_diag.render(
                    z_arr=k_arr,
                    y_arr=log_res,
                    z_min=1.0,
                    z_max=k_max,
                    y_min=y_min_log,
                    y_max=y_max_log,
                    x_label="Iteration Step (k)",
                    y_label="log10 Residual R(k)",
                    primary_label="log10 Residual R(k)",
                )

            pr.end_drawing()

        pr.close_window()


def main():
    app = RaylibCDFTApp()
    app.run()


if __name__ == "__main__":
    main()
