"""Raylib 2D Plotting Engine for cDFT profiles, weighted densities, and diagnostic curves."""

import numpy as np
import pyray as pr

from src.ui.theme import Theme


class Plotter2D:
    """Raylib 2D line and point plotting viewport."""

    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str = "Density Profile rho(z)",
    ) -> None:
        self.rect = pr.Rectangle(x, y, w, h)
        self.title = title

        # Inner plot padding
        self.pad_left = 55.0
        self.pad_right = 20.0
        self.pad_top = 35.0
        self.pad_bottom = 40.0

        self.plot_x = x + self.pad_left
        self.plot_y = y + self.pad_top
        self.plot_w = w - self.pad_left - self.pad_right
        self.plot_h = h - self.pad_top - self.pad_bottom

    def map_to_screen(
        self,
        z: float,
        y_val: float,
        z_min: float,
        z_max: float,
        y_min: float,
        y_max: float,
    ) -> tuple[float, float]:
        """Transform physical coordinates (z, y_val) to screen pixel coordinates (px, py)."""
        z_span = z_max - z_min if z_max > z_min else 1.0
        y_span = y_max - y_min if y_max > y_min else 1.0

        px = self.plot_x + ((z - z_min) / z_span) * self.plot_w

        if np.isinf(y_val) or np.isnan(y_val):
            py = self.plot_y if (y_val > 0 or np.isnan(y_val)) else (self.plot_y + self.plot_h)
        else:
            py = (self.plot_y + self.plot_h) - ((y_val - y_min) / y_span) * self.plot_h

        return px, py

    def render(
        self,
        z_arr: np.ndarray,
        y_arr: np.ndarray,
        z_min: float = 0.0,
        z_max: float = 10.0,
        y_min: float = 0.0,
        y_max: float = 3.0,
        x_label: str = "z / sigma",
        y_label: str = "rho(z)",
        primary_label: str = "rho(z)",
        secondary_curves: list[tuple[np.ndarray, pr.Color, str]] | None = None,
        benchmark_points: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Render axes, gridlines, curve, secondary curves, benchmark points, legend, and hover tooltip."""
        # 1. Background and border
        pr.draw_rectangle_rec(self.rect, Theme.PANEL_BG)
        pr.draw_rectangle_lines_ex(self.rect, 1.0, Theme.PANEL_BORDER)

        # 2. Header Title
        pr.draw_text(
            self.title.encode("utf-8"),
            int(self.rect.x + 12),
            int(self.rect.y + 8),
            14,
            Theme.TEXT_PRIMARY,
        )

        # 3. Inner plot viewport box
        inner_rect = pr.Rectangle(self.plot_x, self.plot_y, self.plot_w, self.plot_h)
        pr.draw_rectangle_rec(inner_rect, Theme.BG_DARK)
        pr.draw_rectangle_lines_ex(inner_rect, 1.0, Theme.AXIS_LINE)

        # 4. Gridlines & Ticks
        num_grid_x = 5
        num_grid_y = 5

        for i in range(num_grid_x + 1):
            ratio = i / num_grid_x
            gx = self.plot_x + ratio * self.plot_w
            z_val = z_min + ratio * (z_max - z_min)

            pr.draw_line(
                int(gx),
                int(self.plot_y),
                int(gx),
                int(self.plot_y + self.plot_h),
                Theme.GRID_LINE,
            )
            tick_str = f"{z_val:.1f}".encode()
            pr.draw_text(
                tick_str,
                int(gx - 10),
                int(self.plot_y + self.plot_h + 6),
                11,
                Theme.TEXT_MUTED,
            )

        for j in range(num_grid_y + 1):
            ratio = j / num_grid_y
            gy = (self.plot_y + self.plot_h) - ratio * self.plot_h
            y_val = y_min + ratio * (y_max - y_min)

            pr.draw_line(
                int(self.plot_x),
                int(gy),
                int(self.plot_x + self.plot_w),
                int(gy),
                Theme.GRID_LINE,
            )
            tick_str = f"{y_val:.1f}".encode()
            pr.draw_text(tick_str, int(self.plot_x - 38), int(gy - 5), 11, Theme.TEXT_MUTED)

        # Axis Titles
        pr.draw_text(
            x_label.encode("utf-8"),
            int(self.plot_x + self.plot_w / 2 - 25),
            int(self.plot_y + self.plot_h + 22),
            12,
            Theme.TEXT_PRIMARY,
        )
        pr.draw_text(
            y_label.encode("utf-8"),
            int(self.rect.x + 8),
            int(self.plot_y + 10),
            12,
            Theme.PRIMARY_BLUE,
        )

        # 5. Main Primary Curve
        if len(z_arr) > 1 and len(y_arr) == len(z_arr):
            points: list[pr.Vector2] = []
            step = max(1, len(z_arr) // 800)
            for k in range(0, len(z_arr), step):
                px, py = self.map_to_screen(z_arr[k], y_arr[k], z_min, z_max, y_min, y_max)
                px = max(self.plot_x, min(self.plot_x + self.plot_w, px))
                py = max(self.plot_y, min(self.plot_y + self.plot_h, py))
                points.append(pr.Vector2(px, py))

            for idx in range(len(points) - 1):
                pr.draw_line_v(points[idx], points[idx + 1], Theme.CURVE_PRIMARY)

        # 6. Secondary Curves (if provided)
        if secondary_curves:
            for sec_y, sec_color, _sec_name in secondary_curves:
                if len(sec_y) == len(z_arr):
                    sec_pts: list[pr.Vector2] = []
                    for i in range(len(z_arr)):
                        px, py = self.map_to_screen(z_arr[i], sec_y[i], z_min, z_max, y_min, y_max)
                        sec_pts.append(pr.Vector2(px, py))

                    for i in range(len(sec_pts) - 1):
                        pr.draw_line_v(sec_pts[i], sec_pts[i + 1], sec_color)

        # 7. Legend Box
        legend_x = self.plot_x + self.plot_w - 180
        legend_y = self.plot_y + 10
        legend_items = [(Theme.CURVE_PRIMARY, primary_label)]
        if secondary_curves:
            for _, s_color, s_name in secondary_curves:
                legend_items.append((s_color, s_name))

        if legend_items:
            leg_h = 10 + len(legend_items) * 18
            leg_rect = pr.Rectangle(legend_x, legend_y, 170, leg_h)
            pr.draw_rectangle_rec(leg_rect, pr.Color(20, 25, 36, 200))
            pr.draw_rectangle_lines_ex(leg_rect, 1.0, Theme.PANEL_BORDER)

            for idx, (l_color, l_name) in enumerate(legend_items):
                ly = legend_y + 8 + idx * 18
                pr.draw_line(
                    int(legend_x + 8),
                    int(ly + 5),
                    int(legend_x + 24),
                    int(ly + 5),
                    l_color,
                )
                pr.draw_text(
                    l_name.encode("utf-8"),
                    int(legend_x + 30),
                    int(ly),
                    11,
                    Theme.TEXT_PRIMARY,
                )

        # 8. Benchmark Points Overlay (if provided)
        if benchmark_points is not None:
            bm_z, bm_y = benchmark_points
            for bz, by in zip(bm_z, bm_y, strict=False):
                if z_min <= bz <= z_max:
                    px, py = self.map_to_screen(bz, by, z_min, z_max, y_min, y_max)
                    px = max(self.plot_x, min(self.plot_x + self.plot_w, px))
                    py = max(self.plot_y, min(self.plot_y + self.plot_h, py))
                    pr.draw_circle(int(px), int(py), 4.0, Theme.BENCHMARK_DOT)
                    pr.draw_circle_lines(int(px), int(py), 4.0, Theme.TEXT_PRIMARY)

        # 9. Interactive Hover Tooltip
        mouse_pos = pr.get_mouse_position()
        if pr.check_collision_point_rec(mouse_pos, inner_rect) and len(z_arr) > 0:
            rel_x = (mouse_pos.x - self.plot_x) / self.plot_w
            hover_z = z_min + rel_x * (z_max - z_min)

            idx = int(np.clip(np.searchsorted(z_arr, hover_z), 0, len(z_arr) - 1))
            actual_z = float(z_arr[idx])
            actual_y = float(y_arr[idx])

            cur_px, cur_py = self.map_to_screen(actual_z, actual_y, z_min, z_max, y_min, y_max)
            cur_px = max(self.plot_x, min(self.plot_x + self.plot_w, cur_px))
            cur_py = max(self.plot_y, min(self.plot_y + self.plot_h, cur_py))

            pr.draw_line(
                int(cur_px),
                int(self.plot_y),
                int(cur_px),
                int(self.plot_y + self.plot_h),
                Theme.WARNING_AMBER,
            )
            pr.draw_circle(int(cur_px), int(cur_py), 5.0, Theme.WARNING_AMBER)

            if np.isinf(actual_y):
                val_str = "inf" if actual_y > 0 else "-inf"
            elif np.isnan(actual_y):
                val_str = "nan"
            else:
                val_str = f"{actual_y:.4f}"

            tt_str = f"z = {actual_z:.3f}, {y_label} = {val_str}".encode()
            tt_w = pr.measure_text(tt_str, 12) + 12
            tt_x = min(mouse_pos.x + 10, self.plot_x + self.plot_w - tt_w)
            tt_y = max(mouse_pos.y - 25, self.plot_y + 5)

            tt_rect = pr.Rectangle(tt_x, tt_y, tt_w, 22)
            pr.draw_rectangle_rec(tt_rect, Theme.PANEL_BG)
            pr.draw_rectangle_lines_ex(tt_rect, 1.0, Theme.WARNING_AMBER)
            pr.draw_text(tt_str, int(tt_x + 6), int(tt_y + 5), 12, Theme.TEXT_PRIMARY)
