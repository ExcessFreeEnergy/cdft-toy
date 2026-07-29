"""Interactive UI widgets for Raylib cDFT application with clean layout and typography."""

from typing import List, Tuple
import pyray as pr
from src.ui.theme import Theme


class UIWidgets:
    """Helper collection of Raylib GUI widgets."""

    @staticmethod
    def draw_panel(
        x: float, y: float, w: float, h: float, title: str = "", bg_color: pr.Color = Theme.PANEL_BG
    ) -> None:
        """Draw a UI container panel with optional header title."""
        rect = pr.Rectangle(x, y, w, h)
        pr.draw_rectangle_rec(rect, bg_color)
        pr.draw_rectangle_lines_ex(rect, 1.0, Theme.PANEL_BORDER)

        if title:
            header_rect = pr.Rectangle(x, y, w, 28)
            pr.draw_rectangle_rec(header_rect, Theme.HEADER_BG)
            pr.draw_rectangle_lines_ex(header_rect, 1.0, Theme.PANEL_BORDER)
            pr.draw_text(title.encode("utf-8"), int(x + 10), int(y + 6), 14, Theme.TEXT_PRIMARY)

    @staticmethod
    def slider(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        fmt: str = "{:.4f}",
    ) -> float:
        """Draw interactive slider control and return updated value."""
        val_str = fmt.format(value)
        text_str = f"{label}: {val_str}"
        pr.draw_text(text_str.encode("utf-8"), int(x), int(y), 13, Theme.TEXT_PRIMARY)

        slider_y = y + 18
        slider_h = 10
        bar_rect = pr.Rectangle(x, slider_y, w, slider_h)

        mouse_pos = pr.get_mouse_position()
        mouse_down = pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT)

        val_ratio = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
        val_ratio = max(0.0, min(1.0, val_ratio))

        hit_rect = pr.Rectangle(x, slider_y - 4, w, slider_h + 8)
        if pr.check_collision_point_rec(mouse_pos, hit_rect) and mouse_down:
            rel_x = mouse_pos.x - x
            val_ratio = max(0.0, min(1.0, rel_x / w))
            value = min_val + val_ratio * (max_val - min_val)

        pr.draw_rectangle_rec(bar_rect, Theme.SLIDER_BG)
        fill_rect = pr.Rectangle(x, slider_y, w * val_ratio, slider_h)
        pr.draw_rectangle_rec(fill_rect, Theme.SLIDER_FILL)
        pr.draw_rectangle_lines_ex(bar_rect, 1.0, Theme.PANEL_BORDER)

        handle_x = x + w * val_ratio
        handle_y = slider_y + slider_h / 2.0
        pr.draw_circle(int(handle_x), int(handle_y), 7.0, Theme.SLIDER_HANDLE)
        pr.draw_circle_lines(int(handle_x), int(handle_y), 7.0, Theme.PRIMARY_BLUE)

        return float(value)

    @staticmethod
    def button(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        bg_color: pr.Color = Theme.PRIMARY_BLUE,
        enabled: bool = True,
    ) -> bool:
        """Draw interactive button and return True if clicked this frame."""
        rect = pr.Rectangle(x, y, w, h)
        mouse_pos = pr.get_mouse_position()
        is_hover = pr.check_collision_point_rec(mouse_pos, rect) and enabled
        is_clicked = is_hover and pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT)

        fill_color = bg_color
        if not enabled:
            fill_color = Theme.SLIDER_BG
        elif is_hover:
            fill_color = Theme.PRIMARY_HOVER

        pr.draw_rectangle_rec(rect, fill_color)
        pr.draw_rectangle_lines_ex(rect, 1.0, Theme.PANEL_BORDER if enabled else Theme.GRID_LINE)

        # Auto-adjust font size to ensure text fits button width without overlapping
        font_size = 13
        text_bytes = label.encode("utf-8")
        text_w = pr.measure_text(text_bytes, font_size)

        while text_w > (w - 6) and font_size > 9:
            font_size -= 1
            text_w = pr.measure_text(text_bytes, font_size)

        text_x = int(x + (w - text_w) / 2.0)
        text_y = int(y + (h - font_size) / 2.0)

        text_color = Theme.TEXT_PRIMARY if enabled else Theme.TEXT_MUTED
        pr.draw_text(text_bytes, text_x, text_y, font_size, text_color)

        return is_clicked

    @staticmethod
    def radio_group(
        x: float, y: float, w: float, label: str, options: List[str], selected_idx: int, cols: int = 2
    ) -> Tuple[int, float]:
        """Draw radio selection grid (multi-column) to prevent text overlap.

        Returns (new_selected_index, total_height_used).
        """
        pr.draw_text(label.encode("utf-8"), int(x), int(y), 13, Theme.TEXT_PRIMARY)
        curr_y = y + 20

        cols = max(1, min(cols, len(options)))
        btn_w = (w - 6 * (cols - 1)) / cols
        btn_h = 26.0

        num_rows = (len(options) + cols - 1) // cols

        for i, opt in enumerate(options):
            col_idx = i % cols
            row_idx = i // cols

            bx = x + col_idx * (btn_w + 6)
            by = curr_y + row_idx * (btn_h + 5)

            is_selected = i == selected_idx
            bg = Theme.PRIMARY_BLUE if is_selected else Theme.SLIDER_BG

            if UIWidgets.button(bx, by, btn_w, btn_h, opt, bg_color=bg):
                selected_idx = i

        total_h = 20.0 + num_rows * (btn_h + 5)
        return selected_idx, total_h
