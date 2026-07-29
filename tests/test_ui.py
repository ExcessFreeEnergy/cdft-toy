"""Unit tests for UI plotter coordinate mapping and widget helper logic (Step 11)."""

import numpy as np
import pytest
from src.ui.plotter import Plotter2D
from src.ui.theme import Theme


def test_plotter_coordinate_mapping():
    """Verify physical (z, y) to screen (px, py) coordinate mapping in Plotter2D."""
    plotter = Plotter2D(x=100.0, y=50.0, w=800.0, h=400.0)

    # Physical origin (z_min, y_min) -> (plot_x, plot_y + plot_h)
    px_origin, py_origin = plotter.map_to_screen(
        z=0.0, y_val=0.0, z_min=0.0, z_max=10.0, y_min=0.0, y_max=3.0
    )
    assert pytest.approx(px_origin) == plotter.plot_x
    assert pytest.approx(py_origin) == plotter.plot_y + plotter.plot_h

    # Top-right max (z_max, y_max) -> (plot_x + plot_w, plot_y)
    px_max, py_max = plotter.map_to_screen(
        z=10.0, y_val=3.0, z_min=0.0, z_max=10.0, y_min=0.0, y_max=3.0
    )
    assert pytest.approx(px_max) == plotter.plot_x + plotter.plot_w
    assert pytest.approx(py_max) == plotter.plot_y

    # Midpoint (5.0, 1.5)
    px_mid, py_mid = plotter.map_to_screen(
        z=5.0, y_val=1.5, z_min=0.0, z_max=10.0, y_min=0.0, y_max=3.0
    )
    assert pytest.approx(px_mid) == plotter.plot_x + 0.5 * plotter.plot_w
    assert pytest.approx(py_mid) == plotter.plot_y + 0.5 * plotter.plot_h


def test_theme_color_palette():
    """Verify dark theme palette initialization."""
    assert Theme.BG_DARK.r == 16
    assert Theme.PANEL_BG.g == 32
    assert Theme.PRIMARY_BLUE.b == 255
