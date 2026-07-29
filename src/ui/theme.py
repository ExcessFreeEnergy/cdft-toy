"""Color theme and layout constants for Raylib UI."""

import pyray as pr


class Theme:
    """Dark theme color palette for Raylib interface."""

    BG_DARK = pr.Color(16, 20, 28, 255)
    PANEL_BG = pr.Color(26, 32, 44, 255)
    PANEL_BORDER = pr.Color(45, 55, 75, 255)
    HEADER_BG = pr.Color(34, 42, 58, 255)

    TEXT_PRIMARY = pr.Color(235, 240, 250, 255)
    TEXT_MUTED = pr.Color(135, 150, 175, 255)
    TEXT_DARK = pr.Color(20, 25, 35, 255)

    PRIMARY_BLUE = pr.Color(64, 156, 255, 255)
    PRIMARY_HOVER = pr.Color(90, 172, 255, 255)

    SUCCESS_GREEN = pr.Color(46, 213, 115, 255)
    WARNING_AMBER = pr.Color(255, 171, 0, 255)
    DANGER_RED = pr.Color(255, 71, 87, 255)

    GRID_LINE = pr.Color(40, 50, 68, 255)
    AXIS_LINE = pr.Color(80, 95, 125, 255)
    CURVE_PRIMARY = pr.Color(0, 212, 255, 255)  # Cyan density curve
    CURVE_SECONDARY = pr.Color(255, 171, 0, 255)  # Amber secondary
    BENCHMARK_DOT = pr.Color(255, 71, 87, 255)  # Red simulation dots

    SLIDER_BG = pr.Color(40, 48, 64, 255)
    SLIDER_FILL = pr.Color(64, 156, 255, 255)
    SLIDER_HANDLE = pr.Color(240, 245, 255, 255)
