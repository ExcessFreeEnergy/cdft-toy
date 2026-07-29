# Classical Density Functional Theory (cDFT / FMT) Simulator

A high-performance Python simulator and interactive **Raylib GUI application** for Classical Density Functional Theory (cDFT) of hard-sphere fluids and mixtures using Fundamental Measure Theory (FMT).

Based on **Roland Roth (2010)**: *Fundamental measure theory for hard-sphere mixtures: a review*, J. Phys.: Condens. Matter **22**, 063102.

---

## 🚀 Quick Start

### Prerequisites
Ensure [uv](https://github.com/astral-sh/uv) is installed.

### Installation & Environment Setup

```bash
# Install dependencies into virtual environment
uv sync
```

---

## 🖥️ Running the Interactive Raylib Desktop GUI

Launch the desktop visualizer and solver application:

```bash
uv run python app_raylib.py
```

### 🎮 GUI Features & Controls

- **Interactive Physics Sliders**:
  - **Bulk Packing Fraction ($\eta$)**: Adjust bulk packing fraction from `0.01` to `0.50`. Automatically recalculates bulk number density $\rho_{\text{bulk}} = \frac{6\eta}{\pi \sigma^3}$.
  - **Domain Length ($L_z$)**: Set 1D spatial domain height $L_z \in [2.0\sigma, 15.0\sigma]$.
  - **Grid Resolution ($dz$)**: Adjust grid spacing $dz \in [0.002\sigma, 0.010\sigma]$ (ensuring $dz \le 0.01\sigma$).
- **Geometry Mode Selection**:
  - **Single Wall ($z=0$)**: Planar hard wall at $z=0$, open bulk at $z = L_z$.
  - **Slit Pore**: Confined fluid between two hard walls at $z=0$ and $z=L_z$.
- **FMT Functional Selection**:
  - **RF (Original)**: Original Rosenfeld functional (PY compressibility equation of state).
  - **WB (White-Bear)**: White-Bear version of FMT based on Mansoori–Carnahan–Starling–Leland (MCSL) equation of state.
  - **WBII (Mark II)**: White-Bear II functional based on CSIII equation of state.
  - **WB-Tensor**: Tarazona tensorial FMT with White-Bear equation of state prefactor for tight confinement stability.
- **Action Buttons**:
  - **Solve / Relax**: Trigger real-time profile relaxation solver.
  - **Reset Profile**: Reset density profile $\rho(z)$ back to unrelaxed step function guess $\rho_{\text{bulk}} e^{-\beta V_{\text{ext}}(z)}$.
  - **Show / Hide Benchmark**: Overlay Monte Carlo simulation benchmark data points (Roth 2010 Figs. 1a & 1b) directly onto the density plot.
- **2D Plotting & Inspection**:
  - **Live Density Curve**: Renders $\rho(z)$ (cyan curve) against initial profile guess (amber line).
  - **Interactive Hover Tooltip**: Move mouse over the plot area to inspect exact numerical values for $z$ and $\rho(z)$.

---

## 🧪 Running CLI Demos & Automated Tests

### 1. Run Step 01 Physics Demonstration

```bash
uv run python scripts/demo_step01.py
```

### 2. Run Test Suite

```bash
uv run pytest -v
```

---

## 📁 Repository Structure

```text
cdft/
├── app_raylib.py        # Raylib 60 FPS Desktop GUI Application launcher
├── pyproject.toml       # UV project dependencies and build settings
├── README.md            # Project documentation and usage guide
├── tasks.md             # Implementation roadmap and step-by-step task tracking
├── review/
│   └── review.md        # Reference paper (Roth 2010 review on FMT)
├── scripts/
│   └── demo_step01.py   # CLI demo for physical parameters and grid discretization
├── src/
│   ├── grid.py          # PhysicalParameters, Grid1D, hard wall potential & integration
│   └── ui/
│       ├── theme.py     # Dark mode color palette & layout constants
│       ├── widgets.py   # Clean Raylib GUI widgets (sliders, auto-fitting buttons, 2x2 radio grid)
│       └── plotter.py   # 2D coordinate mapping, line renderer, & hover tooltips
└── tests/
    ├── test_grid.py     # Unit tests for physical parameters, grid discretization, & integration
    └── test_ui.py       # Headless unit tests for UI plotter coordinate math
```
