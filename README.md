# Classical Density Functional Theory (cDFT / FMT) Hard-Sphere Simulator & Interactive Visualizer

A high-performance Python simulation engine, comprehensive tutorial masterclass, and interactive **Raylib Desktop GUI Application** for Classical Density Functional Theory (cDFT) of hard-sphere fluids and mixtures using Fundamental Measure Theory (FMT).

Based on the topical review paper:  
**Roland Roth (2010)**: *Fundamental measure theory for hard-sphere mixtures: a review*, **Journal of Physics: Condensed Matter** 22, 063102.

---

## 🌟 Key Features

- **4 FMT Functional Variants**:
  - **Rosenfeld (RF)**: Original 1989 Rosenfeld functional with Percus–Yevick (PY) compressibility equation of state.
  - **White-Bear (WB)**: White-Bear functional matching the Carnahan–Starling (CS) equation of state with low-density Taylor expansions ($n_3 < 10^{-3}$) to prevent floating-point division by zero.
  - **White-Bear II (WBII)**: Mark II White-Bear functional based on CSIII bulk thermodynamics.
  - **WB-Tensor**: Tarazona tensorial FMT incorporating 1D planar tensorial weight $\omega_{m2}(z) = 2\pi R (\frac{z^2}{R^2} - \frac{1}{3})$ and trace cancellation $9(v_2^2 n_{m2} - \frac{3}{8}n_{m2}^3)$, removing non-physical divergence spikes under zero-D cavity collapse.
- **Fast Fourier Transform (FFT) Convolution Engine**: Zero-padded FFT convolutions (`scipy.fft`) eliminating periodic boundary wraparound.
- **Roth Adaptive Line-Search Picard Solver**: Optimal scalar mixing parameter calculation $\alpha_{\text{opt}} = -\frac{\int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}}}{\int \Delta \rho_{\text{out}}^2}$ with maximum packing fraction $n_3(z) < 1$ backtracking.
- **Thermodynamic Sum-Rules & Diagnostics**:
  - **Wall Contact Theorem**: 3-point Lagrange polynomial extrapolation verifying $\rho(R^+) = \beta P_{\text{bulk}}$.
  - **Wall Surface Tension**: Spatial grand-potential integration $\beta \gamma$ vs analytical bulk-route surface tension $\beta \gamma_{\text{bulk}} = \left. \frac{\partial \Phi}{\partial n_2} \right|_{\text{bulk}}$.
  - **Excess Adsorption & Gibbs Theorem**: $\Gamma = \int (\rho(z) - \rho_{\text{bulk}}) dz$ and numerical finite-difference Gibbs adsorption verification $-\frac{d\gamma}{d\mu} = \Gamma$.
- **Interactive Raylib Desktop GUI (`app_raylib.py`)**:
  - Auto-scaled window defaulting to **90% monitor width and height**.
  - Live 2D plotting viewports for density profiles $\rho(z)$, weighted densities $n_\alpha(z)$, local free energy density $\Phi(z)$, direct correlation $c^{(1)}(z)$, and residual convergence history $\log_{10} R(k)$.
  - Single-step solver execution button (`Step 1 Iter`) for frame-by-frame inspection.
  - Published Monte Carlo benchmark data overlay (Roth 2010 Figs. 1a & 1b for $\eta = 0.4257$ and $\eta = 0.4783$).
  - **Dimensional Crossover & Confinement Mode** (`Crossover Suite`).

---

## 📖 Source Paper Summary (Roth 2010)

This software project is built as a complete mathematical implementation of Roland Roth's landmark 2010 review paper:

- **Section 2 & 3**: Geometric fundamental measure weight functions, deconvolution of hard-core interaction potentials, and vector parity symmetry.
- **Section 4.1**: Rosenfeld functional derivation, Scaled Particle Theory (SPT), and Percus-Yevick compressibility EOS.
- **Section 4.2 & 4.4**: Dimensional reduction ($3D \to 2D \to 1D \to 0D$), zero-dimensional single-particle cavity collapse, and Tarazona tensorial weight functions.
- **Section 4.3**: White-Bear and White-Bear II functionals incorporating Carnahan-Starling bulk fluid thermodynamics.
- **Section 8**: Numerical FFT convolution algorithms, Section 8.4 Simpson endpoint quadrature modifications, and Section 8.1 adaptive line-search Picard solver.

---

## 🎓 Interactive Masterclass Companion (`tutorial.md`)

The repository includes a comprehensive, self-contained textbook and laboratory companion guide in **[tutorial.md](file:///home/gauss/code/cdft/tutorial.md)**.

### Masterclass Structure:
- **Module 1**: Geometry & Fundamentals of FMT (Weight Integrals & Parity).
- **Module 2**: Scaled Particle Theory & Percus-Yevick Fluid Thermodynamics.
- **Module 3**: White-Bear & White-Bear II Carnahan-Starling Functionals.
- **Module 4**: Dimensional Crossover, Slit-Pore Confinement & Tarazona Tensorial FMT.
- **Module 5**: FFT Convolutions, Endpoint Quadrature & Adaptive Picard Mechanics.
- **Module 6**: Thermodynamic Sum-Rules, Contact Theorem & Gibbs Adsorption.
- **Module 7**: End-to-End Performance Benchmarks & Monte Carlo Regression.
- **Module 8**: **Complete Analytical Solutions to Problem Sets 1–6**.

Each module contains **Hands-On Interactive Experiments** runnable directly in the Raylib GUI or via test and demo scripts.

---

## 🚀 Quick Start

### Prerequisites
Ensure Python 3.10+ and [uv](https://github.com/astral-sh/uv) are installed.

### Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/cdft.git
cd cdft

# Install dependencies into virtual environment
uv sync
```

---

## 🖥️ Running the Interactive Raylib Desktop Application

Launch the 60 FPS interactive desktop visualizer:

```bash
uv run python app_raylib.py
```

### GUI Features & Controls
- **Plot Viewport Mode**:
  - `Density rho`: Renders spatial density profile $\rho(z)$ against initial guess $\rho_{\text{init}}(z)$ and Monte Carlo benchmark dots.
  - `Weighted n`: Renders spatial weighted densities $n_3(z)$ (packing fraction), $n_2(z)$ (surface density), and $v_2(z)$ (vector flux).
  - `Free Energy Phi`: Renders local excess free energy density profile $\Phi(z)$.
  - `Crossover Suite`: Renders zero-D cavity collapse stability comparing scalar divergence spikes in `RF`/`WB` vs bounded stability in `WB-Tensor`.
- **Diagnostic Viewport**: Toggle between `c^(1)(z)` Direct Correlation Profile and `log10 R(k)` Residual Convergence History curve.
- **Action Buttons**: `Solve` (toggle continuous solving), `Step 1 Iter` (advance by 1 iteration), `Reset` (reset density profile), `Show/Hide Benchmark` dots.
- **Sliders & Options**: Slit-pore vs single wall geometry, bulk packing fraction $\eta \in [0.01, 0.50]$, slit pore width $L_z \in [0.10\sigma, 15.0\sigma]$, grid spacing $dz \in [0.002\sigma, 0.010\sigma]$.

---

## 🧪 Demonstration Scripts & Testing

### Running CLI Step Demos
Execute individual demonstration scripts to inspect numerical outputs:

```bash
# Step 12 End-to-End Monte Carlo Regression & Throughput Benchmark Demo
uv run python scripts/demo_step12.py

# Step 10 Dimensional Crossover & Zero-D Confinement Demo
uv run python scripts/demo_step10.py

# Step 09 Thermodynamic Sum-Rules & Diagnostics Demo
uv run python scripts/demo_step09.py
```

### Running the Full Test Suite
Execute the 52 unit and performance benchmark tests:

```bash
uv run pytest -v
```

### Running Linter & Formatter
```bash
uv run ruff check .
uv run ruff format --check .
```

---

## 📁 Repository Structure

```text
cdft/
├── app_raylib.py        # Interactive Raylib 60 FPS Desktop GUI Visualizer & Solver
├── tutorial.md          # Interactive Masterclass Textbook, Companion Guide & Problem Sets
├── pyproject.toml       # Project metadata, dependencies (ruff, pytest, raylib), and linter config
├── README.md            # Comprehensive project documentation
├── scripts/             # Demonstration scripts for steps 01 through 12
│   ├── demo_step01.py   # Grid discretization & physical parameters demo
│   ├── demo_step02.py   # Analytical weight function integrals demo
│   ├── demo_step03.py   # FFT convolution engine & parity demo
│   ├── demo_step04.py   # Rosenfeld functional analytical derivatives demo
│   ├── demo_step05.py   # Fixed Picard solver step demo
│   ├── demo_step06.py   # Roth adaptive line-search Picard solver demo
│   ├── demo_step07.py   # White-Bear (WB) and WBII functional extensions demo
│   ├── demo_step08.py   # Tarazona tensorial FMT (WB-Tensor) demo
│   ├── demo_step09.py   # Thermodynamic sum-rules & diagnostics demo
│   ├── demo_step10.py   # Dimensional crossover & zero-D collapse demo
│   └── demo_step12.py   # Final end-to-end Monte Carlo regression & throughput demo
├── src/
│   ├── grid.py          # PhysicalParameters, Grid1D, external wall potentials & integration
│   ├── weights.py       # PlanarWeights (1D planar weights w0, w1, w2, w3, v1, v2, w_m2 & Simpson endpoint fixes)
│   ├── convolutions.py  # FFTConvolver1D (zero-padded 1D FFT convolutions)
│   ├── weighted_densities.py  # WeightedDensityCalculator & WeightedDensities dataclass
│   ├── crossover.py     # CrossoverAnalyzer (confinement sweeps & zero-D cavity Gaussian profiles)
│   ├── diagnostics.py   # SumRuleDiagnostics (contact theorem, surface tension, excess adsorption, Gibbs rule)
│   ├── functionals/     # FMT Functional Engine implementations
│   │   ├── base.py      # Abstract base class FMTFunctional
│   │   ├── rosenfeld.py # Rosenfeld (RF) functional
│   │   ├── white_bear.py# White-Bear (WB) & White-Bear II (WBII) functionals
│   │   └── tarazona_tensor.py # White-Bear Tensor (WB-Tensor) functional
│   ├── solvers/         # Picard Solver implementations
│   │   ├── base.py      # Base solver interface
│   │   ├── picard.py    # Fixed-step Picard solver
│   │   └── roth_picard.py # Roth adaptive line-search Picard solver
│   └── ui/              # Raylib GUI Components
│       ├── theme.py     # Dark mode color palette & styling constants
│       ├── widgets.py   # Clean GUI widgets (sliders, auto-fit buttons, radio grids, tooltips)
│       └── plotter.py   # 2D plot renderer, coordinate mapping, hover tooltips & dynamic legend box
└── tests/              # Automated Test Suite (52 tests)
    ├── test_benchmarks.py # End-to-end Monte Carlo regression & performance throughput tests
    ├── test_convolutions.py # FFT convolution & parity unit tests
    ├── test_crossover.py  # Dimensional crossover & zero-D cavity unit tests
    ├── test_diagnostics.py# Sum-rule diagnostics & contact theorem unit tests
    ├── test_grid.py       # Grid discretization & physical parameter unit tests
    ├── test_picard.py     # Fixed Picard solver unit tests
    ├── test_rosenfeld.py  # Rosenfeld functional derivative unit tests
    ├── test_roth_picard.py# Adaptive Roth Picard solver unit tests
    ├── test_tarazona_tensor.py # Tarazona tensorial FMT unit tests
    ├── test_ui.py         # GUI coordinate mapping & single-step execution unit tests
    ├── test_weighted_densities.py # Weighted density calculator unit tests
    ├── test_weights.py    # Planar weight function analytical integral unit tests
    └── test_white_bear.py # White-Bear & White-Bear II functional unit tests
```

---

## 📜 References

1. **Roth, R. (2010)**. *Fundamental measure theory for hard-sphere mixtures: a review*. Journal of Physics: Condensed Matter, 22(6), 063102.
