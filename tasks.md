# FMT Toy Simulator Implementation Tasks

## Step 01: Project Setup & Domain Discretization
- [ ] **Objective**: Establish project architecture, physical units, and 1D spatial grid.
- [ ] **Details**:
  - Set up Python environment with NumPy, SciPy, Matplotlib, and Pytest.
  - Define physical parameters: sphere diameter $\sigma = 1.0$, radius $R = 0.5$, bulk packing fraction $\eta$, temperature $\beta = 1.0$.
  - Create a 1D spatial grid along $z \in [0, L_z]$ with grid spacing $dz \le 0.01 \sigma$.
  - Implement external hard wall potential $V_{\text{ext}}(z)$: $\infty$ for $z < R$ or $z > L_z - R$, and $0$ otherwise.
- [ ] **Deliverable**: `src/grid.py` with passing tests for grid spacing and $V_{\text{ext}}(z)$ boundaries.

---

## Step 02: Planar Geometrical Weight Functions & FFT Convolution Module
- [ ] **Objective**: Implement 1D planar weight functions and FFT convolution engine with parity and high-order quadrature support.
- [ ] **Details**:
  - Implement scalar weight functions:
    $$\omega_3(z) = \pi(R^2 - z^2)\Theta(R - |z|)$$
    $$\omega_2(z) = 2\pi R \Theta(R - |z|)$$
    $$\omega_1(z) = \frac{\omega_2(z)}{4\pi R}, \quad \omega_0(z) = \frac{\omega_2(z)}{4\pi R^2}$$
  - Implement vector weight function (odd parity):
    $$w_2^z(z) = 2\pi z \Theta(R - |z|), \quad w_1^z(z) = \frac{w_2^z(z)}{4\pi R}$$
  - Implement FFT-based convolution utility handling zero-padding to prevent periodic boundary wrap-around.
  - Account for parity in functional derivatives ($c^{(1)}$ convolutions): scalar weights are even ($\omega(-z) = \omega(z)$), vector weights are odd ($w^z(-z) = -w^z(z)$).
  - Add optional Section 8.4 endpoint weight modifications ($\frac{3}{8}, \frac{7}{6}, \frac{23}{24}$ factors at boundaries) for $\mathcal{O}(dz^4)$ Fourier-space Simpson quadrature accuracy.
- [ ] **Deliverable**: `src/weights.py` and `src/convolutions.py` with unit tests verifying analytical integrals ($\int \omega_3 dz = \frac{4}{3}\pi R^3$, $\int \omega_2 dz = 4\pi R^2$).

---

## Step 03: Weighted Density Calculator
- [ ] **Objective**: Compute spatial weighted densities $n_\alpha(z)$ from a density profile $\rho(z)$.
- [ ] **Details**:
  - Implement functions to evaluate:
    - Scalar weighted densities: $n_0(z), n_1(z), n_2(z), n_3(z)$.
    - Vector weighted density z-component: $v_1(z), v_2(z)$.
  - Ensure $n_3(z)$ represents local packing fraction and verify $n_3(z) < 1$ across the domain.
  - Verify bulk limit values against Scaled Particle Theory (SPT) variables: $n_3 \to \eta$, $n_2 \to \rho_{\text{bulk}} 4\pi R^2$, $n_1 \to \rho_{\text{bulk}} R$, $n_0 \to \rho_{\text{bulk}}$, and $v_1, v_2 \to 0$.
- [ ] **Deliverable**: `src/weighted_densities.py` returning all 6 weighted density arrays for a given input $\rho(z)$.

---

## Step 04: Rosenfeld (RF) Excess Free Energy Density & Partial Derivatives
- [ ] **Objective**: Implement original Rosenfeld FMT functional density $\Phi^{\text{RF}}$ and derivatives $\frac{\partial \Phi}{\partial n_\alpha}$.
- [ ] **Details**:
  - Implement free energy density:
    $$\Phi^{\text{RF}} = -n_0 \ln(1 - n_3) + \frac{n_1 n_2 - v_1 v_2}{1 - n_3} + \frac{n_2^3 - 3 n_2 v_2^2}{24\pi(1 - n_3)^2}$$
  - Calculate analytic derivatives $\frac{\partial \Phi^{\text{RF}}}{\partial n_0}, \dots, \frac{\partial \Phi^{\text{RF}}}{\partial v_2}$.
  - Compute total excess free energy $F_{\text{ex}} = \int \Phi(z) dz$.
- [ ] **Deliverable**: `src/functionals/rosenfeld.py` with unit tests checking derivatives via finite differences.

---

## Step 05: One-Body Direct Correlation $c^{(1)}(z)$ & Basic Picard Solver
- [ ] **Objective**: Build the functional derivative $c^{(1)}(z)$ and a fixed-step Picard solver.
- [ ] **Details**:
  - Implement one-body direct correlation function:
    $$c^{(1)}(z) = -\sum_\alpha \int dz' \frac{\partial \Phi}{\partial n_\alpha}(z') \omega_\alpha(z' - z)$$
    using FFT convolutions with appropriate parity sign flips for vector components.
  - Implement basic Picard update loop:
    $$\tilde{\rho}^{(j)}(z) = \rho_{\text{bulk}} \exp\left(-\beta V_{\text{ext}}(z) + c^{(1)}(z) + \beta \mu_{\text{ex}}\right)$$
    $$\rho^{(j+1)}(z) = (1 - \alpha)\rho^{(j)}(z) + \alpha \tilde{\rho}^{(j)}(z)$$
    where $\beta \mu_{\text{ex}}$ is the functional-specific bulk excess chemical potential.
- [ ] **Deliverable**: `src/solver.py` executing basic density profile relaxation at a hard wall.

---

## Step 06: Roth's Adaptive Line-Search Mixing Scheme
- [ ] **Objective**: Implement self-adjusting step-size $\alpha_{\text{opt}}$ based on grand potential minimization (Roth Sec. 8.1 / Fig. 7).
- [ ] **Details**:
  - Calculate physical grand potential $\Omega[\rho(z)]$:
    $$\Omega = F_{\text{ideal}} + F_{\text{ex}} + \int \rho(z)(V_{\text{ext}}(z) - \mu) dz$$
  - Calculate $\alpha_{\text{max}}$ to ensure $\max_z n_3(z) < 1.0$ after mixing.
  - Perform 3-point evaluation of grand potential:
    - $\Omega_1 = \Omega(\alpha=0)$
    - $\Omega_2 = \Omega(0.45\alpha_{\text{max}})$
    - If $\Omega_2 \le \Omega_1$, evaluate $\Omega_3 = \Omega(0.90\alpha_{\text{max}})$; else evaluate $\Omega_3 = \Omega(0.225\alpha_{\text{max}})$.
  - Fit a quadratic polynomial to $(\alpha_i, \Omega_i)$ for $i=1,2,3$ to find the optimal mixing step $\alpha_{\text{opt}} \in [0, 0.9\alpha_{\text{max}}]$.
- [ ] **Deliverable**: Extended `src/solver.py` with robust convergence within ~100-200 iterations for dense packing fractions ($\eta = 0.45$).

---

## Step 07: White-Bear (WB) & White-Bear Mark II Functional Extensions
- [x] **Objective**: Add WB and WBII functionals with low-density Taylor expansions for numerical stability.
- [x] **Details**:
  - Implement White-Bear term $f_4^{\text{WB}}(n_3)$:
    $$f_4^{\text{WB}}(n_3) = \frac{n_3 + (1 - n_3)^2 \ln(1 - n_3)}{36\pi n_3^2 (1 - n_3)^2}$$
  - Implement low-density Taylor expansion for $f_4^{\text{WB}}(n_3)$ when $n_3 < 10^{-4}$ ($f_4^{\text{WB}}(n_3) = \frac{1}{24\pi}(1 + \frac{2}{3}n_3 + \frac{1}{2}n_3^2 + \dots)$) to prevent $0/0$ indeterminate forms / `NaN` errors.
  - Implement White-Bear Mark II auxiliary functions $\phi_2(n_3)$ and $\phi_3(n_3)$ along with their low-density series expansions.
  - Implement exact functional-specific bulk excess chemical potentials $\beta \mu_{\text{ex}}^{\text{WB}}$ and $\beta \mu_{\text{ex}}^{\text{WBII}}$.
  - Add modular switching between `RF`, `WB`, and `WBII` functionals in the solver engine.
- [x] **Deliverable**: `src/functionals/white_bear.py` and modular factory interface in `src/functionals/__init__.py`.

---

## Step 08: Tarazona Tensorial Weights & Tensorial FMT Engine
- [ ] **Objective**: Implement tensorial weight function $\omega_{m2}$ and exact 1D planar Tarazona tensor functional.
- [ ] **Details**:
  - Implement 1D planar scalar component of tensorial weight function $\omega_{m2}(z) = (\frac{z^2}{R^2} - \frac{1}{3})\omega_2(z)$.
  - Compute tensorial weighted density $n_{m2}(z) = \int \rho(z') \omega_{m2}(z - z') dz'$.
  - Implement exact 1D reduced Tarazona modified 3rd term (accounting for uniaxial trace reduction $\frac{1}{2}\mathrm{Tr}(\mathbf{n}_{m2}^3) = \frac{3}{8}n_{m2}^3$):
    $$\Phi_3^{\text{Tensor}} = \frac{n_2^3 - 3n_2 v_2^2 + 9(v_2^2 n_{m2} - \frac{3}{8} n_{m2}^3)}{24\pi(1 - n_3)^2}$$
  - Combine tensor terms with White-Bear equation of state prefactors to implement `WB-Tensor` functional (Roth Eq. 378).
- [ ] **Deliverable**: `src/functionals/tarazona_tensor.py` capable of handling tight confinement without numerical overflow.

---

## Step 09: Thermodynamic Observables & Sum-Rule Validation
- [ ] **Objective**: Calculate wall surface tension and verify contact theorem and Gibbs adsorption sum-rules.
- [ ] **Details**:
  - Compute wall surface tension: $\gamma = \frac{1}{A}(\Omega + P_{\text{bulk}} V)$.
  - Validate surface tension against analytical bulk-route formula: $\beta \gamma_{\text{bulk}} = \left.\frac{\partial \Phi}{\partial n_2}\right|_{\text{bulk}}$.
  - Verify contact theorem to 4 significant figures: $\rho(z = R^+) = \beta P_{\text{bulk}}$.
  - Verify Gibbs adsorption theorem: $\Gamma = \int (\rho(z) - \rho_{\text{bulk}}) dz = -\frac{\partial \gamma}{\partial \mu}$.
- [ ] **Deliverable**: `src/diagnostics.py` printing sum-rule accuracy metrics.

---

## Step 10: Dimensional Crossover & Collapse Visualizer Engine
- [ ] **Objective**: Create a confinement test suite to demonstrate free energy divergence vs stability under dimensional reduction.
- [ ] **Details**:
  - Set up a slit-pore confinement scenario with adjustable width $L \in [0.1\sigma, 3.0\sigma]$.
  - Force extreme confinement / delta-like density distribution.
  - Compute local free energy density profiles $\Phi(z)$ for `RF`, `WB`, and `WB-Tensor`.
  - Capture divergence spikes ($1/(1-n_3)^2$ blowing up) in standard scalar versions vs well-behaved curves in Tensor FMT.
- [ ] **Deliverable**: `src/crossover.py` producing comparative collapse/divergence plots.

---

## Step 11: Real-Time Interactive Solver & Physics Visualization Application (Raylib / Python)
- [ ] **Objective**: Build a high-performance desktop GUI using Raylib (`pyray`) for interactive problem solving, real-time solver animation, and confinement collapse visualization.
- [ ] **Details**:
  - Implement Raylib-based desktop interface:
    - Interactive parameter controls: FMT functional selection (`RF`, `WB`, `WBII`, `WB-Tensor`), bulk packing fraction $\eta \in [0.01, 0.50]$, pore width $L \in [0.1\sigma, 15.0\sigma]$, grid spacing $dz$, and geometry mode (single wall vs slit pore).
    - Real-time plotting viewport: Render continuous density profile $\rho(z)$, weighted densities $n_\alpha(z)$, and direct correlation $c^{(1)}(z)$ curves.
    - Solver controls: Start/Pause/Step iteration loop, live grand potential $\Omega$ convergence graph, and mixing parameter $\alpha_{\text{opt}}$ meter.
    - Dimensional Collapse Mode: Interactive pore width $L$ slider demonstrating divergence in scalar FMT vs stability in Tensorial FMT.
    - Benchmark overlay: Display published Monte Carlo simulation benchmark points ($\eta = 0.4257, 0.4783$).
- [ ] **Deliverable**: `app_raylib.py` launching a Raylib/Python desktop application.

---

## Step 12: End-to-End Test Suite & Benchmarking
- [ ] **Objective**: Automated testing and verification against published Monte Carlo benchmark data.
- [ ] **Details**:
  - Add regression test suite comparing computed contact densities against published MC benchmark values (e.g., figure 1 in Roth 2010 for $\eta = 0.4257, 0.4783$).
  - Benchmark performance: Ensure FFT convolutions and Picard iterations complete under 5 seconds for 1000 grid points.
- [ ] **Deliverable**: Complete, fully documented repository with `pytest` suite passing 100%.
