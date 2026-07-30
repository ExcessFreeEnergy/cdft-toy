# Classical Density Functional Theory (FMT) Interactive Masterclass & Tutorial

A Comprehensive Hands-On Companion to Roland Roth's Topical Review:  
**"Fundamental measure theory for hard-sphere mixtures: a review"** (*J. Phys.: Condens. Matter 22, 2010, 063102*)

---

## Welcome & Overview

Welcome to the **Classical Density Functional Theory (cDFT) & Fundamental Measure Theory (FMT) Interactive Masterclass**. 

This tutorial serves as a self-contained theoretical textbook, computational laboratory manual, and problem-set companion for mastering Classical Density Functional Theory (cDFT) as formulated by Yasha Rosenfeld (1989), Tarazona (2000), and Roland Roth (2010).

### Software Lab Tools Included
The codebase provides three primary interactive lab environments to run the experiments described in this tutorial:
1. **Interactive Physics Visualizer (`app_raylib.py`)**: A high-performance Raylib desktop application for real-time solver animation, spatial density profile visualization, diagnostic sum-rule metrics, and dimensional collapse exploration.
2. **Step Demonstration Scripts (`scripts/demo_step01.py` through `demo_step12.py`)**: Executable Python scripts demonstrating individual physical modules, convolution passes, and benchmark tables.
3. **Automated Test Suite (`uv run pytest`)**: A suite of 52 unit and performance benchmark tests verifying mathematical derivatives, sum-rules, and throughput.

---

## Module 1: The Geometry & Fundamentals of FMT

### 1.1 Theoretical Deep Dive (Roth 2010 Sec. 2 & Sec. 3)

Classical Density Functional Theory states that for an inhomogeneous fluid of hard spheres in an external potential $V_{\text{ext}}(\mathbf{r})$, there exists a unique grand potential functional:

$$\Omega[\rho] = F_{\text{id}}[\rho] + F_{\text{ex}}[\rho] + \int \rho(\mathbf{r}) [V_{\text{ext}}(\mathbf{r}) - \mu] d\mathbf{r}$$

where the ideal gas free energy is known exactly:

$$F_{\text{id}}[\rho] = k_B T \int \rho(\mathbf{r}) [\ln(\Lambda^3 \rho(\mathbf{r})) - 1] d\mathbf{r}$$

The core challenge of statistical mechanics is approximating the **excess free energy functional** $F_{\text{ex}}[\rho]$ resulting from hard-sphere volume exclusion.

In **Fundamental Measure Theory (FMT)**, the excess free energy is expressed as the spatial integral of a local excess free energy density $\Phi(\mathbf{r})$:

$$F_{\text{ex}}[\rho] = k_B T \int \Phi(\{n_\alpha(\mathbf{r})\}) d\mathbf{r}$$

where $\Phi(\mathbf{r})$ depends on a set of **weighted densities** $n_\alpha(\mathbf{r})$ obtained by convoluting the spatial density profile $\rho(\mathbf{r})$ with geometric weight functions $w_\alpha(\mathbf{r})$ characteristic of single hard spheres:

$$n_\alpha(\mathbf{r}) = \int \rho(\mathbf{r}') w_\alpha(\mathbf{r} - \mathbf{r}') d\mathbf{r}' = (\rho * w_\alpha)(\mathbf{r})$$

For 3D hard spheres of radius $R = \sigma/2$, the six Rosenfeld weight functions consist of four scalars ($w_3, w_2, w_1, w_0$) and two vectors ($\mathbf{w}_2, \mathbf{w}_1$):

| Weight Name | Geometric Meaning | Mathematical Form |
| :--- | :--- | :--- |
| $w_3(\mathbf{r})$ | Sphere Volume Characteristic | $\Theta(R - \vert\mathbf{r}\vert)$ |
| $w_2(\mathbf{r})$ | Sphere Surface Area Characteristic | $\delta(R - \vert\mathbf{r}\vert)$ |
| $w_1(\mathbf{r})$ | Sphere Radius Characteristic | $\frac{w_2(\mathbf{r})}{4\pi R}$ |
| $w_0(\mathbf{r})$ | Euler Characteristic / Number Density | $\frac{w_2(\mathbf{r})}{4\pi R^2}$ |
| $\mathbf{w}_2(\mathbf{r})$ | Directed Surface Normal Vector | $\frac{\mathbf{r}}{\vert\mathbf{r}\vert} \delta(R - \vert\mathbf{r}\vert)$ |
| $\mathbf{w}_1(\mathbf{r})$ | Directed Surface Vector | $\frac{\mathbf{w}_2(\mathbf{r})}{4\pi R}$ |

#### 1D Planar Geometry Reduction
In a 1D planar system where density varies only along $z$ ($\rho(\mathbf{r}) = \rho(z)$), the 3D surface integrals reduce analytically to 1D planar weight arrays over $z \in [-R, R]$:

- **Volume weight**: $w_3(z) = \pi (R^2 - z^2)$
- **Surface weight**: $w_2(z) = 2\pi R$
- **Vector weight**: $v_2(z) = 2\pi z$ (odd parity under $z \to -z$)

---

### 1.2 Interactive Lab Test 1: Weight Integrals & Parity Inversion

#### Experiment 1.1: Verification of Analytical Weight Integrals
Run the weight function test script to verify that discrete numerical quadrature of 1D planar weights matches exact analytical geometry:

```bash
uv run pytest tests/test_weights.py -v
```

**Expected Outcome**:
- $\int_{-R}^R w_3(z) dz = \frac{4}{3}\pi R^3 = v_{\text{sphere}}$
- $\int_{-R}^R w_2(z) dz = 4\pi R^2 = s_{\text{sphere}}$
- $\int_{-R}^R v_2(z) dz = 0$ (due to odd vector parity under $z \to -z$)

#### Experiment 1.2: Vector Flux Inversion
Run `scripts/demo_step03.py` to observe Fourier-space convolutions and sign inversion of $v_2(z)$:

```bash
uv run python scripts/demo_step03.py
```

Observe that near a hard wall at $z = 0$, $v_2(z) > 0$ (pointing away from the wall), reflecting asymmetrical surface flux.

---

### 1.3 Problem Set 1: Geometry & Weight Calculus

#### Problem 1.1 (Analytical Weight Proof)
Prove by integration in cylindrical coordinates $(r_\perp, \theta, z)$ that the 3D surface delta weight $w_2(\mathbf{r}) = \delta(R - \vert\mathbf{r}\vert)$ reduces in 1D planar geometry to $w_2(z) = 2\pi R$ for $\vert z\vert \le R$ and $0$ otherwise.

#### Problem 1.2 (Bulk Limits of Weighted Densities)
For a uniform bulk fluid with constant density $\rho(z) = \rho_{\text{bulk}}$ and packing fraction $\eta = \frac{\pi}{6} \rho_{\text{bulk}} \sigma^3$, evaluate the bulk values of all six weighted densities: $n_3^{\text{bulk}}, n_2^{\text{bulk}}, n_1^{\text{bulk}}, n_0^{\text{bulk}}, v_1^{\text{bulk}}, v_2^{\text{bulk}}$.

*Solutions are provided in Section 8.*

---

## Module 2: Scaled Particle Theory (SPT) & The Percus-Yevick Fluid

### 2.1 Theoretical Deep Dive (Roth 2010 Sec. 4.1)

Yasha Rosenfeld (1989) discovered that dimensional consistency and Scaled Particle Theory (SPT) uniquely constrain the functional form of the excess free energy density $\Phi(\mathbf{n})$ to a sum of three scaled terms:

$$\Phi^{\text{RF}} = \Phi_1^{\text{RF}} + \Phi_2^{\text{RF}} + \Phi_3^{\text{RF}}$$

$$\Phi_1^{\text{RF}} = -n_0 \ln(1 - n_3)$$

$$\Phi_2^{\text{RF}} = \frac{n_1 n_2 - \mathbf{n}_1 \cdot \mathbf{n}_2}{1 - n_3}$$

$$\Phi_3^{\text{RF}} = \frac{n_2^3 - 3 n_2 \mathbf{n}_2 \cdot \mathbf{n}_2}{24\pi (1 - n_3)^2}$$

#### Physical Insights & Properties
1. **Low-Density Recovery**: As $n_3 \to 0$, $-\ln(1-n_3) \to n_3$, and $\Phi^{\text{RF}} \to n_0 n_3 + n_1 n_2 + \frac{n_2^3}{24\pi}$, exactly recovering the second cluster virial coefficient $B_2 = 4 v_{\text{sphere}}$.
2. **Percus-Yevick Equation of State**: Evaluating $\Phi^{\text{RF}}$ in uniform bulk fluid ($\mathbf{n}_1 = \mathbf{n}_2 = 0$) yields the exact Percus-Yevick (PY) compressibility equation of state:

$$\beta P_{\text{PY,comp}} = \left. \left( n_3 \frac{\partial \Phi}{\partial n_3} + n_2 \frac{\partial \Phi}{\partial n_2} + n_1 \frac{\partial \Phi}{\partial n_1} + n_0 \frac{\partial \Phi}{\partial n_0} - \Phi \right) \right|_{\text{bulk}} = \rho_{\text{bulk}} \frac{1 + \eta + \eta^2}{(1 - \eta)^3}$$

---

### 2.2 Interactive Lab Test 2: Rosenfeld Functional Derivatives & Bulk EOS

#### Experiment 2.1: Finite-Difference Derivative Verification
Run `tests/test_rosenfeld.py` to verify analytical partial derivatives $\frac{\partial \Phi}{\partial n_\alpha}$ against central finite-difference gradients:

```bash
uv run pytest tests/test_rosenfeld.py -v
```

#### Experiment 2.2: Live Visualizer Physics Setup
1. Launch the interactive Raylib application:
   ```bash
   uv run python app_raylib.py
   ```
2. Set **FMT Functional Variant** to `RF (Original)`.
3. Set **Bulk Packing Fraction ($\eta$)** to `0.4257`.
4. Observe the primary density profile plot $\rho(z)$ forming characteristic packing shells near the hard wall at $z = 0$.

---

### 2.3 Problem Set 2: Scaled Particle Thermodynamics

#### Problem 2.1 (Bulk PY Pressure Derivation)
Using $\Phi^{\text{RF}}_{\text{bulk}} = -n_0 \ln(1 - n_3) + \frac{n_1 n_2}{1 - n_3} + \frac{n_2^3}{24\pi (1 - n_3)^2}$, substitute the bulk weighted densities as functions of $\eta$ and derive the PY bulk pressure $\beta P_{\text{PY,comp}}(\eta)$.

#### Problem 2.2 (Bulk Excess Chemical Potential)
Derive the bulk excess chemical potential $\beta \mu_{\text{ex}}^{\text{PY}} = \left. \frac{\partial \Phi^{\text{RF}}}{\partial \rho} \right|_{\text{bulk}}$.

---

## Module 3: White-Bear & White-Bear II Functionals

### 3.1 Theoretical Deep Dive (Roth 2010 Sec. 4.3)

While the Rosenfeld (RF) functional accurately describes hard-sphere fluids at low to moderate packing fractions, its PY compressibility equation of state underestimates bulk pressure at high densities ($\eta > 0.40$). The exact Carnahan-Starling (CS) equation of state provides superior agreement with Monte Carlo simulations:

$$\beta P_{\text{CS}} = \rho_{\text{bulk}} \frac{1 + \eta + \eta^2 - \eta^3}{(1 - \eta)^3}$$

To embed the Carnahan-Starling EOS into cDFT, Roth et al. developed the **White-Bear (WB)** and **White-Bear II (WBII)** functionals.

#### White-Bear (WB) Functional
The WB functional replaces the third term $\Phi_3$ with a modified function $f_4^{\text{WB}}(n_3)$:

$$\Phi_3^{\text{WB}} = (n_2^3 - 3 n_2 \mathbf{n}_2 \cdot \mathbf{n}_2) f_4^{\text{WB}}(n_3)$$

$$f_4^{\text{WB}}(n_3) = \frac{n_3 + (1 - n_3)^2 \ln(1 - n_3)}{36\pi n_3^2 (1 - n_3)^2}$$

#### Low-Density Taylor Series Expansion
As $n_3 \to 0$, $f_4^{\text{WB}}(n_3)$ encounters an indeterminate form $\frac{0}{0}$. To ensure numerical stability, the software evaluates the Taylor series for $n_3 < 10^{-3}$:

$$f_4^{\text{WB}}(n_3) = \frac{1}{24\pi} \left( 1 + \frac{4}{3} n_3 + \frac{3}{2} n_3^2 + \frac{8}{5} n_3^3 + O(n_3^4) \right)$$

---

### 3.2 Interactive Lab Test 3: High-Density Hard Wall Adsorption

#### Experiment 3.1: Comparative Functional Benchmarking
Run `scripts/demo_step07.py` to compare bulk pressure and surface contact density across `RF`, `WB`, and `WBII` at $\eta = 0.4783$:

```bash
uv run python scripts/demo_step07.py
```

**Observed Results**:
- `RF` Contact Density: $\rho(R^+) = 10.5678$ (matches PY pressure)
- `WB` / `WBII` Contact Density: $\rho(R^+) = 9.9233$ (matches CS pressure)

---

### 3.3 Problem Set 3: Carnahan-Starling FMT Extensions

#### Problem 3.1 (Low-Density Series Derivation)
Perform L'Hôpital's rule or Taylor expansion on $f_4^{\text{WB}}(n_3) = \frac{n_3 + (1-n_3)^2 \ln(1-n_3)}{36\pi n_3^2 (1-n_3)^2}$ to prove that $\lim_{n_3 \to 0} f_4^{\text{WB}}(n_3) = \frac{1}{24\pi}$.

---

## Module 4: Dimensional Crossover & Tarazona Tensorial FMT

### 4.1 Theoretical Deep Dive (Roth 2010 Sec. 4.2 & Sec. 4.4)

#### The Dimensional Collapse Problem
A fundamental test of a physical density functional is **dimensional crossover**: restricting the 3D density profile to a line ($\rho(\mathbf{r}) = \delta(x)\delta(y)\rho^{(1D)}(z)$) or cavity should smoothly recover lower-dimensional physics.

For scalar FMT functionals (`RF`, `WB`, `WBII`), under extreme confinement or zero-dimensional (0D) single-particle cavities, $\mathbf{v}_2 \to n_2$, causing:

$$n_2^3 - 3 n_2 \mathbf{v}_2 \cdot \mathbf{v}_2 \to -2 n_2^3$$

The third term $\Phi_3 \propto \frac{n_2^3}{(1-n_3)^2}$ diverges non-physically, causing solver failure or numerical collapse!

#### Tarazona Tensorial FMT Solution
Tarazona (2000) resolved this divergence by introducing a 1D planar tensorial weight function:

$$\omega_{m2}(z) = 2\pi R \left( \frac{z^2}{R^2} - \frac{1}{3} \right), \quad z \in [-R, R]$$

which generates the tensorial weighted density $n_{m2}(z)$.

In 1D planar geometry, the uniaxial tensorial trace cancels the scalar divergence:

$$\Phi_3^{\text{Tensor}} = \frac{n_2^3 - 3 n_2 v_2^2 + 9 \left( v_2^2 n_{m2} - \frac{3}{8} n_{m2}^3 \right)}{24\pi (1 - n_3)^2}$$

Under zero-dimensional cavity collapse, the bracketed numerator vanishes identically ($0/0$ cancellation), preventing free energy divergence!

---

### 4.2 Interactive Lab Test 4: Zero-D Cavity Collapse Visualizer

#### Experiment 4.1: Crossover Suite Mode in Raylib GUI
1. Launch `app_raylib.py`.
2. Under **Plot Viewport Mode**, select **`Crossover Suite`**.
3. Observe the live comparative plot of peak free energy density $\max \Phi(z)$ vs cavity width $\alpha$:
   - **Scalar Functionals (`RF`, `WB`)**: Display steep divergence spikes (Red / Amber curves).
   - **WB-Tensor (Blue curve)**: Remains bounded and finite.

#### Experiment 4.2: Slit-Pore Confinement Sweep
Run `scripts/demo_step10.py` to inspect numerical metrics across narrow slit pores ($L_z = 0.60\sigma$ to $2.50\sigma$):

```bash
uv run python scripts/demo_step10.py
```

---

### 4.3 Problem Set 4: Tensorial Reduction & Trace Calculus

#### Problem 4.1 (1D Uniaxial Trace Reduction)
Given the 3D tensorial weighted density matrix in 1D planar geometry:

$$\mathbf{n}_{m2}(z) = n_{m2}(z) \begin{pmatrix} -1/2 & 0 & 0 \\ 0 & -1/2 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Prove that $\frac{1}{2} \text{Tr}(\mathbf{n}_{m2}^3) = \frac{3}{8} n_{m2}^3$ and $\mathbf{v}_2 \cdot \mathbf{n}_{m2} \cdot \mathbf{v}_2 = v_2^2 n_{m2}$.

---

## Module 5: Numerical Implementation, FFT Convolutions & Picard Iterations

### 5.1 Theoretical Deep Dive (Roth 2010 Sec. 8)

#### Convolution via Fast Fourier Transform (FFT)
Direct spatial convolution is $O(N^2)$. Utilizing the Convolution Theorem, spatial convolutions are computed in $O(N \log N)$ time:

$$n_\alpha(z) = \mathcal{F}^{-1} \left[ \mathcal{F}[\rho(z)] \cdot \mathcal{F}[w_\alpha(z)] \right]$$

To eliminate periodic boundary wraparound artifacts, arrays are padded to $N_{\text{fft}} = N_{\text{grid}} + N_w - 1$.

#### High-Order Simpson Endpoint Modifications
To correct discrete grid quadrature errors at weight sphere boundaries $z = \pm R$, Section 8.4 endpoint weights are applied:

- Endpoints ($z = \pm R$): multiply by $3/8$
- Index 1 ($z = \pm (R - dz)$): multiply by $7/6$
- Index 2 ($z = \pm (R - 2dz)$): multiply by $23/24$

#### Roth Adaptive Line-Search Picard Solver
Standard fixed Picard mixing $\rho^{(k+1)} = (1-\alpha)\rho^{(k)} + \alpha \rho_{\text{target}}$ diverges at high packing fractions ($\eta > 0.40$).

Roth's adaptive Picard solver calculates the optimal scalar mixing parameter $\alpha_{\text{opt}}$ at each iteration:

$$\alpha_{\text{opt}} = -\frac{\int \Delta \rho_{\text{in}}(z) \Delta \rho_{\text{out}}(z) dz}{\int [\Delta \rho_{\text{out}}(z)]^2 dz}$$

where $\Delta \rho_{\text{in}} = \rho^{(k)} - \rho^{(k-1)}$ and $\Delta \rho_{\text{out}} = (\rho_{\text{target}}^{(k)} - \rho^{(k)}) - (\rho_{\text{target}}^{(k-1)} - \rho^{(k-1)})$.

---

### 5.2 Interactive Lab Test 5: Solver Performance & Single-Stepping

#### Experiment 5.1: Single-Step Iteration Debugging
1. Launch `app_raylib.py`.
2. Click **`Show R(k) History`** in the lower diagnostic panel.
3. Click **`Step 1 Iter`** repeatedly to advance the solver frame-by-frame and observe the exponential decay of the residual norm $\log_{10} R(k)$.

#### Experiment 5.2: Speedup Verification
Run `scripts/demo_step06.py` to compare convergence speed of Fixed Picard vs Roth Adaptive Picard:

```bash
uv run python scripts/demo_step06.py
```

---

### 5.3 Problem Set 5: Picard Convergence Mechanics

#### Problem 5.1 (Optimal Mixing Derivation)
Derive the formula for $\alpha_{\text{opt}}$ by minimizing the norm of the expected next-iteration residual $E(\alpha) = \int [\Delta \rho_{\text{in}} + \alpha \Delta \rho_{\text{out}}]^2 dz$ with respect to $\alpha$.

---

## Module 6: Thermodynamic Sum-Rules & Diagnostics

### 6.1 Theoretical Deep Dive (Roth 2010 Sec. 2 & Sec. 8)

Four exact thermodynamic sum-rules validate cDFT solutions:

1. **Wall Contact Theorem**:
   $$\rho(R^+) = \beta P_{\text{bulk}}$$
2. **Spatial Surface Tension Integration**:
   $$\beta \gamma = \int_0^{L_z} [\omega(z) + \beta P_{\text{bulk}}] dz$$
3. **Bulk-Route Analytical Surface Tension**:
   $$\beta \gamma_{\text{bulk}} = \left. \frac{\partial \Phi}{\partial n_2} \right|_{\text{bulk}}$$
4. **Gibbs Adsorption Theorem**:
   $$-\frac{d\gamma}{d\mu} = \Gamma = \int_0^{L_z} (\rho(z) - \rho_{\text{bulk}}) dz$$

---

### 6.2 Interactive Lab Test 6: Sum-Rule Validation Suite

#### Experiment 6.1: Real-Time Sum-Rule Diagnostics
Launch `app_raylib.py` and inspect the sidebar panel **System Thermodynamics & Info**:
- Verify that **Contact $\rho(R^+)$** agrees with **Bulk Pressure** to $< 0.3\%$ relative error.

#### Experiment 6.2: Gibbs Adsorption Consistency
Run `scripts/demo_step09.py` to test Gibbs adsorption theorem consistency:

```bash
uv run python scripts/demo_step09.py
```

---

### 6.3 Problem Set 6: Sum-Rule Derivations

#### Problem 6.1 (Contact Theorem Proof)
By integrating the hydrostatic force balance equation $\frac{dP}{dz} = -\rho(z) \frac{dV_{\text{ext}}}{dz}$ across a hard wall at $z=0$, prove that $\rho(R^+) = \beta P_{\text{bulk}}$.

---

## Module 7: Full End-to-End Performance Benchmarking

Run the complete benchmark suite to verify performance and Monte Carlo regression:

```bash
uv run python scripts/demo_step12.py
```

**Target Benchmarks**:
- **FFT Convolutions**: $> 5000$ convolutions/sec
- **Picard Iterations**: $> 1000$ iterations/sec
- **Contact Density Regression**: Matches published Monte Carlo values in Roth (2010) Fig. 1.

---

## Module 8: Solutions to Problem Sets

### Solution to Problem 1.1 (Analytical Weight Proof)
In cylindrical coordinates $(r_\perp, \theta, z)$, $\mathbf{r}^2 = r_\perp^2 + z^2$.  
$w_2(z) = \int \delta(R - \sqrt{r_\perp^2 + z^2}) r_\perp dr_\perp d\theta = 2\pi \int_0^\infty \delta(R - \sqrt{r_\perp^2 + z^2}) r_\perp dr_\perp$.  
Let $u = \sqrt{r_\perp^2 + z^2}$, so $du = \frac{r_\perp}{u} dr_\perp \implies r_\perp dr_\perp = u du$.  
For $\vert z\vert \le R$, $u$ ranges from $\vert z\vert$ to $\infty$.  
$w_2(z) = 2\pi \int_{\vert z\vert}^\infty \delta(R - u) u du = 2\pi R$. For $\vert z\vert > R$, the delta function root lies outside the integration range, so $w_2(z) = 0$. $\blacksquare$

### Solution to Problem 1.2 (Bulk Weighted Densities)
For uniform $\rho(z) = \rho_{\text{bulk}}$:
- $n_3^{\text{bulk}} = \rho_{\text{bulk}} v_{\text{sphere}} = \frac{4}{3}\pi R^3 \rho_{\text{bulk}} = \eta$
- $n_2^{\text{bulk}} = \rho_{\text{bulk}} s_{\text{sphere}} = 4\pi R^2 \rho_{\text{bulk}} = \frac{6\eta}{R} = \frac{12\eta}{\sigma}$
- $n_1^{\text{bulk}} = \frac{n_2^{\text{bulk}}}{4\pi R} = R \rho_{\text{bulk}} = \frac{3\eta}{\pi \sigma^2}$
- $n_0^{\text{bulk}} = \frac{n_2^{\text{bulk}}}{4\pi R^2} = \rho_{\text{bulk}} = \frac{6\eta}{\pi \sigma^3}$
- $v_1^{\text{bulk}} = v_2^{\text{bulk}} = 0$ (due to spherical symmetry in bulk). $\blacksquare$

### Solution to Problem 2.1 (Bulk PY Pressure Derivation)
In bulk fluid, $\mathbf{n}_1 = \mathbf{n}_2 = 0$. Substituting $n_0, n_1, n_2, n_3$ as functions of $\eta$:  
$\Phi_{\text{bulk}}^{\text{RF}} = -\frac{6\eta}{\pi\sigma^3} \ln(1-\eta) + \frac{18\eta^2}{\pi\sigma^3 (1-\eta)} + \frac{36\eta^3}{\pi\sigma^3 (1-\eta)^2}$.  
Differentiating $\beta P = \eta \frac{\partial \Phi}{\partial \eta} - \Phi$ yields:  
$\beta P_{\text{PY}} = \rho_{\text{bulk}} \frac{1 + \eta + \eta^2}{(1 - \eta)^3}$. $\blacksquare$

### Solution to Problem 3.1 (Low-Density Series Derivation)
Expand $\ln(1-n_3) = -n_3 - \frac{n_3^2}{2} - \frac{n_3^3}{3} - O(n_3^4)$ for small $n_3$:  
$(1-n_3)^2 \ln(1-n_3) = (1 - 2n_3 + n_3^2)(-n_3 - \frac{n_3^2}{2} - \frac{n_3^3}{3}) = -n_3 + \frac{3}{2}n_3^2 + \frac{1}{6}n_3^3 + O(n_3^4)$.  
Numerator: $n_3 + (1-n_3)^2 \ln(1-n_3) = \frac{3}{2}n_3^2 + \frac{1}{6}n_3^3 + O(n_3^4)$.  
Denominator: $36\pi n_3^2 (1-n_3)^2 = 36\pi n_3^2 (1 - 2n_3 + O(n_3^2))$.  
Dividing numerator by denominator:  
$f_4^{\text{WB}}(n_3) = \frac{\frac{3}{2}n_3^2 + \frac{1}{6}n_3^3}{36\pi n_3^2 (1 - 2n_3)} = \frac{1}{24\pi} \left( 1 + \frac{4}{3}n_3 + O(n_3^2) \right)$.  
Taking $n_3 \to 0$ yields $\frac{1}{24\pi}$. $\blacksquare$

### Solution to Problem 4.1 (1D Uniaxial Trace Reduction)
For $\mathbf{n}_{m2} = n_{m2} \text{diag}(-1/2, -1/2, 1)$:  
$\mathbf{n}_{m2}^3 = n_{m2}^3 \text{diag}(-1/8, -1/8, 1)$.  
$\text{Tr}(\mathbf{n}_{m2}^3) = n_{m2}^3 (-1/8 - 1/8 + 1) = \frac{3}{4} n_{m2}^3 \implies \frac{1}{2} \text{Tr}(\mathbf{n}_{m2}^3) = \frac{3}{8} n_{m2}^3$.  
For 1D planar vector $\mathbf{v}_2 = (0, 0, v_2)$:  
$\mathbf{v}_2 \cdot \mathbf{n}_{m2} \cdot \mathbf{v}_2 = (0, 0, v_2) \cdot (0, 0, v_2 n_{m2})^T = v_2^2 n_{m2}$. $\blacksquare$

### Solution to Problem 5.1 (Optimal Mixing Derivation)
Let $E(\alpha) = \int [\Delta \rho_{\text{in}}(z) + \alpha \Delta \rho_{\text{out}}(z)]^2 dz = \int (\Delta \rho_{\text{in}})^2 dz + 2\alpha \int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}} dz + \alpha^2 \int (\Delta \rho_{\text{out}})^2 dz$.  
To find the minimum, set $\frac{dE}{d\alpha} = 0$:  
$2 \int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}} dz + 2\alpha \int (\Delta \rho_{\text{out}})^2 dz = 0 \implies \alpha_{\text{opt}} = -\frac{\int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}} dz}{\int (\Delta \rho_{\text{out}})^2 dz}$. $\blacksquare$

### Solution to Problem 6.1 (Contact Theorem Proof)
The hydrostatic force balance equation for a fluid in an external potential is $\nabla P(\mathbf{r}) = -\rho(\mathbf{r}) \nabla V_{\text{ext}}(\mathbf{r})$.  
In 1D planar geometry for a hard wall at $z=0$, $V_{\text{ext}}(z) = \infty$ for $z < R$ and $0$ for $z \ge R$.  
Integrating from $z = 0$ to $z = \infty$:  
$\int_0^\infty \frac{dP}{dz} dz = P(\infty) - P(0^+) = P_{\text{bulk}} - P(0^+)$.  
The force on the hard wall per unit area is $P(0^+) = k_B T \rho(R^+)$.  
Since $P(\infty) = P_{\text{bulk}}$ and the wall exerts hard-core repulsion, $\rho(R^+) = \beta P_{\text{bulk}}$. $\blacksquare$
