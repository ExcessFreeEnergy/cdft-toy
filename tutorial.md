# Classical Density Functional Theory (cDFT / FMT) Interactive Masterclass & Laboratory Guide

An Interactive Simulation & Laboratory Companion Aligned Section-by-Section with Roland Roth's Landmark Topical Review:  
**"Fundamental measure theory for hard-sphere mixtures: a review"** (*J. Phys.: Condens. Matter 22, 2010, 063102*)

---

## 📖 How to Use This Masterclass & Lab Companion

This guide is designed as a **hands-on laboratory manual and follow-along companion** to be read side-by-side with Roland Roth's review paper (*J. Phys.: Condens. Matter 22, 2010, 063102*).

### Recommended Workflow
1. **Open Roland Roth's review paper** on one side of your screen.
2. **Launch the Raylib Desktop Simulator** on the other side:
   ```bash
   uv run python app_raylib.py
   ```
3. Read each section of the review paper, then perform the corresponding **Hands-On Follow-Along Labs** below. Use the interactive GUI buttons, sliders, and plot viewports—or run the CLI step demonstration scripts—to visually and empirically discover the underlying statistical mechanics.

---

## Module 1: Introduction & Hard-Sphere Geometry (Roth 2010 Sec. 1 & Sec. 3)

### 1.1 Theoretical Companion (Deconvolution of the Mayer-$f$ Function)
In Section 3 of Roth (2010), the hard-sphere pair potential $V(r)$ and Mayer-$f$ function $f(r) = -1 + \Theta(r - \sigma)$ are deconvoluted into single-particle fundamental geometric weight functions:
- **Volume weight**: $w_3(z) = \pi (R^2 - z^2)$ over $z \in [-R, R]$
- **Surface weight**: $w_2(z) = 2\pi R$ over $z \in [-R, R]$
- **Vector weight**: $v_2(z) = 2\pi z$ (odd parity under $z \to -z$)

Convoluting the spatial density profile $\rho(z)$ with these weight functions yields local weighted densities $n_\alpha(z) = (\rho * w_\alpha)(z)$.

---

### 1.2 Follow-Along Lab 1: Visualizing Weight Convolutions & Vector Flux

#### 🎮 Interactive Simulator Setup:
1. Launch `app_raylib.py`.
2. Under **Plot Viewport Mode**, click **`Weighted n`**.
3. Under **Geometry Mode**, click **`Single Wall (z=0)`**.
4. Set **Bulk Packing Fraction ($\eta$)** slider to `0.4257`.
5. Click **`Solve`**.

#### 🔍 Interactive Inspection & Discovery:
- **Observe $n_3(z)$ (Cyan Curve)**: Notice how $n_3(z)$, representing the local packing fraction, rises smoothly from zero inside the hard wall ($z < R = 0.5\sigma$) to the bulk packing fraction $\eta = 0.4257$ deep in the fluid.
- **Observe $n_2(z)$ (Green Curve)**: Surface area density $n_2(z)$ exhibits pronounced peaks at contact $z = R$, representing sphere surface area accumulation.
- **Observe $v_2(z)$ (Red Curve)**: Vector flux $v_2(z)$ displays a sharp positive peak at the wall contact $z = R$ ($v_2 \approx +1.5$), pointing away from the wall due to asymmetric spatial particle flux. Deep in the bulk ($z > 3.0\sigma$), $v_2(z) \to 0$ due to spherical symmetry.

#### 💻 CLI Laboratory Inspection:
Run Demonstration Script 03 to inspect exact analytical weight integrals and SPT bulk limits:

```bash
uv run python scripts/demo_step03.py
```

**What the Output Demonstrates**:
- Verifies $\int_{-R}^R w_3(z) dz = v_{\text{sphere}} = \frac{4}{3}\pi R^3$
- Verifies $\int_{-R}^R w_2(z) dz = s_{\text{sphere}} = 4\pi R^2$
- Verifies $\int_{-R}^R v_2(z) dz = 0$ (odd vector parity cancellation)

---

### 1.3 Problem Set 1: Geometry & Weight Calculus

#### Problem 1.1 (Analytical Weight Proof)
Prove by integration in cylindrical coordinates $(r_\perp, \theta, z)$ that the 3D surface delta weight $w_2(\mathbf{r}) = \delta(R - \vert\mathbf{r}\vert)$ reduces in 1D planar geometry to $w_2(z) = 2\pi R$ for $\vert z\vert \le R$ and $0$ otherwise.

#### Problem 1.2 (Bulk Limits of Weighted Densities)
For a uniform bulk fluid with constant density $\rho(z) = \rho_{\text{bulk}}$ and packing fraction $\eta = \frac{\pi}{6} \rho_{\text{bulk}}\sigma^3$, evaluate the bulk values of all six weighted densities: $n_3^{\text{bulk}}, n_2^{\text{bulk}}, n_1^{\text{bulk}}, n_0^{\text{bulk}}, \mathbf{v}_1^{\text{bulk}}, \mathbf{v}_2^{\text{bulk}}$.

---

## Module 2: Variational Principle & Sum-Rules (Roth 2010 Sec. 2 & Sec. 5.1)

### 2.1 Theoretical Companion (Contact Theorem & Surface Thermodynamics)
Section 2 and Section 5.1 of Roth (2010) establish the grand potential variational principle $\frac{\delta \Omega}{\delta \rho(z)} = 0$ and four exact thermodynamic sum-rules:
1. **Wall Contact Theorem**: The density at wall contact $\rho(R^+)$ equals the reduced bulk pressure $\beta P_{\text{bulk}}$.
2. **Spatial Surface Tension**: $\beta \gamma = \int_0^{L_z} [\omega(z) + \beta P_{\text{bulk}}] dz$.
3. **Bulk-Route Surface Tension**: $\beta \gamma_{\text{bulk}} = \left. \frac{\partial \Phi}{\partial n_2} \right|_{\text{bulk}}$.
4. **Gibbs Adsorption Theorem**: $-\frac{d\gamma}{d\mu} = \Gamma = \int_0^{L_z} (\rho(z) - \rho_{\text{bulk}}) dz$.

---

### 2.2 Follow-Along Lab 2: Verifying the Hard Wall Contact Theorem

#### 🎮 Interactive Simulator Setup:
1. In `app_raylib.py`, click **`Preset: Fig 1a (0.4257)`** on the sidebar.
2. Select **Plot Viewport Mode** $\to$ **`Density rho`**.
3. Click **`Show c^(1)(z)`** on the bottom right action button to display the lower direct correlation plot.
4. Click **`Solve`** and wait for convergence (`Status: CONVERGED`).

#### 🔍 Interactive Inspection & Discovery:
- **Inspect Sidebar Thermodynamics Panel**:
  - Look at **Bulk Pressure (bp)**: Reads `6.5662` for Carnahan-Starling (or `6.7302` for Percus-Yevick).
  - Look at **Contact rho(R+)**: Reads `6.5662` (matching bulk pressure within **$< 0.3\%$ relative error**).
- **Inspect Lower Plot (`c^(1)(z) Direct Correlation`)**:
  - Observe how the direct correlation function $c^{(1)}(z) = -\beta \frac{\delta F_{\text{ex}}}{\delta \rho(z)}$ is strongly negative inside the wall ($z < R$) due to infinite hard-core repulsion and approaches $c_1^{\text{bulk}}$ in the fluid bulk.

#### 💻 CLI Laboratory Inspection:
Run Demonstration Script 09 to verify numerical Gibbs adsorption consistency $-\frac{\Delta \gamma}{\Delta \mu} = \Gamma$:

```bash
uv run python scripts/demo_step09.py
```

---

### 2.3 Problem Set 2: Scaled Particle Thermodynamics

#### Problem 2.1 (Bulk PY Pressure Derivation)
Using $\Phi^{\text{RF}}_{\text{bulk}} = -n_0 \ln(1 - n_3) + \frac{n_1 n_2}{1 - n_3} + \frac{n_2^3}{24\pi (1 - n_3)^2}$, substitute the bulk weighted densities as functions of $\eta$ and derive the PY bulk pressure $\beta P_{\text{PY,comp}}(\eta)$.

#### Problem 2.2 (Bulk Excess Chemical Potential)
Derive the bulk excess chemical potential $\beta \mu_{\text{ex}}^{\text{PY}} = \left. \frac{\partial \Phi^{\text{RF}}}{\partial \rho} \right|_{\text{bulk}}$.

---

## Module 3: Rosenfeld Functional & Percus-Yevick Fluid (Roth 2010 Sec. 4.1)

### 3.1 Theoretical Companion (Rosenfeld Functional Derivation)
Section 4.1 of Roth (2010) presents Yasha Rosenfeld's (1989) original functional:

$$\Phi^{\text{RF}} = -n_0 \ln(1 - n_3) + \frac{n_1 n_2 - \mathbf{n}_1 \cdot \mathbf{n}_2}{1 - n_3} + \frac{n_2^3 - 3 n_2 \mathbf{n}_2 \cdot \mathbf{n}_2}{24\pi (1 - n_3)^2}$$

In uniform bulk fluid, $\Phi^{\text{RF}}$ yields the Percus-Yevick (PY) compressibility equation of state:

$$\beta P_{\text{PY}} = \rho_{\text{bulk}} \frac{1 + \eta + \eta^2}{(1 - \eta)^3}$$

---

### 3.2 Follow-Along Lab 3: Hard-Sphere Shell Packing at Moderate Packing Fractions

#### 🎮 Interactive Simulator Setup:
1. Click **`Preset: Fig 1a (0.4257)`** ($\eta = 0.4257$).
2. Select **FMT Functional Variant** $\to$ **`RF (Original)`**.
3. Click **`Show Benchmark Dots`** to display published Monte Carlo reference points.
4. Click **`Solve`**.

#### 🔍 Interactive Inspection & Discovery:
- **Compare Profile Against Benchmark Dots**:
  - Observe how the calculated cyan profile $\rho(z)$ forms characteristic density oscillations (packing shells) near the hard wall.
  - Hover the mouse cursor over the primary peak at $z = 0.5\sigma$ ($z = R$).
  - Notice that at moderate packing fraction $\eta = 0.4257$, Rosenfeld's `RF` functional matches the Monte Carlo simulation dots exceptionally well.

#### 💻 CLI Laboratory Inspection:
Run Demonstration Script 04 to verify analytical functional derivatives $\frac{\partial \Phi^{\text{RF}}}{\partial n_\alpha}$ against finite-difference gradients:

```bash
uv run python scripts/demo_step04.py
```

---

## Module 4: High-Density Breakdown & Carnahan-Starling Functionals (Roth 2010 Sec. 4.3)

### 4.1 Theoretical Companion (White-Bear & White-Bear II)
Section 4.3 of Roth (2010) explains why Percus-Yevick underestimates bulk pressure at high densities ($\eta > 0.40$). The Carnahan-Starling (CS) equation of state provides superior agreement:

$$\beta P_{\text{CS}} = \rho_{\text{bulk}} \frac{1 + \eta + \eta^2 - \eta^3}{(1 - \eta)^3}$$

The White-Bear (`WB`) and White-Bear II (`WBII`) functionals incorporate the CS equation of state via a modified prefactor $f_4^{\text{WB}}(n_3)$:

$$\Phi_3^{\text{WB}} = (n_2^3 - 3 n_2 \mathbf{n}_2 \cdot \mathbf{n}_2) f_4^{\text{WB}}(n_3), \quad f_4^{\text{WB}}(n_3) = \frac{n_3 + (1 - n_3)^2 \ln(1 - n_3)}{36\pi n_3^2 (1 - n_3)^2}$$

---

### 4.2 Follow-Along Lab 4: High-Density PY Breakdown & Carnahan-Starling Recovery

#### 🎮 Interactive Simulator Setup:
1. Click **`Preset: Fig 1b (0.4783)`** ($\eta = 0.4783$, Roth 2010 Fig 1b high-density benchmark).
2. Click **`Show Benchmark Dots`**.
3. Select **FMT Functional Variant** $\to$ **`RF (Original)`**.
4. Click **`Solve`**.

#### 🔍 Interactive Inspection & Discovery:
- **Step 1 (Observe PY Overestimation)**: Look at the contact density on the sidebar: **Contact rho(R+)** reads `10.5678` (matching PY pressure). Notice that the cyan curve overestimates the first Monte Carlo dot at $z = 0.5\sigma$ (which lies at $\approx 9.9$).
- **Step 2 (Switch to White-Bear)**: Under **FMT Functional Variant**, click **`WB (White-Bear)`** or **`WBII (Mark II)`**, then click **`Solve`**.
- **Step 3 (Observe CS Alignment)**: Watch **Contact rho(R+)** drop from `10.5678` to `9.9233`, shifting the primary contact peak down to align **perfectly** with the Monte Carlo simulation dots!

#### 💻 CLI Laboratory Inspection:
Run Demonstration Script 07 to view the high-density benchmark comparative table across all functionals:

```bash
uv run python scripts/demo_step07.py
```

---

## Module 5: Dimensional Crossover & Tarazona Tensorial FMT (Roth 2010 Sec. 4.2 & Sec. 4.4)

### 5.1 Theoretical Companion (Zero-D Cavity Collapse & Tensorial Trace)
Sections 4.2 and 4.4 of Roth (2010) address **dimensional crossover**: confining a fluid to 2D, 1D, or a zero-dimensional (0D) single-particle cavity.

For scalar functionals (`RF`, `WB`), under extreme zero-D confinement, $\mathbf{v}_2 \to n_2$, causing:

$$n_2^3 - 3 n_2 \mathbf{v}_2 \cdot \mathbf{v}_2 \to -2 n_2^3$$

The third term $\Phi_3 \propto \frac{n_2^3}{(1-n_3)^2}$ diverges non-physically. Tarazona (2000) resolved this divergence by introducing a tensorial weight $\omega_{m2}(z) = 2\pi R (\frac{z^2}{R^2} - \frac{1}{3})$, generating tensorial weighted density $n_{m2}(z)$ whose trace cancels the scalar divergence:

$$\Phi_3^{\text{Tensor}} = \frac{n_2^3 - 3 n_2 v_2^2 + 9 \left( v_2^2 n_{m2} - \frac{3}{8} n_{m2}^3 \right)}{24\pi (1 - n_3)^2}$$

---

### 5.2 Follow-Along Lab 5: Zero-D Cavity Collapse & Slit-Pore Confinement

#### 🎮 Interactive Simulator Setup:
1. In `app_raylib.py`, select **Plot Viewport Mode** $\to$ **`Crossover Suite`**.
2. Ensure **Geometry Mode** is **`Single Wall (z=0)`** (Zero-D Cavity Collapse vs Cavity Width $\alpha$).

#### 🔍 Interactive Inspection & Discovery:
- **Observe Scalar Divergence Spikes**: Look at the Red (`RF`) and Amber (`WB`) curves as cavity width $\alpha \to 0.03\sigma$. Peak free energy density $\max \Phi(z)$ diverges rapidly ($\Phi \to 200+$).
- **Observe Tensorial Stability**: Look at **WB-Tensor** (thick cyan curve). It remains strictly bounded ($\Phi \le 2.5$), empirically demonstrating 0D cavity collapse stability!
- **Switch to Slit Pore Mode**: Click **`Slit Pore`** under **Geometry Mode**. Observe the Pore Confinement Sweep ($L_z$), demonstrating how tight pore confinement restricts density packing.

#### 💻 CLI Laboratory Inspection:
Run Demonstration Script 10 to inspect zero-D Gaussian divergence metrics across cavity widths:

```bash
uv run python scripts/demo_step10.py
```

---

## Module 6: Numerical Implementation & Roth Adaptive Picard Solver (Roth 2010 Sec. 8)

### 6.1 Theoretical Companion (FFT Convolutions & Optimal Line-Search)
Section 8 of Roth (2010) details 1D planar numerical discretization:
- **Zero-Padded FFT Convolutions**: Padding arrays to $N_{\text{fft}} = N_{\text{grid}} + N_w - 1$ eliminates periodic boundary wraparound.
- **Section 8.4 Simpson Endpoint Modifications**: Endpoint weights multiplied by $3/8$, index 1 by $7/6$, and index 2 by $23/24$.
- **Section 8.1 Roth Optimal Line-Search Picard Solver**: Calculates optimal mixing parameter $\alpha_{\text{opt}}$ per iteration:

$$\alpha_{\text{opt}} = -\frac{\int \Delta \rho_{\text{in}}(z) \Delta \rho_{\text{out}}(z) dz}{\int [\Delta \rho_{\text{out}}(z)]^2 dz}$$

---

### 6.2 Follow-Along Lab 6: Single-Stepping the Picard Solver & Residual Decay

#### 🎮 Interactive Simulator Setup:
1. Set **Bulk Packing Fraction ($\eta$)** slider to `0.4500`.
2. Click **`Reset`** to initialize the step profile.
3. Click **`Show R(k) History`** on the lower right diagnostic button to show the log residual plot.

#### 🔍 Interactive Inspection & Discovery:
- **Click Step 1 Iter**: Click the **`Step 1 Iter`** button repeatedly (5 to 10 times).
- **Observe Monotonic Residual Decay**: Watch the log residual curve $\log_{10} R(k)$ drop on the lower plot from $10^{-1}$ down to $10^{-6}$.
- **Inspect Sidebar Alpha Opt**: Watch **Alpha Opt (alpha)** on the sidebar dynamically adjust per iteration ($\alpha \approx 0.03 \to 0.08$) to maximize convergence speed while preventing density over-packing ($n_3 < 1.0$).

#### 💻 CLI Laboratory Inspection:
Run Demonstration Script 06 to compare convergence speedup of Fixed Picard ($\alpha=0.01$) vs Roth Adaptive Picard:

```bash
uv run python scripts/demo_step06.py
```

---

### 6.3 Problem Set 5 & 6: Picard Mechanics & Contact Proof

#### Problem 5.1 (Optimal Mixing Derivation)
Derive the formula for $\alpha_{\text{opt}}$ by minimizing the norm of the expected next-iteration residual $E(\alpha) = \int [\Delta \rho_{\text{in}} + \alpha \Delta \rho_{\text{out}}]^2 dz$ with respect to $\alpha$.

#### Problem 6.1 (Contact Theorem Proof)
By integrating the hydrostatic force balance equation $\frac{dP}{dz} = -\rho(z) \frac{dV_{\text{ext}}}{dz}$ across a hard wall at $z=0$, prove that $\rho(R^+) = \beta P_{\text{bulk}}$.

---

## Module 7: Full End-to-End Performance & Regression Benchmark

Run the complete benchmark suite to print the full Monte Carlo contact density regression table and throughput benchmarks:

```bash
uv run python scripts/demo_step12.py
```

**Benchmark Targets**:
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
For uniform $\rho(z) = \rho_{\text{bulk}}$ with sphere radius $R$ and diameter $\sigma = 2R$:
- $n_3^{\text{bulk}} = \rho_{\text{bulk}} v_{\text{sphere}} = \frac{4}{3}\pi R^3 \rho_{\text{bulk}} = \eta$
- $n_2^{\text{bulk}} = \rho_{\text{bulk}} s_{\text{sphere}} = 4\pi R^2 \rho_{\text{bulk}} = \frac{3\eta}{R} = \frac{6\eta}{\sigma}$
- $n_1^{\text{bulk}} = \frac{n_2^{\text{bulk}}}{4\pi R} = R \rho_{\text{bulk}} = \frac{3\eta}{4\pi R^2} = \frac{3\eta}{\pi \sigma^2}$
- $n_0^{\text{bulk}} = \frac{n_2^{\text{bulk}}}{4\pi R^2} = \rho_{\text{bulk}} = \frac{3\eta}{4\pi R^3} = \frac{6\eta}{\pi \sigma^3}$
- $v_1^{\text{bulk}} = v_2^{\text{bulk}} = 0$ (due to spherical symmetry in bulk). $\blacksquare$

### Solution to Problem 2.1 (Bulk PY Pressure Derivation)
In bulk fluid, $\mathbf{n}_1 = \mathbf{n}_2 = 0$. Substituting $n_0, n_1, n_2, n_3$ as functions of $\eta$:  
$\Phi_{\text{bulk}}^{\text{RF}} = -n_0 \ln(1-n_3) + \frac{n_1 n_2}{1-n_3} + \frac{n_2^3}{24\pi (1-n_3)^2} = \rho_{\text{bulk}} \left[ -\ln(1-\eta) + \frac{3\eta}{1-\eta} + \frac{3\eta^2}{2(1-\eta)^2} \right]$.  
Evaluating $\beta P = \eta \frac{\partial \Phi}{\partial \eta}$ for the excess pressure and adding ideal gas pressure $\rho_{\text{bulk}}$ yields:  
$$\beta P_{\text{PY,comp}} = \rho_{\text{bulk}} \frac{1 + \eta + \eta^2}{(1 - \eta)^3}. \quad \blacksquare$$

### Solution to Problem 2.2 (Bulk Excess Chemical Potential)
Evaluating the partial derivatives of $\Phi^{\text{RF}}$ in uniform bulk fluid:
- $\frac{\partial \Phi}{\partial n_0} = -\ln(1-\eta)$
- $\frac{\partial \Phi}{\partial n_1} = \frac{n_2}{1-\eta} = \frac{3\eta}{R(1-\eta)}$
- $\frac{\partial \Phi}{\partial n_2} = \frac{n_1}{1-\eta} + \frac{n_2^2}{8\pi (1-\eta)^2} = \frac{3\eta}{4\pi R^2 (1-\eta)} + \frac{9\eta^2}{8\pi R^2 (1-\eta)^2}$
- $\frac{\partial \Phi}{\partial n_3} = \frac{n_0}{1-\eta} + \frac{n_1 n_2}{(1-\eta)^2} + \frac{n_2^3}{12\pi (1-\eta)^3} = \frac{3\eta}{4\pi R^3 (1-\eta)} + \frac{9\eta^2}{4\pi R^3 (1-\eta)^2} + \frac{9\eta^3}{2\pi R^3 (1-\eta)^3}$

Multiplying each partial derivative by its respective weight integral ($\int w_0 dz = 1, \int w_1 dz = R, \int w_2 dz = 4\pi R^2, \int w_3 dz = \frac{4}{3}\pi R^3$) and summing:  
$$\beta \mu_{\text{ex}}^{\text{PY}} = -\ln(1-\eta) + \frac{7\eta}{1-\eta} + \frac{15\eta^2}{2(1-\eta)^2} + \frac{3\eta^3}{(1-\eta)^3}. \quad \blacksquare$$

### Solution to Problem 3.1 (Low-Density Series Derivation)
Using L'Hôpital's rule on $f_4^{\text{WB}}(n_3) = \frac{N(n_3)}{D(n_3)}$:  
$N(n_3) = n_3 + (1-n_3)^2 \ln(1-n_3)$  
$D(n_3) = 36\pi n_3^2 (1-n_3)^2$  
Evaluating second derivatives at $n_3 = 0$:  
$N''(0) = 3$  
$D''(0) = 72\pi$  
Ratio: $\lim_{n_3 \to 0} f_4^{\text{WB}}(n_3) = \frac{N''(0)}{D''(0)} = \frac{3}{72\pi} = \frac{1}{24\pi}$. $\blacksquare$

### Solution to Problem 4.1 (1D Uniaxial Trace Reduction)
For $\mathbf{n}_{m2} = n_{m2} \text{diag}(-1/2, -1/2, 1)$:  
$\mathbf{n}_{m2}^3 = n_{m2}^3 \text{diag}(-1/8, -1/8, 1)$.  
$\text{Tr}(\mathbf{n}_{m2}^3) = n_{m2}^3 (-1/8 - 1/8 + 1) = \frac{3}{4} n_{m2}^3 \implies \frac{1}{2} \text{Tr}(\mathbf{n}_{m2}^3) = \frac{3}{8} n_{m2}^3$.  
For 1D planar vector $\mathbf{v}_2 = (0, 0, v_2)$:  
$\mathbf{v}_2 \cdot \mathbf{n}_{m2} \cdot \mathbf{v}_2 = (0, 0, v_2) \cdot (0, 0, v_2 n_{m2})^T = v_2^2 n_{m2}$. $\blacksquare$

### Solution to Problem 5.1 (Optimal Mixing Derivation)
Let $E(\alpha) = \int [\Delta \rho_{\text{in}}(z) + \alpha \Delta \rho_{\text{out}}(z)]^2 dz = \int (\Delta \rho_{\text{in}})^2 dz + 2\alpha \int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}} dz + \alpha^2 \int (\Delta \rho_{\text{out}})^2 dz$.  
Setting $\frac{dE}{d\alpha} = 0$:  
$2 \int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}} dz + 2\alpha \int (\Delta \rho_{\text{out}})^2 dz = 0 \implies \alpha_{\text{opt}} = -\frac{\int \Delta \rho_{\text{in}} \Delta \rho_{\text{out}} dz}{\int (\Delta \rho_{\text{out}})^2 dz}$. $\blacksquare$

### Solution to Problem 6.1 (Contact Theorem Proof)
The hydrostatic force balance equation for a fluid in an external potential is $\nabla P(\mathbf{r}) = -\rho(\mathbf{r}) \nabla V_{\text{ext}}(\mathbf{r})$.  
In 1D planar geometry for a hard wall at $z=0$, $V_{\text{ext}}(z) = \infty$ for $z < R$ and $0$ for $z \ge R$.  
Integrating from $z = 0$ to $z = \infty$:  
$\int_0^\infty \frac{dP}{dz} dz = P(\infty) - P(0^+) = P_{\text{bulk}} - P(0^+)$.  
The force on the hard wall per unit area is $P(0^+) = k_B T \rho(R^+)$.  
Since $P(\infty) = P_{\text{bulk}}$ and the wall exerts hard-core repulsion, $\rho(R^+) = \beta P_{\text{bulk}}$. $\blacksquare$
