# Numerical Modelling of Reactive CO₂ Absorption in Packed Columns

**Course:** CLL113 — Numerical Methods in Chemical Engineering  
**Instructor:** Prof. Jayati Sarkar · Department of Chemical Engineering, IIT Delhi  
**Author:** Krishan Sawariya · Oct 2026 – Nov 2026  
**Languages:** C++17 · Python 3 (NumPy, Pandas, Matplotlib)

---

## Overview

This project implements a comprehensive suite of numerical methods from the CLL113 syllabus to model reactive CO₂ absorption into aqueous monoethanolamine (MEA) solutions, replicating the benchmark pilot-column dataset of **Tontiwachwuthikul et al. (1992)** (*Chemical Engineering Science*, Run 1: $3.0\text{ M}$ MEA, $6.55\text{ m}$ packed bed).

The numerical framework replicates the $6.55\text{ m}$ pilot packed absorption column across all **11 experimental measurement stations** with **97.75% precision ($R^2 = 0.9994$, $\text{MAPE} = 2.25\%$)**.

---

## Numerical Methods Implemented

| # | Method | Application |
|---|--------|-------------|
| 1 | Non-linear Root Finding (Newton–Raphson) | Kent–Eisenberg VLE speciation |
| 2 | Classic 4th-Order Runge–Kutta (RK-4) | Liquid film reaction–diffusion boundary-layer ODE ($Ha = 1.0\text{--}20.0$) |
| 3 | Finite Differences (2-pt $O(h)$ & 3-pt $O(h^2)$ forward + central) | Interfacial mass-transfer gradient & truncation error analysis |
| 4 | Composite Trapezoidal Rule — $O(h^2)$ | Packed column height integration |
| 5 | Composite Simpson's 1/3 Rule — $O(h^4)$ | Packed column height integration |
| 6 | Composite Simpson's 3/8 Rule — $O(h^4)$ | Packed column height integration |
| 7 | Combined Simpson's (1/3 + 3/8) Scheme | Arbitrary grid subinterval counts $n$ while preserving $O(h^4)$ |
| 8 | 3-Point Gauss–Legendre Quadrature | High-accuracy numerical integration |
| 9 | Richardson Extrapolation | Cancellation of leading $O(h^2)$ error |
| 10 | Non-Linear Arrhenius Least-Squares Regression | Second-order $\text{CO}_2$–MEA kinetics ($k_2(T)$) |

---

## Project Structure

```text
co2_capture_cll113/
├── cpp/
│   ├── main_solver.cpp        # Full C++17 numerical solver
│   └── Makefile               # Build configuration (target: solver)
├── scripts/
│   └── simulation_and_plots.py  # Python validation, convergence & figure generation
├── data/
│   ├── tontiwachwuthikul_1992_benchmark.csv  # 11-station pilot benchmark data
│   ├── integration_convergence.csv           # Quadrature error decay across n = 6..192
│   └── numerical_differentiation_study.csv   # 2-pt vs 3-pt finite-difference error study
├── paper/
│   ├── main.tex               # LaTeX term research paper
│   ├── references.bib         # BibTeX references
│   └── figures/               # Generated 300-DPI PNG figures
│       ├── fig1_column_validation.png
│       ├── fig2_integration_convergence.png
│       ├── fig3_rk4_film_profiles.png
│       ├── fig4_arrhenius_regression.png
│       └── fig5_process_sensitivity.png
└── README.md
```

---

## Key Verified Results

### Level 1 Column Validation
- **Total Packing Height Computed:** $Z_{\text{calc}} = 6.5516\text{ m}$ (experimental: $6.5500\text{ m}$, $< 0.03\%$ deviation)
- **Axial Profile Agreement (11 stations):** $R^2 = 0.9994$, $\text{MAE} = 0.00110$, $\text{MAPE} = 2.25\%$, **97.75% precision**

### Quadrature Convergence Study ($n = 6$ to $192$)
| Rule | Asymptotic Behavior Verified | Error at $n = 192$ |
|------|------------------------------|--------------------|
| Composite Trapezoidal | $O(h^2)$, empirical slope $\approx 2.00$ | $1.48 \times 10^{-4}\text{ m}$ |
| Composite Simpson's 1/3 | $O(h^4)$, empirical slope $\approx 3.99$ | $2.36 \times 10^{-9}\text{ m}$ |
| Composite Simpson's 3/8 | $O(h^4)$, empirical slope $\approx 3.98$ | $5.30 \times 10^{-9}\text{ m}$ |
| 3-Point Gauss–Legendre | Machine-precision convergence | $9.57 \times 10^{-13}\text{ m}$ |

### Film ODE, Differentiation & Kinetics
- **RK-4 Boundary Layer ODE:** Captured transition from intermediate ($Ha = 1.0$, $E = 1.31$) to fast reaction regime ($Ha = 20.0$, $E = 20.00$).
- **Flux Gradient Differentiation ($Ha = 5.0$, $h = 0.004$):** 3-point forward difference reduces error from $4.96 \times 10^{-2}$ (2-pt) to $2.41 \times 10^{-4}$.
- **Arrhenius Regression ($278.15\text{--}338.15\text{ K}$):** $A = 1.198 \times 10^{13}\text{ L/(mol}\cdot\text{s)}$, $E_a = 53.20\text{ kJ/mol}$ ($R^2 = 0.9999$).

---

## Build & Run

### C++17 Solver
```bash
cd cpp
make
./solver
```
Requires: `g++` with C++17 support (`-std=c++17`).

### Python Validation & Plotting
```bash
pip install numpy pandas matplotlib
python scripts/simulation_and_plots.py
```

---

## References

1. Tontiwachwuthikul, P., Meisen, A., & Lim, C. J. (1992). $\text{CO}_2$ absorption by NaOH, monoethanolamine and 2-amino-2-methyl-1-propanol solutions in a packed column. *Chemical Engineering Science*, 47(2), 381–390.
2. Kent, R. L., & Eisenberg, B. (1976). Better data for amine treating. *Hydrocarbon Processing*, 55(2), 87–90.
3. Chapra, S. C., & Canale, R. P. (2015). *Numerical Methods for Engineers* (7th ed.). McGraw-Hill.

---

## Author

**Krishan Sawariya**  
B.Tech Chemical Engineering, IIT Delhi (2025–2029)  
[GitHub](https://github.com/KrishanSawariya) · [LinkedIn](https://www.linkedin.com/in/krishan-sawariya-739779327)
