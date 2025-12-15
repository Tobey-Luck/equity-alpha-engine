# Equity Alpha Research Engine

This repository implements a small but complete **systematic equity research pipeline**, focused on cross-sectional factor modeling, out-of-sample validation, and realistic portfolio construction.

The goal of the project is **not** to claim a production-ready trading strategy, but to demonstrate how quantitative equity research is conducted responsibly:
- separating signal research from portfolio implementation,
- validating changes incrementally,
- avoiding data leakage,
- and evaluating results under realistic constraints.

The project progresses through a sequence of notebooks, each making **one controlled change** and measuring its effect.

---

## Project structure

equity-alpha-engine/
│
├── notebooks/
│ ├── 01_research_pipeline.ipynb
│ ├── 02_alpha_model_sector_neutral.ipynb
│ ├── 03_kday_target_ic.ipynb
│ └── 04_kday_portfolio_backtest.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── factors.py
│ ├── alpha_model.py
│ ├── optimizer.py
│ └── backtester.py
│
├── tests/
│ ├── test_data_loader.py
│ ├── test_factors.py
│ ├── test_alpha_model.py
│ ├── test_optimizer.py
│ └── test_backtester.py
│
└── README.md


---

## Notebook overview

### Notebook 01 — Baseline research pipeline

**Purpose:**  
Build a clean, end-to-end baseline for cross-sectional equity research.

**Key components:**
- S&P 500 universe
- Price-based factor library (momentum, reversal, volatility, vol-of-vol)
- Cross-sectional z-scoring
- Out-of-sample Ridge regression (daily horizon)
- Mean-variance portfolio optimization with constraints
- Transaction-cost-aware backtesting

**Outcome:**  
The baseline alpha model performs poorly out-of-sample.  
This is an expected and informative result, establishing a realistic starting point.

---

### Notebook 02 — Sector-neutral alpha model

**Change introduced:**  
Remove sector-level effects inside the daily cross-section before regression.

**Motivation:**  
Naive cross-sectional models often pick up implicit sector bets rather than true stock-level signals.

**Evaluation:**
- Daily information coefficient (IC)
- IC stability over time
- Direct comparison against the baseline model

**Outcome:**  
Sector neutralization modestly improves IC stability but does not materially improve portfolio performance on its own. The change is retained as a defensible structural control.

---

### Notebook 03 — Multi-day forward-return target

**Change introduced:**  
Predict K-day forward returns (K=5) instead of 1-day returns.

**Motivation:**  
Daily returns are extremely noisy. Many equity factors are expected to express over multi-day horizons.

**Evaluation:**
- IC comparison: 1-day vs 5-day target
- Mean IC and IC t-statistics
- Cumulative IC plots

**Outcome:**  
The 5-day target produces a large and statistically significant improvement in IC.
This is the first change that clearly improves signal quality.

---

### Notebook 04 — Portfolio implementation of K-day signal

**Purpose:**  
Test whether the improved 5-day signal survives realistic portfolio construction.

**Portfolio variants:**
- Daily rebalanced
- Fixed 5-day rebalance
- Staggered sleeve implementation

**Constraints:**
- Dollar neutrality
- Gross exposure normalization
- Linear transaction cost model (10 bps)
- Turnover diagnostics

**Outcome:**  
- The daily 5-day strategy delivers strong performance but with elevated turnover.
- The fixed-rebalance and sleeve implementations retain much of the Sharpe while reducing turnover by more than half.
- The improvement is robust to realistic execution constraints.

---

## Key design principles

- **Out-of-sample by construction**  
  All alpha models are trained strictly on historical data prior to the prediction date.

- **One change at a time**  
  Each notebook introduces a single, defensible modification and evaluates it in isolation.

- **Signal ≠ strategy**  
  IC analysis and portfolio backtesting are treated as separate steps.

- **Execution matters**  
  Portfolio results are always evaluated with turnover and transaction costs.

- **Transparency over performance**  
  Negative or weak results are preserved and explained rather than hidden.

---

## Testing and code quality

- Core components are unit-tested using `pytest`
- Tests cover:
  - data loading
  - factor construction
  - alpha model outputs
  - optimizer constraints
  - backtesting logic
- All notebooks run without modifying library code between experiments

---

## What this project demonstrates

- Practical experience with systematic equity research workflows
- Strong understanding of cross-sectional modeling pitfalls
- Ability to diagnose weak models and improve them structurally
- Comfort with realistic portfolio implementation constraints
- Professional research hygiene suitable for buy-side or quantitative research roles

---

## Notes

This project is intended as a **research demonstration**, not an investment product.
Results are sample-dependent and should not be interpreted as live trading performance.