# Equity Alpha Research Engine

This repository implements a compact but complete systematic equity research pipeline, focused on cross-sectional
factor modeling, out-of-sample validation, and realistic portfolio construction.

The goal of this project is **not** to claim a production-ready trading strategy, but to demonstrate how quantitative
equity research is conducted responsibly:

- separating signal research from portfolio implementation,
- validating changes incrementally,
- avoiding data leakage,
- and evaluating results under realistic constraints.

The project progresses through a sequence of notebooks, each making **one controlled change** and measuring its effect.

---

## Project structure

```text
equity-alpha-engine/
├── notebooks/
│   ├── 01_research_pipeline.ipynb
│   ├── 02_alpha_model_sector_neutral.ipynb
│   ├── 03_kday_target_ic.ipynb
│   └── 04_kday_portfolio_backtest.ipynb
├── src/
│   ├── data_loader.py
│   ├── factors.py
│   ├── alpha_model.py
│   ├── optimizer.py
│   └── backtester.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_factors.py
│   ├── test_alpha_model.py
│   ├── test_optimizer.py
│   └── test_backtester.py
├── README.md
└── requirements.txt
<<<<<<< HEAD
=======
```
>>>>>>> 212cb39 (Fix project structure formatting in README)

---

## Notebook overview

### Notebook 01 — Baseline research pipeline

**Purpose:**  
Build a clean, end-to-end baseline for cross-sectional equity research.

**Key components:**
- S&P 500 universe
- Price-based factor library (momentum, reversal, volatility, vol-of-vol)
- Cross-sectional z-scoring
- Out-of-sample Ridge regression (1-day horizon)
- Mean–variance portfolio optimization with constraints
- Transaction-cost-aware backtesting

**Outcome:**  
The baseline alpha model performs poorly out-of-sample. This is an expected and informative result, establishing a realistic starting point.

---

### Notebook 02 — Sector-neutral alpha model

**Change introduced:**  
Remove sector-level effects inside each daily cross-section before regression.

**Motivation:**  
Naive cross-sectional models often learn implicit sector bets rather than true stock-level signals.

**Evaluation:**
- Daily Information Coefficient (IC)
- Direct comparison against the baseline Ridge model

**Outcome:**  
Sector neutralization modestly improves IC stability but does not materially improve portfolio performance.  
This demonstrates disciplined iteration without overfitting.

---

### Notebook 03 — Multi-day forward-return target

**Change introduced:**  
Predict K-day forward returns (K = 5) instead of 1-day returns.

**Motivation:**  
Daily returns are extremely noisy; many equity factors express more clearly over multi-day horizons.

**Evaluation:**
- IC comparison between 1-day and K-day targets
- Stability and persistence of IC over time

**Outcome:**  
The K-day target produces a **large, statistically significant improvement in IC**, demonstrating a clear structural improvement.

---

### Notebook 04 — Portfolio implementation of K-day alpha

**Purpose:**  
Translate the improved K-day alpha signal into realistic portfolios.

**Implementations tested:**
- Daily rebalancing
- K-day discrete rebalancing
- Staggered K-day sleeves (overlapping sub-portfolios)

**Metrics evaluated:**
- Net performance (with transaction costs)
- Turnover
- Sharpe ratio
- Drawdowns

**Outcome:**  
Staggered K-day implementations retain much of the alpha while significantly reducing turnover, producing a cleaner and more realistic strategy profile.

---

## Key takeaways

- Improving **signal definition** (target horizon) mattered more than parameter tuning.
- Structural changes beat cosmetic optimization.
- Portfolio implementation choices materially affect realized performance.
- A research process that produces negative results is still valuable if it is disciplined and transparent.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies:
- pandas
- numpy
- yfinance
- matplotlib
- seaborn
- pyarrow
- pytest

---

## Disclaimer

<<<<<<< HEAD
- Practical experience with systematic equity research workflows
- Strong understanding of cross-sectional modeling pitfalls
- Ability to diagnose weak models and improve them structurally
- Comfort with realistic portfolio implementation constraints
- Professional research hygiene suitable for buy-side or quantitative research roles

---

## Notes

This project is intended as a **research demonstration**, not an investment product.
Results are sample-dependent and should not be interpreted as live trading performance.
=======
This project is for research and educational purposes only.  
It does not constitute investment advice or a recommendation to trade any financial instrument.
>>>>>>> 212cb39 (Fix project structure formatting in README)
