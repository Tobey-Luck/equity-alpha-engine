from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer import PortfolioOptimizer, OptimizerConfig


def make_toy_data():
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD"]

    rng = np.random.default_rng(0)
    # Simulate random returns with some correlation
    base = rng.normal(0, 0.01, size=(len(dates), 1))
    noise = rng.normal(0, 0.01, size=(len(dates), len(tickers)))
    rets = base + noise
    returns = pd.DataFrame(rets, index=dates, columns=tickers)

    # Simple expected returns: last day's realized return
    mu_panel = returns.shift(1).stack()
    mu_panel.index.set_names(["date", "ticker"], inplace=True)

    return returns, mu_panel


def test_optimize_day_basic():
    returns, mu_panel = make_toy_data()
    d = returns.index[-1]

    mu_d = mu_panel.xs(d, level="date")
    cov_d = returns.iloc[-60:].cov()

    cfg = OptimizerConfig(
        target_gross_leverage=1.0,
        max_weight=0.5,
        risk_aversion=10.0,
        turnover_penalty=0.5,
        enforce_dollar_neutral=True,
    )
    opt = PortfolioOptimizer(cfg)

    w = opt.optimize_day(mu_d, cov_d)

    assert isinstance(w, pd.Series)
    assert set(w.index) == set(mu_d.index)

    # Dollar-neutral (within numerical tolerance)
    assert abs(w.sum()) < 1e-6

    # Gross exposure constraint
    assert w.abs().sum() <= cfg.target_gross_leverage + 1e-6

    # Box constraint
    assert (w.abs() <= cfg.max_weight + 1e-6).all()


def test_optimize_time_series_shape_and_constraints():
    returns, mu_panel = make_toy_data()

    cfg = OptimizerConfig(
        target_gross_leverage=1.0,
        max_weight=0.5,
        risk_aversion=5.0,
        turnover_penalty=1.0,
        enforce_dollar_neutral=True,
    )
    opt = PortfolioOptimizer(cfg)

    weights = opt.optimize_time_series(
        exp_ret_panel=mu_panel,
        returns=returns,
        cov_lookback_days=40,
        min_cov_days=20,
    )

    # Non-empty result with proper MultiIndex
    assert isinstance(weights, pd.Series)
    assert isinstance(weights.index, pd.MultiIndex)
    assert weights.index.names == ["date", "ticker"]
    assert len(weights) > 0

    # Check constraints on a few random dates
    for d, w_d in weights.groupby(level="date"):
        # Dollar-neutral
        assert abs(w_d.sum()) < 1e-5
        # Gross exposure
        assert w_d.abs().sum() <= cfg.target_gross_leverage + 1e-6
        # Box
        assert (w_d.abs() <= cfg.max_weight + 1e-6).all()