import numpy as np
import pandas as pd

from src.factors import (
    FactorCalculator,
    FactorConfig,
    zscore_cross_section,
    demean_cross_section,
)


def make_fake_prices(n_days: int = 50, tickers=None) -> pd.DataFrame:
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]

    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    # Simple random walk in log space to avoid negatives
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.0005, scale=0.01, size=(n_days, len(tickers)))
    log_prices = steps.cumsum(axis=0)
    prices = np.exp(log_prices) * 100.0

    return pd.DataFrame(prices, index=idx, columns=tickers)


def test_factor_panel_basic_shape():
    """
    Original core test: only base factors enabled.
    """
    adj_close = make_fake_prices()
    cfg = FactorConfig(
        ret_lag=1,
        momentum_windows=[21],
        volatility_windows=[21],
    )
    calc = FactorCalculator(cfg)

    panel = calc.build_factor_panel(adj_close)

    # We expect a MultiIndex with ['date', 'ticker'] and 3 factor columns.
    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["date", "ticker"]

    assert set(panel.columns) == {"ret_1d", "mom_21d", "vol_21d"}

    # There should be at least some non-NaN values.
    assert panel.notna().any().any()

    # Volatility should be non-negative.
    assert (panel["vol_21d"].dropna() >= 0).all()


def test_factor_panel_with_extras():
    """
    New test: enable extra factors and check that they are present and sane.
    """
    adj_close = make_fake_prices()

    cfg = FactorConfig(
        ret_lag=1,
        momentum_windows=[21],
        volatility_windows=[21],
        include_short_term_reversal=True,
        reversal_window=5,
        include_vol_of_vol=True,
        vov_windows=[21],
        include_vol_adj_momentum=True,
        vol_adj_mom_pairs=[(21, 21)],
    )
    calc = FactorCalculator(cfg)
    panel = calc.build_factor_panel(adj_close)

    expected_base = {"ret_1d", "mom_21d", "vol_21d"}
    expected_extras = {"rev_5d", "vov_21x21d", "mom_21d_over_vol_21d"}

    assert expected_base.issubset(set(panel.columns))
    assert expected_extras.issubset(set(panel.columns))

    # New factors should have some finite values
    for col in expected_extras:
        assert col in panel.columns
        assert panel[col].notna().any()


def test_zscore_and_demean_cross_section():
    """
    Test cross-sectional z-scoring and demeaning utilities.
    """
    adj_close = make_fake_prices()
    cfg = FactorConfig(
        ret_lag=1,
        momentum_windows=[21],
        volatility_windows=[21],
    )
    calc = FactorCalculator(cfg)
    panel = calc.build_factor_panel(adj_close)

    # Z-score cross-sectionally
    z_panel = zscore_cross_section(panel)

    # For each date and factor, mean ~ 0, std ~ 1 (where std is defined).
    for date, group in z_panel.groupby(level="date"):
        # Skip dates with too few names to define stats
        if len(group) < 2:
            continue

        means = group.mean(axis=0)
        stds = group.std(axis=0, ddof=0)

        # Some columns may be all NaN; ignore those.
        valid = stds.notna() & (stds > 0)
        if not valid.any():
            continue

        means = means[valid]
        stds = stds[valid]

        # Means close to zero
        assert np.all(np.isfinite(means))
        assert np.allclose(means.values, 0.0, atol=1e-6)

        # Std close to one
        assert np.allclose(stds.values, 1.0, atol=1e-5)

    # Demean cross-sectionally
    d_panel = demean_cross_section(panel)

    for date, group in d_panel.groupby(level="date"):
        if len(group) < 2:
            continue
        means = group.mean(axis=0)

        # Cross-sectional means should be ~0
        # Ignore all-NaN columns
        means = means[means.notna()]
        if len(means) == 0:
            continue

        assert np.allclose(means.values, 0.0, atol=1e-6)