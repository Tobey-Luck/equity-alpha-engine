from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import Backtester


def make_dummy_prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    data = {
        "AAA": [100, 101, 102, 101, 103],
        "BBB": [50, 49, 48, 50, 51],
    }
    return pd.DataFrame(data, index=idx)


def make_dummy_signal() -> pd.Series:
    """
    Simple long-short signal: long AAA, short BBB every day.
    """
    prices = make_dummy_prices()
    idx = prices.index

    tuples = []
    values = []

    for d in idx:
        tuples.append((d, "AAA"))
        values.append(1.0)
        tuples.append((d, "BBB"))
        values.append(-1.0)

    index = pd.MultiIndex.from_tuples(tuples, names=["date", "ticker"])
    return pd.Series(values, index=index, name="signal")


def test_backtester_zero_cost_matches_gross():
    prices = make_dummy_prices()
    signal = make_dummy_signal()

    bt = Backtester(transaction_cost_bps=0.0)
    result = bt.run(prices, signal)

    ret_net = result["returns"]
    ret_gross = result["returns_gross"]

    # Same index
    assert ret_net.index.equals(ret_gross.index)

    # With zero costs, net and gross must match
    assert np.allclose(ret_net.values, ret_gross.values, atol=1e-12)

    # Weights should be dollar-neutral and unit-gross each day
    weights = result["weights"]["weight"].unstack("ticker")
    row_sums = weights.sum(axis=1)
    gross = weights.abs().sum(axis=1)

    assert np.allclose(row_sums.values, 0.0, atol=1e-12)
    assert np.allclose(gross.values, 1.0, atol=1e-12)


def test_backtester_positive_cost_reduces_returns():
    prices = make_dummy_prices()
    signal = make_dummy_signal()

    # Backtester without costs
    bt0 = Backtester(transaction_cost_bps=0.0)
    res0 = bt0.run(prices, signal)
    ret0 = res0["returns"]

    # Backtester with significant costs
    bt1 = Backtester(transaction_cost_bps=100.0)  # 100 bps per unit turnover
    res1 = bt1.run(prices, signal)
    ret1 = res1["returns"]

    # Same index
    assert ret0.index.equals(ret1.index)

    # With positive costs and non-zero turnover, net returns must
    # be less than or equal to the zero-cost case, and strictly less
    # in aggregate.
    assert np.all(ret1.values <= ret0.values + 1e-12)
    assert ret1.sum() < ret0.sum()