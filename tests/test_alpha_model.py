import pandas as pd
import numpy as np

from src.factors import FactorCalculator, FactorConfig
from src.alpha_model import AlphaModel
from tests.test_factors import make_fake_prices


def test_alpha_model_runs():
    prices = make_fake_prices()

    # Build simple factors
    cfg = FactorConfig(
        ret_lag=1,
        momentum_windows=[5],
        volatility_windows=[5],
    )
    calc = FactorCalculator(cfg)
    panel = calc.build_factor_panel(prices)

    factor_cols = list(panel.columns)

    model = AlphaModel(factor_columns=factor_cols)
    result = model.fit(panel, prices)

    assert "betas" in result
    assert "predictions" in result
    assert "residuals" in result
    assert "ic" in result

    # Some IC values should exist
    assert len(result["ic"]) > 0
