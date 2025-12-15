from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import pandas as pd


@dataclass
class Backtester:
    """
    Simple daily-rebalanced long-short backtester with optional
    transaction costs.

    Assumptions
    -----------
    - Input prices are daily adjusted closes (wide, date x ticker).
    - Input signal is a cross-sectional score by (date, ticker).
    - Each day we:
        1) Convert scores into dollar-neutral, unit-gross weights.
        2) Hold those weights for one day.
        3) Realize portfolio return as the weighted average of
           single-stock returns.
        4) Pay proportional transaction costs based on daily turnover.

    The output 'returns' series is net of transaction costs.
    """

    # Transaction cost in basis points per unit of one-way turnover.
    # Example: 10.0 means 10 bps (0.001) cost per 1.0 of notional traded.
    transaction_cost_bps: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        prices: pd.DataFrame,
        signal: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Run a daily-rebalanced long-short backtest.

        Parameters
        ----------
        prices : DataFrame
            Wide DataFrame with DatetimeIndex (sorted ascending) and
            columns = ticker symbols.
        signal : Series or DataFrame
            Cross-sectional signal by (date, ticker). If Series, index
            must be a MultiIndex (date, ticker). If DataFrame, must
            have the same index with a 'signal' column.

        Returns
        -------
        dict
            {
              'returns':       net portfolio returns (after costs),
              'returns_gross': gross portfolio returns (before costs),
              'weights':       portfolio weights by (date, ticker),
              'turnover':      daily turnover
            }
        """
        prices = self._validate_prices(prices)
        signal_df = self._normalize_signal(signal)

        # Build weights from signal: dollar-neutral, unit-gross each day
        weights = self._signal_to_weights(signal_df)

        # Align weights to price dates (fill missing dates with zero weights)
        weights_wide = (
            weights["weight"]
            .unstack("ticker")
            .reindex(prices.index)
            .fillna(0.0)
        )

        # Daily simple returns
        rets_wide = prices.pct_change().fillna(0.0)

        # Weights used to earn return on date t are the weights chosen
        # at the end of t-1. Approximate with a 1-day lag.
        w_lag = weights_wide.shift(1).fillna(0.0)

        # Gross portfolio return: sum_i w_{t-1,i} * r_{t,i}
        returns_gross = (w_lag * rets_wide).sum(axis=1)
        returns_gross.name = "returns_gross"

        # Daily turnover: sum |w_t - w_{t-1}| (one-way notional traded)
        dw = weights_wide.diff().fillna(weights_wide)
        turnover = dw.abs().sum(axis=1)
        turnover.name = "turnover"

        # Transaction costs: cost_rate * turnover
        cost_rate = self.transaction_cost_bps / 10000.0
        costs = cost_rate * turnover

        # Net returns = gross - costs
        returns_net = returns_gross - costs
        returns_net.name = "returns"

        # Pack weights back into long format
        weights_long = (
            weights_wide.stack()
            .to_frame("weight")
            .sort_index()
        )

        return {
            "returns": returns_net,
            "returns_gross": returns_gross,
            "weights": weights_long,
            "turnover": turnover,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)

        prices = prices.sort_index()

        if prices.isna().all(axis=1).any():
            # Drop dates with all-NaN prices
            prices = prices.loc[~prices.isna().all(axis=1)]

        return prices

    @staticmethod
    def _normalize_signal(
        signal: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Normalize signal input into a DataFrame with columns=['signal']
        and a MultiIndex (date, ticker).
        """
        if isinstance(signal, pd.Series):
            df = signal.to_frame("signal")
        elif isinstance(signal, pd.DataFrame):
            if "signal" in signal.columns:
                df = signal[["signal"]].copy()
            else:
                # If multiple columns, take the first as 'signal'
                first_col = signal.columns[0]
                df = signal[[first_col]].rename(columns={first_col: "signal"})
        else:
            raise TypeError("signal must be a pandas Series or DataFrame.")

        if not isinstance(df.index, pd.MultiIndex):
            # Assume wide DataFrame date x ticker; stack to long
            df = df.stack().to_frame("signal")

        if df.index.names != ["date", "ticker"]:
            df.index.set_names(["date", "ticker"], inplace=True)

        # Sort by date, then ticker
        df = df.sort_index()

        return df

    @staticmethod
    def _signal_to_weights(signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw signal scores into daily dollar-neutral, unit-gross
        portfolio weights by date.

        Each day:
          - Subtract cross-sectional mean → net exposure ≈ 0.
          - Rescale so sum |w_i| = 1 → unit gross exposure.
        """
        def _one_day(group: pd.DataFrame) -> pd.DataFrame:
            s = group["signal"].astype(float)
            s = s.replace([np.inf, -np.inf], np.nan)

            if s.isna().all():
                w = s.fillna(0.0)
            else:
                # Center
                s = s - s.mean(skipna=True)

                gross = s.abs().sum(skipna=True)
                if not np.isfinite(gross) or gross <= 0.0:
                    w = s * 0.0
                else:
                    w = s / gross

            out = pd.DataFrame({"weight": w})
            return out

        # IMPORTANT: group_keys=False so we keep the original (date, ticker) index
        weights = signal_df.groupby(level="date", group_keys=False).apply(_one_day)

        return weights
