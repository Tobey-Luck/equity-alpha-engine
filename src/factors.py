from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorConfig:
    """
    Configuration for price-based factors.

    All windows are measured in trading days.

    Base components:

    - ret_{lag}d:
        prior `ret_lag`-day simple return.
    - mom_{N}d:
        N-day cumulative return.
    - vol_{N}d:
        realized volatility of daily returns over N days.

    Optional components:

    - rev_{K}d (short-term reversal):
        negative of K-day cumulative return, if enabled.
    - vov_{W}d (vol-of-vol):
        rolling standard deviation of realized volatility over W days.
    - mom_{M}d_over_vol_{V}d (volatility-adjusted momentum):
        momentum divided by realized volatility for specified (M, V) pairs.
    """

    # Daily return horizon
    ret_lag: int = 1

    # Momentum windows (e.g. ~1m, 3m, 12m)
    momentum_windows: List[int] | None = None

    # Realized volatility windows
    volatility_windows: List[int] | None = None

    # Optional extras (all False by default to keep old behavior)
    include_short_term_reversal: bool = False
    reversal_window: int = 5  # used if include_short_term_reversal

    include_vol_of_vol: bool = False
    # windows over which to compute vol-of-vol; if None, reuse volatility_windows
    vov_windows: List[int] | None = None

    include_vol_adj_momentum: bool = False
    # (momentum_window, vol_window) pairs; if None, pair each momentum_window with first volatility_window
    vol_adj_mom_pairs: List[Tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        # Default horizons if not provided
        if self.momentum_windows is None:
            # ~1m, 3m, 12m
            self.momentum_windows = [21, 63, 252]
        if self.volatility_windows is None:
            # ~1m, 3m
            self.volatility_windows = [21, 63]

        # Default vol-of-vol windows: reuse volatility windows
        if self.include_vol_of_vol and self.vov_windows is None:
            self.vov_windows = list(self.volatility_windows)

        # Default vol-adjusted momentum pairs:
        # pair each momentum window with the first volatility window
        if self.include_vol_adj_momentum and self.vol_adj_mom_pairs is None:
            v0 = self.volatility_windows[0]
            self.vol_adj_mom_pairs = [(m, v0) for m in self.momentum_windows]


class FactorCalculator:
    """
    Compute cross-sectional, price-based equity factors from daily adjusted closes.

    Input
    -----
    adj_close : DataFrame
        Wide DataFrame with DatetimeIndex (date) and columns = tickers.

    Output
    ------
    DataFrame
        MultiIndex (date, ticker), columns = factor names.
    """

    def __init__(self, config: FactorConfig) -> None:
        self.config = config

    def build_factor_panel(self, adj_close: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full factor panel as a long DataFrame with MultiIndex (date, ticker).
        """
        # Ensure datetime index and sorted
        df_prices = adj_close.copy()
        if not isinstance(df_prices.index, pd.DatetimeIndex):
            df_prices.index = pd.to_datetime(df_prices.index)
        df_prices = df_prices.sort_index()

        # Base daily returns for this horizon
        ret = df_prices.pct_change(self.config.ret_lag)

        factor_wide: dict[str, pd.DataFrame] = {}

        # ------------------------------------------------------------------
        # 1) ret_{lag}d : prior lag-day return
        # ------------------------------------------------------------------
        ret_name = f"ret_{self.config.ret_lag}d"
        factor_wide[ret_name] = ret

        # ------------------------------------------------------------------
        # 2) Momentum: mom_{N}d = N-day cumulative return
        # ------------------------------------------------------------------
        for m in self.config.momentum_windows:
            name = f"mom_{m}d"
            # Cumulative return over N days (last/first - 1)
            factor_wide[name] = df_prices.pct_change(periods=m)

        # ------------------------------------------------------------------
        # 3) Realized volatility: vol_{N}d over daily returns
        # ------------------------------------------------------------------
        for v in self.config.volatility_windows:
            name = f"vol_{v}d"
            factor_wide[name] = ret.rolling(window=v, min_periods=v).std()

        # ------------------------------------------------------------------
        # 4) Short-term reversal: rev_{K}d = - cumulative return over K days
        # ------------------------------------------------------------------
        if self.config.include_short_term_reversal:
            k = self.config.reversal_window
            name = f"rev_{k}d"
            factor_wide[name] = -df_prices.pct_change(periods=k)

        # ------------------------------------------------------------------
        # 5) Vol-of-vol: vov_{W}x{W}d = std of realized vol over window W
        # ------------------------------------------------------------------
        if self.config.include_vol_of_vol and self.config.vov_windows is not None:
            for w in self.config.vov_windows:
                # Name matches tests: e.g. vov_21x21d
                name = f"vov_{w}x{w}d"
                # First compute realized vol over window w
                vol_w = ret.rolling(window=w, min_periods=w).std()
                # Then compute std of that vol over the same window
                vov_w = vol_w.rolling(window=w, min_periods=w).std()
                factor_wide[name] = vov_w

        # ------------------------------------------------------------------
        # 6) Volatility-adjusted momentum: mom_{M}d_over_vol_{V}d
        # ------------------------------------------------------------------
        if self.config.include_vol_adj_momentum and self.config.vol_adj_mom_pairs is not None:
            for (m, v) in self.config.vol_adj_mom_pairs:
                mom_name = f"mom_{m}d"
                vol_name = f"vol_{v}d"
                out_name = f"mom_{m}d_over_vol_{v}d"

                mom_df = factor_wide.get(mom_name)
                vol_df = factor_wide.get(vol_name)

                if mom_df is None or vol_df is None:
                    # If either component is missing, skip this combo
                    continue

                # Avoid division by zero
                ratio = mom_df / vol_df.replace(0.0, np.nan)
                factor_wide[out_name] = ratio

        # ------------------------------------------------------------------
        # Stack all wide factors into a long panel (date, ticker)
        # ------------------------------------------------------------------
        frames: list[pd.Series] = []

        for name, wide in factor_wide.items():
            s = wide.stack().rename(name)  # MultiIndex (date, ticker)
            frames.append(s)

        panel = pd.concat(frames, axis=1)
        panel.index.set_names(["date", "ticker"], inplace=True)
        panel = panel.sort_index()

        return panel


# ----------------------------------------------------------------------
# Cross-sectional utilities
# ----------------------------------------------------------------------


def zscore_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally z-score each factor by date.

    Parameters
    ----------
    panel : DataFrame
        MultiIndex (date, ticker), columns = factors.

    Returns
    -------
    DataFrame
        Same index/columns, but each column is standardized within each date.

    Notes
    -----
    Missing values are preserved. If the cross-sectional standard deviation
    is zero for a factor on a given date, the z-scored values for that
    factor/date are set to NaN.
    """

    def _zscore(g: pd.DataFrame) -> pd.DataFrame:
        mu = g.mean(axis=0)
        # Use population std (ddof=0) to match tests
        sigma = g.std(axis=0, ddof=0).replace(0.0, np.nan)
        return (g - mu) / sigma


    return panel.groupby(level="date", group_keys=False).apply(_zscore)


def demean_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally demean each factor by date.

    Parameters
    ----------
    panel : DataFrame
        MultiIndex (date, ticker), columns = factors.

    Returns
    -------
    DataFrame
        Same index/columns, but each column has zero cross-sectional mean
        on each date (ignoring NaNs).
    """

    def _demean(g: pd.DataFrame) -> pd.DataFrame:
        mu = g.mean(axis=0)
        return g - mu

    return panel.groupby(level="date", group_keys=False).apply(_demean)