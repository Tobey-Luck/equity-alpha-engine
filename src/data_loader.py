from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class DataConfig:
    data_dirname: str = "data"
    prices_subdir: str = "prices"
    raw_subdir: str = "raw"


class EquityDataLoader:
    """
    Handles downloading, caching, and loading daily equity prices.

    Price data is stored as one parquet file per ticker:

        data/prices/raw/{TICKER}.parquet

    and loaded into a wide DataFrame with columns = tickers.
    """

    def __init__(self, project_root: Path, config: Optional[DataConfig] = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = config or DataConfig()

        self.data_dir = self.project_root / self.config.data_dirname
        self.prices_raw_dir = self.data_dir / self.config.prices_subdir / self.config.raw_subdir

        self.prices_raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_adjusted_close(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load adjusted close prices for a set of tickers between start and end.

        Returns
        -------
        DataFrame
            Index: DatetimeIndex
            Columns: ticker symbols (wide format)

        Notes
        -----
        For large universes (e.g. S&P 500), some tickers may have missing or
        incomplete histories, or Yahoo Finance may occasionally return an empty
        DataFrame. In those cases the ticker is skipped rather than causing the
        entire load to fail.
        """
        tickers = list(tickers)
        frames: List[pd.Series] = []
        good_tickers: List[str] = []

        for ticker in tickers:
            try:
                s = self._load_single_ticker_adj_close(
                    ticker=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    force_refresh=force_refresh,
                )
            except (ValueError, KeyError) as e:
                # Log and skip tickers with no usable data
                print(f"[DataLoader] Skipping {ticker}: {e}")
                continue

            frames.append(s)
            good_tickers.append(ticker)

        if not frames:
            raise ValueError("No tickers with valid price data.")

        df = pd.concat(frames, axis=1)
        df.columns = good_tickers

        # Ensure sorted index and drop duplicate dates
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        return df


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _raw_price_path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_")
        return self.prices_raw_dir / f"{safe}.parquet"

    def _load_single_ticker_adj_close(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        force_refresh: bool,
    ) -> pd.Series:
        """
        Load a single ticker's adjusted close from cache or Yahoo.
        """
        path = self._raw_price_path(ticker)

        if path.exists() and not force_refresh:
            df = pd.read_parquet(path)
        else:
            df = self._download_from_yahoo(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
            )
            df.to_parquet(path)

        # Defensive cleaning
        if "Adj Close" in df.columns:
            s = df["Adj Close"].copy()
        elif "Close" in df.columns:
            s = df["Close"].copy()
        else:
            raise KeyError(f"No 'Adj Close' or 'Close' column for ticker {ticker}.")

        s.name = ticker
        s = s.sort_index()
        return s

    @staticmethod
    def _download_from_yahoo(
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Download OHLCV data for one ticker from Yahoo.
        """
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"Downloaded empty DataFrame for ticker {ticker}.")

        # Standardize index to DatetimeIndex naive UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df