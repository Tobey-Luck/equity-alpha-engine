from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EquityDataLoader, DataConfig


class FakeDownloadResult(pd.DataFrame):
    """
    Tiny helper so that isinstance(result.index, DatetimeIndex) holds.
    """
    @property
    def _constructor(self):
        return FakeDownloadResult


def fake_yahoo_download(*args, **kwargs):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    data = {
        "Open": np.arange(5, dtype=float),
        "High": np.arange(5, dtype=float) + 1.0,
        "Low": np.arange(5, dtype=float) - 1.0,
        "Close": np.arange(5, dtype=float) + 0.5,
        "Adj Close": np.arange(5, dtype=float) + 0.25,
        "Volume": np.ones(5, dtype=float) * 1000,
    }
    return FakeDownloadResult(data, index=idx)


@pytest.fixture
def tmp_loader(tmp_path, monkeypatch):
    # Patch yfinance.download before importing loader that uses it
    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_yahoo_download, raising=True)

    loader = EquityDataLoader(
        project_root=tmp_path,
        config=DataConfig(data_dirname="data", prices_subdir="prices", raw_subdir="raw"),
    )
    return loader


def test_load_adjusted_close_basic(tmp_loader):
    df = tmp_loader.load_adjusted_close(
        tickers=["AAA", "BBB"],
        start="2020-01-01",
        end="2020-01-10",
    )

    assert list(df.columns) == ["AAA", "BBB"]
    assert len(df) == 5
    assert df.notna().all().all()