from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


class AlphaModel:
    """
    In-sample cross-sectional regression model for predicting next-period
    returns from factor exposures.

    Inputs
    ------
    factor_panel : DataFrame
        MultiIndex (date, ticker), columns = factor names.
    adj_close : DataFrame
        Wide DataFrame of adjusted close prices (date x ticker).

    Outputs (from .fit)
    -------------------
    dict with keys:
        'betas'       : DataFrame (date x factor)
        'predictions' : Series with MultiIndex (date, ticker)
        'residuals'   : Series with MultiIndex (date, ticker)
        'ic'          : Series indexed by date (information coefficient)
    """

    def __init__(self, factor_columns: List[str]) -> None:
        self.factor_columns = factor_columns

    def fit(self, factor_panel: pd.DataFrame, adj_close: pd.DataFrame) -> Dict[str, Any]:
        # Forward returns
        ret_fwd_wide = adj_close.pct_change().shift(-1)
        ret_long = ret_fwd_wide.stack().rename("ret_fwd")
        ret_long.index.set_names(["date", "ticker"], inplace=True)

        # Merge factors and returns
        merged = factor_panel.join(ret_long, how="inner")
        merged = merged.sort_index()

        dates = merged.index.get_level_values("date").unique()
        factors = self.factor_columns

        betas = []
        predictions = []
        residuals = []
        pred_dates = []
        res_dates = []
        ic_values = []

        for d in dates:
            df_d = merged.xs(d, level="date").dropna()

            if len(df_d) < 2:
                continue

            X = df_d[factors].values
            y = df_d["ret_fwd"].values

            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            betas.append(pd.Series(beta, index=factors, name=d))

            y_hat = X @ beta
            pred = pd.Series(y_hat, index=df_d.index, name="pred")
            res = pd.Series(y - y_hat, index=df_d.index, name="res")

            predictions.append(pred)
            residuals.append(res)
            pred_dates.append(d)
            res_dates.append(d)

            if np.std(y) > 0 and np.std(y_hat) > 0:
                ic = np.corrcoef(y, y_hat)[0, 1]
                ic_values.append((d, ic))

        if not betas:
            return {
                "betas": pd.DataFrame(columns=factors),
                "predictions": pd.Series(dtype=float),
                "residuals": pd.Series(dtype=float),
                "ic": pd.Series(dtype=float, name="ic"),
            }

        betas_df = pd.DataFrame(betas)

        if predictions:
            predictions_panel = pd.concat(
                predictions, keys=pred_dates, names=["date", "ticker"]
            )
            residuals_panel = pd.concat(
                residuals, keys=res_dates, names=["date", "ticker"]
            )
        else:
            predictions_panel = pd.Series(dtype=float)
            residuals_panel = pd.Series(dtype=float)

        ic_series = pd.Series({d: ic for (d, ic) in ic_values}, name="ic")

        return {
            "betas": betas_df,
            "predictions": predictions_panel,
            "residuals": residuals_panel,
            "ic": ic_series,
        }


class AlphaModelOOS:
    """
    Out-of-sample cross-sectional regression model.

    For each date t we:
      1. Use a rolling history of previous dates (t - lookback_days ... t-1)
         to estimate factor betas from cross-sectional data.
      2. Apply those betas to the factor exposures on date t to obtain
         predicted next-period returns.

    This avoids using information from date t's realized returns when
    generating predictions for that date.
    """

    def __init__(
        self,
        factor_columns: List[str],
        lookback_days: int = 60,
        min_history_days: int = 40,
        min_stocks: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        factor_columns : list of str
            Names of factor columns to use as regressors.
        lookback_days : int
            Number of past *dates* to include in the regression window.
        min_history_days : int
            Minimum number of distinct past dates required to run a regression.
        min_stocks : int
            Minimum number of stocks required in both the history window
            and evaluation cross-section for a valid regression.
        """
        self.factor_columns = factor_columns
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self.min_stocks = min_stocks

    def fit(self, factor_panel: pd.DataFrame, adj_close: pd.DataFrame) -> Dict[str, Any]:
        # Forward returns
        ret_fwd_wide = adj_close.pct_change().shift(-1)
        ret_long = ret_fwd_wide.stack().rename("ret_fwd")
        ret_long.index.set_names(["date", "ticker"], inplace=True)

        # Merge factors and returns
        merged = factor_panel.join(ret_long, how="inner")
        merged = merged.sort_index()

        all_dates = merged.index.get_level_values("date").unique()
        factors = self.factor_columns

        betas = []
        beta_dates = []
        predictions = []
        residuals = []
        pred_dates = []
        res_dates = []
        ic_dict: Dict[pd.Timestamp, float] = {}

        for i, d in enumerate(all_dates):
            # History window: strictly before d
            start_idx = max(0, i - self.lookback_days)
            hist_dates = all_dates[start_idx:i]

            if len(hist_dates) < self.min_history_days:
                continue

            # Historical panel for regression
            hist_panel = merged.loc[(hist_dates, slice(None)), :]
            hist_panel = hist_panel.dropna(subset=factors + ["ret_fwd"])

            if hist_panel.empty:
                continue

            # Require enough stocks * days
            if len(hist_panel) < self.min_stocks * self.min_history_days:
                continue

            X_hist = hist_panel[factors].values
            y_hist = hist_panel["ret_fwd"].values

            beta, *_ = np.linalg.lstsq(X_hist, y_hist, rcond=None)
            betas.append(pd.Series(beta, index=factors, name=d))
            beta_dates.append(d)

            # Evaluation cross-section on date d
            df_d = merged.xs(d, level="date")
            df_d = df_d.dropna(subset=factors + ["ret_fwd"])

            if len(df_d) < self.min_stocks:
                continue

            X_d = df_d[factors].values
            y_d = df_d["ret_fwd"].values

            y_hat_d = X_d @ beta
            pred = pd.Series(y_hat_d, index=df_d.index, name="pred")
            res = pd.Series(y_d - y_hat_d, index=df_d.index, name="res")

            predictions.append(pred)
            residuals.append(res)
            pred_dates.append(d)
            res_dates.append(d)

            if np.std(y_d) > 0 and np.std(y_hat_d) > 0:
                ic_dict[d] = float(np.corrcoef(y_d, y_hat_d)[0, 1])

        if not betas:
            return {
                "betas": pd.DataFrame(columns=factors),
                "predictions": pd.Series(dtype=float),
                "residuals": pd.Series(dtype=float),
                "ic": pd.Series(dtype=float, name="ic"),
            }

        betas_df = pd.DataFrame(betas)
        betas_df.index.name = "date"

        if predictions:
            predictions_panel = pd.concat(
                predictions, keys=pred_dates, names=["date", "ticker"]
            )
            residuals_panel = pd.concat(
                residuals, keys=res_dates, names=["date", "ticker"]
            )
        else:
            predictions_panel = pd.Series(dtype=float)
            residuals_panel = pd.Series(dtype=float)

        ic_series = pd.Series(ic_dict, name="ic").sort_index()

        return {
            "betas": betas_df,
            "predictions": predictions_panel,
            "residuals": residuals_panel,
            "ic": ic_series,
        }
        
@dataclass
class AlphaModelOOSRidge:
    """
    Rolling out-of-sample (OOS) cross-sectional alpha model using Ridge regression.

    On each date t, we:
      1. Build a training sample from the previous `lookback_days` calendar days
         of factor exposures and *next-day* forward returns.
      2. Fit a Ridge regression of forward returns on factors.
      3. Apply the fitted beta to the cross-section of factors on date t to
         obtain predictions for the cross-section.

    Parameters
    ----------
    factor_columns : list of str
        Names of factor columns in the factor panel.
    lookback_days : int
        Number of calendar days of history used to estimate betas on each date.
    min_history_days : int
        Minimum number of distinct past dates required to run a regression.
    min_stocks : int
        Minimum number of stocks in the cross-section to run a regression.
    ridge_lambda : float
        Ridge regularization strength (lambda). Larger values mean more shrinkage.
    """

    factor_columns: List[str]
    lookback_days: int = 60
    min_history_days: int = 40
    min_stocks: int = 50
    ridge_lambda: float = 10.0

    def fit(
        self,
        factor_panel: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Fit rolling Ridge regressions and produce out-of-sample predictions.

        Parameters
        ----------
        factor_panel : DataFrame
            MultiIndex (date, ticker), columns must include `factor_columns`.
            In your notebook this is typically the z-scored factor panel.
        prices : DataFrame
            Wide DataFrame with DatetimeIndex and columns = tickers, used to
            compute forward returns.

        Returns
        -------
        dict
            {
              "predictions": Series with MultiIndex (date, ticker),
              "ic": DataFrame with columns ["ic"] indexed by date
            }
        """
        # Ensure datetime index for prices
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)

        prices = prices.sort_index()

        # Forward 1-day returns: r_{t+1} / r_t - 1, aligned so that
        # the factor on date t is used to predict return from t to t+1.
        fwd_ret = prices.pct_change().shift(-1)

        # Long form forward returns: index (date, ticker)
        fwd_long = fwd_ret.stack().rename("fwd_ret")

        # Ensure consistent MultiIndex names for the join
        if fwd_long.index.names != ["date", "ticker"]:
            fwd_long.index.set_names(["date", "ticker"], inplace=True)

        features = factor_panel[self.factor_columns].copy()
        if features.index.names != ["date", "ticker"]:
            features.index.set_names(["date", "ticker"], inplace=True)

        # Align factor panel with forward returns
        data = features.join(fwd_long, how="inner")


        # Ensure index names
        if data.index.names != ["date", "ticker"]:
            data.index.set_names(["date", "ticker"], inplace=True)

        all_dates = data.index.get_level_values("date")
        unique_dates = np.array(sorted(all_dates.unique()))

        predictions: List[pd.Series] = []
        ic_records: List[tuple] = []

        for current_date in unique_dates:
            # History window: (current_date - lookback_days, current_date)
            start_date = current_date - pd.Timedelta(days=self.lookback_days)
            hist_mask = (all_dates >= start_date) & (all_dates < current_date)
            hist = data[hist_mask]

            if hist.empty:
                continue

            hist_dates = hist.index.get_level_values("date").unique()
            if len(hist_dates) < self.min_history_days:
                continue

            # Training design matrix and response
            X_train = hist[self.factor_columns].to_numpy()
            y_train = hist["fwd_ret"].to_numpy()

            # Drop any rows with NaN features or responses
            mask_valid = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
            if mask_valid.sum() < self.min_stocks:
                continue

            X_train = X_train[mask_valid]
            y_train = y_train[mask_valid]

            # Center features (optional but improves conditioning)
            x_mean = X_train.mean(axis=0)
            Xc = X_train - x_mean

            # Ridge regression: (X'X + λI)⁻¹ X'y
            XtX = Xc.T @ Xc
            lam = float(self.ridge_lambda)
            XtX_reg = XtX + lam * np.eye(XtX.shape[0])

            XtY = Xc.T @ y_train
            beta = np.linalg.solve(XtX_reg, XtY)

            # Cross-section on current_date for prediction
            day_mask = all_dates == current_date
            day = data[day_mask]

            if len(day) < self.min_stocks:
                continue

            X_day = day[self.factor_columns].to_numpy()
            X_dayc = X_day - x_mean
            y_hat = X_dayc @ beta

            # Store predictions as a Series indexed by (date, ticker)
            idx_day = day.index
            pred_series = pd.Series(y_hat, index=idx_day, name="prediction")
            predictions.append(pred_series)

            # Compute daily IC for diagnostics
            y_day = day["fwd_ret"].to_numpy()
            # Drop NaNs before correlation
            mask_ic = np.isfinite(y_day) & np.isfinite(y_hat)
            if mask_ic.sum() >= self.min_stocks:
                y_d = y_day[mask_ic]
                p_d = y_hat[mask_ic]
                if np.std(y_d) > 0 and np.std(p_d) > 0:
                    ic = np.corrcoef(p_d, y_d)[0, 1]
                    ic_records.append((current_date, ic))

        if predictions:
            pred_all = pd.concat(predictions).sort_index()
            pred_all.name = "prediction"
        else:
            pred_all = pd.Series(dtype=float, name="prediction")

        if ic_records:
            ic_df = (
                pd.DataFrame(ic_records, columns=["date", "ic"])
                .set_index("date")
                .sort_index()
            )
        else:
            ic_df = pd.DataFrame(columns=["ic"])

        return {
            "predictions": pred_all,
            "ic": ic_df,
        }

@dataclass
class AlphaModelOOSRidgeEnsemble:
    """
    Ensemble of multiple rolling OOS Ridge models with different lookbacks
    and regularization strengths.

    For each (lookback, lambda) pair we:
      1. Fit a rolling Ridge model to predict next-day returns from factors.
      2. Collect out-of-sample predictions and daily IC.

    We then:
      - Average predictions across ensemble members for each (date, ticker).
      - Optionally compute an equal-weight average IC per date.

    Parameters
    ----------
    factor_columns : list of str
        Names of factor columns in the factor panel.
    lookbacks : list of int
        Rolling lookback windows (in calendar days) for each ensemble member.
    lambdas : list of float
        Ridge regularization strengths for each ensemble member.
        Must be the same length as `lookbacks`.
    min_history_days : int
        Minimum number of distinct past dates required to run a regression.
    min_stocks : int
        Minimum number of stocks in the cross-section to run a regression.
    """

    factor_columns: List[str]
    lookbacks: List[int]
    lambdas: List[float]
    min_history_days: int = 40
    min_stocks: int = 50

    def fit(
        self,
        factor_panel: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict[str, Any]:
        if len(self.lookbacks) != len(self.lambdas):
            raise ValueError("lookbacks and lambdas must have the same length.")

        member_preds: List[pd.Series] = []
        member_ic: List[pd.DataFrame] = []

        for lb, lam in zip(self.lookbacks, self.lambdas):
            model = AlphaModelOOSRidge(
                factor_columns=self.factor_columns,
                lookback_days=lb,
                min_history_days=self.min_history_days,
                min_stocks=self.min_stocks,
                ridge_lambda=lam,
            )
            res = model.fit(factor_panel, prices)

            pred = res["predictions"].rename(f"pred_lb{lb}_lam{lam}")
            member_preds.append(pred)

            ic = res["ic"]
            if isinstance(ic, pd.DataFrame):
                ic_col = ic["ic"].rename(f"ic_lb{lb}_lam{lam}")
            else:
                ic_col = ic.rename(f"ic_lb{lb}_lam{lam}")
            member_ic.append(ic_col.to_frame())

        if not member_preds:
            return {
                "predictions": pd.Series(dtype=float, name="prediction"),
                "ic_members": pd.DataFrame(),
                "ic_ensemble": pd.Series(dtype=float, name="ic_ensemble"),
            }

        # Align predictions into a DataFrame and average across columns
        pred_df = pd.concat(member_preds, axis=1).sort_index()
        pred_mean = pred_df.mean(axis=1)
        pred_mean.name = "prediction"

        # Align member ICs and compute equal-weight average per date
        ic_members = (
            pd.concat(member_ic, axis=1)
            .sort_index()
        )
        ic_ensemble = ic_members.mean(axis=1)
        ic_ensemble.name = "ic_ensemble"

        return {
            "predictions": pred_mean,
            "ic_members": ic_members,
            "ic_ensemble": ic_ensemble,
        }
        
@dataclass
class AlphaModelOOSRidgeSectorNeutral:
    """
    Rolling out-of-sample (OOS) cross-sectional Ridge model with *sector demeaning*
    applied inside each daily cross-section.

    For each date t:
      1) Build training data from the previous `lookback_days` calendar days.
      2) On each day inside the training window, demean factor exposures and forward
         returns within sector.
      3) Fit Ridge regression on the pooled (stacked) training data.
      4) On date t, demean factor exposures within sector and predict next-day returns.
      5) Compute daily IC against *raw* forward returns (or sector-demeaned; configurable).

    Notes
    -----
    - This is designed to be a separate class so Notebook 1 stays stable.
    - `sector_map` is a Series mapping ticker -> sector label.
    """

    factor_columns: List[str]
    sector_map: pd.Series
    lookback_days: int = 60
    min_history_days: int = 40
    min_stocks: int = 50
    ridge_lambda: float = 10.0
    min_names_per_sector: int = 3
    ic_on_sector_demeaned_returns: bool = False  # if True, IC uses sector-demeaned y too

    def _ensure_index_names(self, obj: Any, names: List[str]) -> Any:
        if getattr(obj, "index", None) is not None and list(obj.index.names) != names:
            obj = obj.copy()
            obj.index.set_names(names, inplace=True)
        return obj

    def _demean_within_sector(
        self,
        x: pd.Series,
        sector_for_index: pd.Series,
        min_names_per_sector: int,
    ) -> pd.Series:
        """
        Demean a Series indexed by ticker within sector.
        Sectors with too few names are treated as their own group and demeaned anyway
        (but this threshold helps avoid pathological tiny groups).
        """
        sec = sector_for_index.reindex(x.index).fillna("Unknown").astype(str)

        # Compute group sizes; used only to optionally guard tiny sectors
        sizes = sec.value_counts()
        small = sec.map(sizes) < min_names_per_sector

        # Demean within sector
        demeaned = x - x.groupby(sec).transform("mean")

        # If sector is tiny, optional choice: just global demean instead.
        # Here: apply a global demean for tiny sectors to reduce noise.
        if small.any():
            demeaned.loc[small] = x.loc[small] - x.loc[small].mean()

        return demeaned

    def fit(self, factor_panel: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, Any]:
        # --- prices -> forward returns ---
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        fwd_ret = prices.pct_change().shift(-1)
        fwd_long = fwd_ret.stack().rename("fwd_ret")
        fwd_long.index.set_names(["date", "ticker"], inplace=True)

        # --- features ---
        features = factor_panel[self.factor_columns].copy()
        features = self._ensure_index_names(features, ["date", "ticker"])

        # join
        data = features.join(fwd_long, how="inner")
        data = self._ensure_index_names(data, ["date", "ticker"])

        all_dates = data.index.get_level_values("date")
        unique_dates = np.array(sorted(all_dates.unique()))

        # ensure sector_map is ticker-indexed
        sector_map = self.sector_map.copy()
        sector_map.index = sector_map.index.astype(str)
        sector_map = sector_map.astype(str)

        predictions: List[pd.Series] = []
        ic_records: List[tuple] = []

        for current_date in unique_dates:
            # history window: [current_date - lookback_days, current_date)
            start_date = current_date - pd.Timedelta(days=self.lookback_days)
            hist_mask = (all_dates >= start_date) & (all_dates < current_date)
            hist = data[hist_mask]
            if hist.empty:
                continue

            hist_dates = hist.index.get_level_values("date").unique()
            if len(hist_dates) < self.min_history_days:
                continue

            # Build stacked training arrays by sector-demeaning per day
            X_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []

            # loop days in training window to demean within sector day-by-day
            for d in hist_dates:
                day = hist.xs(d, level="date")
                day = day.replace([np.inf, -np.inf], np.nan)
                day = day.dropna(subset=self.factor_columns + ["fwd_ret"])
                if len(day) < self.min_stocks:
                    continue

                # day index is tickers
                tickers = day.index.astype(str)
                sec_for_day = sector_map.reindex(tickers).fillna("Unknown")

                # sector-demean X columns
                X_day = []
                for col in self.factor_columns:
                    x = day[col].astype(float)
                    x_dm = self._demean_within_sector(
                        x, sec_for_day, self.min_names_per_sector
                    )
                    X_day.append(x_dm.to_numpy())
                X_day_mat = np.column_stack(X_day)

                # sector-demean y
                y = day["fwd_ret"].astype(float)
                y_dm = self._demean_within_sector(
                    y, sec_for_day, self.min_names_per_sector
                ).to_numpy()

                # validity mask
                m = np.isfinite(y_dm) & np.isfinite(X_day_mat).all(axis=1)
                if m.sum() < self.min_stocks:
                    continue

                X_list.append(X_day_mat[m])
                y_list.append(y_dm[m])

            if not X_list:
                continue

            X_train = np.vstack(X_list)
            y_train = np.concatenate(y_list)

            if X_train.shape[0] < self.min_stocks * 5:
                # not enough effective samples
                continue

            # center features for stability
            x_mean = X_train.mean(axis=0)
            Xc = X_train - x_mean

            # Ridge closed-form
            XtX = Xc.T @ Xc
            lam = float(self.ridge_lambda)
            XtX_reg = XtX + lam * np.eye(XtX.shape[0])
            XtY = Xc.T @ y_train
            try:
                beta = np.linalg.solve(XtX_reg, XtY)
            except np.linalg.LinAlgError:
                continue

            # Predict on current_date cross-section
            day_mask = all_dates == current_date
            day_full = data[day_mask]
            if len(day_full) < self.min_stocks:
                continue

            day_df = day_full.droplevel("date")  # index -> tickers
            day_df = day_df.replace([np.inf, -np.inf], np.nan)

            # need features for prediction; drop rows with NaNs in factors
            day_feat = day_df[self.factor_columns].astype(float)
            keep = np.isfinite(day_feat.to_numpy()).all(axis=1)
            if keep.sum() < self.min_stocks:
                continue

            day_feat = day_feat.loc[keep]
            tickers = day_feat.index.astype(str)
            sec_for_day = sector_map.reindex(tickers).fillna("Unknown")

            # sector-demean prediction-day features
            X_day_cols = []
            for col in self.factor_columns:
                x = day_feat[col]
                x_dm = self._demean_within_sector(
                    x, sec_for_day, self.min_names_per_sector
                )
                X_day_cols.append(x_dm.to_numpy())
            X_day = np.column_stack(X_day_cols)

            # apply training centering and predict
            X_dayc = X_day - x_mean
            y_hat = X_dayc @ beta

            # store predictions as MultiIndex (date, ticker)
            idx_day = pd.MultiIndex.from_product(
                [[current_date], day_feat.index],
                names=["date", "ticker"],
            )
            pred_series = pd.Series(y_hat, index=idx_day, name="prediction")
            predictions.append(pred_series)

            # IC diagnostics
            y_day_raw = day_df.loc[day_feat.index, "fwd_ret"].astype(float).to_numpy()
            if self.ic_on_sector_demeaned_returns:
                y_day = self._demean_within_sector(
                    pd.Series(y_day_raw, index=day_feat.index),
                    sec_for_day,
                    self.min_names_per_sector,
                ).to_numpy()
            else:
                y_day = y_day_raw

            m_ic = np.isfinite(y_day) & np.isfinite(y_hat)
            if m_ic.sum() >= self.min_stocks:
                yd = y_day[m_ic]
                pdh = y_hat[m_ic]
                if np.std(yd) > 0 and np.std(pdh) > 0:
                    ic = float(np.corrcoef(pdh, yd)[0, 1])
                    ic_records.append((current_date, ic))

        if predictions:
            pred_all = pd.concat(predictions).sort_index()
            pred_all.name = "prediction"
        else:
            pred_all = pd.Series(dtype=float, name="prediction")

        if ic_records:
            ic_df = (
                pd.DataFrame(ic_records, columns=["date", "ic"])
                .set_index("date")
                .sort_index()
            )
        else:
            ic_df = pd.DataFrame(columns=["ic"])

        return {"predictions": pred_all, "ic": ic_df}
    
@dataclass
class AlphaModelOOSRidgeKDay:
    """
    Rolling out-of-sample (OOS) cross-sectional Ridge model that predicts K-day forward returns.

    For each date t:
      1) Train on the prior lookback window using (factors_t, r_{t -> t+K})
      2) Predict on date t
      3) Report daily IC between predictions and realized K-day forward returns

    Notes
    -----
    - Forward return is computed as price_{t+K} / price_t - 1.
    - Uses a calendar-day lookback window (same convention as AlphaModelOOSRidge).
    """

    factor_columns: List[str]
    lookback_days: int = 60
    min_history_days: int = 40
    min_stocks: int = 50
    ridge_lambda: float = 10.0
    horizon_days: int = 5

    def fit(self, factor_panel: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, Any]:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        # K-day forward returns: P_{t+K}/P_t - 1
        fwd_ret = prices.shift(-self.horizon_days) / prices - 1.0
        fwd_long = fwd_ret.stack().rename("fwd_ret")
        if fwd_long.index.names != ["date", "ticker"]:
            fwd_long.index.set_names(["date", "ticker"], inplace=True)

        X = factor_panel[self.factor_columns].copy()
        if X.index.names != ["date", "ticker"]:
            X.index.set_names(["date", "ticker"], inplace=True)

        data = X.join(fwd_long, how="inner")
        if data.index.names != ["date", "ticker"]:
            data.index.set_names(["date", "ticker"], inplace=True)

        all_dates = data.index.get_level_values("date")
        unique_dates = np.array(sorted(all_dates.unique()))

        predictions: List[pd.Series] = []
        ic_records: List[tuple] = []

        for current_date in unique_dates:
            start_date = current_date - pd.Timedelta(days=self.lookback_days)
            hist_mask = (all_dates >= start_date) & (all_dates < current_date)
            hist = data[hist_mask]
            if hist.empty:
                continue

            hist_dates = hist.index.get_level_values("date").unique()
            if len(hist_dates) < self.min_history_days:
                continue

            X_train = hist[self.factor_columns].to_numpy()
            y_train = hist["fwd_ret"].to_numpy()

            mask_valid = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
            if mask_valid.sum() < self.min_stocks:
                continue

            X_train = X_train[mask_valid]
            y_train = y_train[mask_valid]

            x_mean = X_train.mean(axis=0)
            Xc = X_train - x_mean

            XtX = Xc.T @ Xc
            lam = float(self.ridge_lambda)
            XtX_reg = XtX + lam * np.eye(XtX.shape[0])
            XtY = Xc.T @ y_train
            beta = np.linalg.solve(XtX_reg, XtY)

            day = data[all_dates == current_date]
            if len(day) < self.min_stocks:
                continue

            X_day = day[self.factor_columns].to_numpy()
            X_dayc = X_day - x_mean
            y_hat = X_dayc @ beta

            pred_series = pd.Series(y_hat, index=day.index, name="prediction")
            predictions.append(pred_series)

            y_day = day["fwd_ret"].to_numpy()
            mask_ic = np.isfinite(y_day) & np.isfinite(y_hat)
            if mask_ic.sum() >= self.min_stocks:
                y_d = y_day[mask_ic]
                p_d = y_hat[mask_ic]
                if np.std(y_d) > 0 and np.std(p_d) > 0:
                    ic = float(np.corrcoef(p_d, y_d)[0, 1])
                    ic_records.append((current_date, ic))

        pred_all = (
            pd.concat(predictions).sort_index().rename("prediction")
            if predictions else pd.Series(dtype=float, name="prediction")
        )

        ic_df = (
            pd.DataFrame(ic_records, columns=["date", "ic"]).set_index("date").sort_index()
            if ic_records else pd.DataFrame(columns=["ic"])
        )

        return {"predictions": pred_all, "ic": ic_df}