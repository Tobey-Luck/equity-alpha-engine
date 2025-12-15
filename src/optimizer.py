from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None


@dataclass
class OptimizerConfig:
    """
    Mean-variance style optimizer with basic constraints.

    Objective (conceptually):
        maximize   mu'w - risk_aversion * w' Sigma w - turnover_penalty * ||w - w_prev||_1
        subject to sum(w) = 0 (optional dollar-neutral)
                   sum(|w|) <= target_gross_leverage
                   |w_i| <= max_weight

    Notes
    -----
    - If CVXPY is available, we solve the constrained QP.
    - If the solver fails (or CVXPY is unavailable), we fall back to a stable
      closed-form ridge-regularized mean-variance weight, then project back to constraints.
    """
    target_gross_leverage: float = 1.0
    max_weight: float = 0.05
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.0
    enforce_dollar_neutral: bool = True


class PortfolioOptimizer:
    def __init__(self, cfg: Optional[OptimizerConfig] = None) -> None:
        self.cfg = cfg or OptimizerConfig()
        

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def optimize_day(
    self,
    mu: pd.Series,
    cov: pd.DataFrame,
    w_prev: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Optimize weights for a single date.

        Parameters
        ----------
        mu : Series
            Expected returns indexed by ticker.
        cov : DataFrame
            Covariance matrix (tickers x tickers).
        w_prev : Series, optional
            Previous-day weights indexed by ticker (used for turnover penalty).

        Returns
        -------
        Series
            Weights indexed by ticker, name="w".
        """
        if not isinstance(mu, pd.Series):
            raise TypeError("mu must be a pandas Series indexed by ticker.")
        if not isinstance(cov, pd.DataFrame):
            raise TypeError("cov must be a pandas DataFrame (tickers x tickers).")

        mu = mu.dropna().astype(float)
        if mu.empty:
            return pd.Series(dtype=float, name="w")

        # Align covariance to mu tickers and drop any missing rows/cols
        tickers = mu.index.intersection(cov.index).intersection(cov.columns)
        if len(tickers) < 2:
            return pd.Series(0.0, index=tickers, name="w")

        mu_vec = mu.loc[tickers]
        cov_d = cov.loc[tickers, tickers]

        w = self._solve_one_day(mu_vec, cov_d, w_prev=w_prev)

        w = w.reindex(tickers).fillna(0.0)
        w.name = "w"
        return w
    
    def optimize_time_series(
        self,
        exp_ret_panel: pd.Series | pd.DataFrame,
        returns: pd.DataFrame,
        cov_lookback_days: int = 60,
        min_cov_days: int = 40,
    ) -> pd.Series:
        """
        Optimize weights each date using rolling covariance estimated from returns.

        Parameters
        ----------
        exp_ret_panel
            Expected returns panel, indexed by MultiIndex (date, ticker).
            Can be:
              - Series named "mu" or "signal"
              - DataFrame with a single column (we take the first column)
        returns
            Wide daily returns DataFrame: index=date, columns=tickers
        cov_lookback_days
            Rolling lookback window length (in rows/dates of `returns`)
        min_cov_days
            Minimum non-NaN observations required per ticker inside the lookback
            window to include it in the covariance estimate.

        Returns
        -------
        Series
            MultiIndex (date, ticker) weights, name="w"
        """
        mu = self._normalize_expected_return_input(exp_ret_panel)
        mu = mu.sort_index()

        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index()

        all_dates = mu.index.get_level_values("date").unique()
        out_weights: List[pd.Series] = []

        w_prev: Optional[pd.Series] = None

        for d in all_dates:
            mu_d = mu.xs(d, level="date").dropna()
            if mu_d.empty:
                continue

            # Build covariance window ending at d (use history strictly before d)
            if d not in returns.index:
                # If returns doesn't contain that date (rare), skip
                continue

            d_loc = returns.index.get_loc(d)
            if isinstance(d_loc, slice):
                d_loc = d_loc.start

            start = max(0, d_loc - cov_lookback_days)
            hist = returns.iloc[start:d_loc]  # strictly before date d

            if hist.shape[0] < min_cov_days:
                # not enough history overall
                continue

            # Select tickers with enough non-NaN history and also in mu_d
            nn = hist.notna().sum(axis=0)
            tickers_ok = nn[nn >= min_cov_days].index
            tickers = mu_d.index.intersection(tickers_ok)

            if len(tickers) < 2:
                continue

            mu_vec = mu_d.loc[tickers].astype(float)

            # Covariance estimation (pairwise complete)
            hist_sub = hist.loc[:, tickers]
            cov = hist_sub.cov(min_periods=min_cov_days)

            # If covariance has NaNs (should be rare), drop problematic tickers
            good = cov.notna().all(axis=0) & cov.notna().all(axis=1)
            cov = cov.loc[good, good]
            mu_vec = mu_vec.loc[cov.index]

            if len(mu_vec) < 2:
                continue

            # Solve one-day optimization
            w_d = self._solve_one_day(mu_vec, cov, w_prev=w_prev)

            # Store weights with (date, ticker) index
            w_d.index.name = "ticker"
            w_d.name = "w"
            w_d = pd.concat({d: w_d}, names=["date"])
            out_weights.append(w_d)

            # update prev weights on the same ticker set for turnover penalty
            w_prev = w_d.droplevel("date")

        if not out_weights:
            return pd.Series(dtype=float, name="w")

        w_all = pd.concat(out_weights).sort_index()
        w_all.index = w_all.index.set_names(["date", "ticker"])
        w_all.name = "w"
        return w_all

    # ---------------------------------------------------------------------
    # Input normalization
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_expected_return_input(exp_ret_panel: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(exp_ret_panel, pd.DataFrame):
            if exp_ret_panel.shape[1] == 0:
                raise ValueError("exp_ret_panel DataFrame has no columns.")
            s = exp_ret_panel.iloc[:, 0].copy()
        else:
            s = exp_ret_panel.copy()

        if not isinstance(s.index, pd.MultiIndex):
            raise TypeError("exp_ret_panel must have MultiIndex (date, ticker).")

        if s.index.names != ["date", "ticker"]:
            s.index = s.index.set_names(["date", "ticker"])

        s = s.astype(float)
        s.name = "mu"
        return s

    # ---------------------------------------------------------------------
    # One-day solver (CVXPY if available, fallback otherwise)
    # ---------------------------------------------------------------------
    def _solve_one_day(
        self,
        mu: pd.Series,          # index tickers
        cov: pd.DataFrame,      # tickers x tickers
        w_prev: Optional[pd.Series],
    ) -> pd.Series:
        # Try CVXPY constrained solve if available
        if cp is not None:
            try:
                return self._solve_one_day_cvxpy(mu, cov, w_prev)
            except Exception:
                # Fall through to deterministic fallback
                return self._fallback_weights(mu, cov)

        # No CVXPY installed: use fallback
        return self._fallback_weights(mu, cov)

    def _solve_one_day_cvxpy(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        w_prev: Optional[pd.Series],
    ) -> pd.Series:
        tickers = mu.index
        Sigma = cov.loc[tickers, tickers].to_numpy(dtype=float)
        m = mu.to_numpy(dtype=float)

        n = len(tickers)
        w = cp.Variable(n)

        # Quadratic risk term
        risk = cp.quad_form(w, Sigma)

        # Turnover term: L1 norm of changes
        turnover = 0.0
        if self.cfg.turnover_penalty > 0.0 and w_prev is not None:
            wprev = w_prev.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            turnover = cp.norm1(w - wprev)

        obj = cp.Maximize(m @ w - float(self.cfg.risk_aversion) * risk - float(self.cfg.turnover_penalty) * turnover)

        constraints = []

        if self.cfg.enforce_dollar_neutral:
            constraints.append(cp.sum(w) == 0.0)

        constraints.append(cp.norm1(w) <= float(self.cfg.target_gross_leverage))
        constraints.append(w <= float(self.cfg.max_weight))
        constraints.append(w >= -float(self.cfg.max_weight))

        prob = cp.Problem(obj, constraints)

        # Use a stable default. OSQP handles QP well. ECOS can also work.
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            raise RuntimeError("CVXPY solver returned no solution.")

        w_val = np.asarray(w.value).reshape(-1)
        w_s = pd.Series(w_val, index=tickers, name="w")

        # Final cleanup: enforce constraints numerically
        w_s = self._project_and_rescale(w_s)
        return w_s

    # ---------------------------------------------------------------------
    # Deterministic fallback: ridge-regularized MV + projection
    # ---------------------------------------------------------------------
    def _fallback_weights(self, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        tickers = mu.index
        Sigma = cov.loc[tickers, tickers].to_numpy(dtype=float)
        m = mu.to_numpy(dtype=float)

        # Stabilize covariance
        eps = 1e-4
        Sigma = Sigma + eps * np.eye(Sigma.shape[0])

        # Regularization strength (bigger => more shrinkage)
        # Keep positive, and make it somewhat increasing when risk_aversion is small.
        lam = 1e-2 + 1.0 / max(1e-6, float(self.cfg.risk_aversion))

        A = Sigma + lam * np.eye(Sigma.shape[0])

        try:
            w = np.linalg.solve(A, m)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, m, rcond=None)[0]

        w_s = pd.Series(w, index=tickers, name="w")

        # Project to constraints
        w_s = self._project_and_rescale(w_s)
        return w_s

    # ---------------------------------------------------------------------
    # Constraint projection utilities
    # ---------------------------------------------------------------------
    def _project_and_rescale(self, w: pd.Series) -> pd.Series:
        w = w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Dollar-neutral
        if self.cfg.enforce_dollar_neutral and len(w) > 0:
            w = w - w.mean()

        # Box
        w = w.clip(-float(self.cfg.max_weight), float(self.cfg.max_weight))

        # Gross leverage
        gross = float(w.abs().sum())
        if np.isfinite(gross) and gross > 0.0:
            w = w * (float(self.cfg.target_gross_leverage) / gross)
        else:
            w[:] = 0.0

        return w