from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Dict

from main import log_returns
from explainability import (
    cvar_risk_contributions,
    variance_risk_contributions,
)


# ----------------------------------------------------------------- #
#  Sharpe-ratio contributions                                       #
# ----------------------------------------------------------------- #


def shapley_sharpe(R, rf=0.0, as_pct=False):
    from itertools import combinations

    n = R.shape[1]
    mu = R.mean(0).values
    cov = np.cov(R, rowvar=False)
    factorial = np.vectorize(math.factorial, otypes=[np.int64])

    def sr(weights):
        w = np.asarray(weights)
        num = w @ mu - rf
        den = np.sqrt(w @ cov @ w)
        return 0.0 if den == 0 else num / den

    phi = np.zeros(n)
    for i in range(n):
        for k in range(n):
            for S in combinations([j for j in range(n) if j != i], k):
                S = list(S)
                ws = np.zeros(n)
                ws[S] = 1 / len(S) if S else 0
                wsi = ws.copy()
                wsi[i] = 1 / (len(S) + 1)
                ws[S] *= len(S) / (len(S) + 1)
                phi[i] += sr(wsi) - sr(ws)
        phi[i] /= factorial(n)
    return pd.Series(phi, index=R.columns, name="Shapley Sharpe Δ")


def sharpe_contributions(
    R: pd.DataFrame | np.ndarray,
    weights: pd.Series | np.ndarray,
    rf: float = 0.0,
    as_pct: bool = False,  # <- default: raw Δ-Sharpe
):
    R = R.values if isinstance(R, pd.DataFrame) else np.asarray(R)
    w = weights.values if isinstance(weights, pd.Series) else np.asarray(weights)

    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)

    sigma_p = np.sqrt(w @ cov @ w)
    if sigma_p == 0:
        return pd.Series(0, index=weights.index, name="Sharpe Δ")

    sharpe = (w @ mu - rf) / sigma_p
    mc = (mu - rf) / sigma_p - sharpe * (cov @ w) / sigma_p**2
    contrib = w * mc

    if as_pct:
        total = np.abs(contrib).sum()
        if total > 0:
            contrib = 100 * contrib / total

    return pd.Series(contrib, index=weights.index, name="Sharpe Δ")


def walk_forward_explain(
    prices: pd.DataFrame,
    optimiser: Callable[[pd.DataFrame], Dict],
    init_cash: float = 100.0,
    freq: str = "M",
    label: str = "strategy",
    rf: int = 0,
    as_pct: bool = True,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward back-test **plus** risk-attribution logging.

    Returns
    -------
    equity_ser   : pd.Series  – cumulative equity curve
    weights_df   : pd.DataFrame  – weight matrix (dates × assets)
    risk_df      : pd.DataFrame  – Shapley risk-contributions (dates × assets, %)
    """
    periods = pd.period_range(prices.index.min(), prices.index.max(), freq=freq)

    eq_curve, w_log, rc_log = [], [], []
    equity = init_cash

    for i in range(len(periods) - 1):
        # 1) OPTIMISE with info up to current period-end
        R = log_returns(prices.loc[: periods[i].end_time])
        if R.empty:
            continue
        res = optimiser(R)
        w = res["weights"]

        # 1a)  ↳  Risk attribution on the *in-sample* data R
        # if "beta" in res:  # we assume CVaR optimiser returned β
        #     rc = cvar_risk_contributions(R, w, beta=res["beta"])
        # else:  # Mean-Variance or anything else
        #     rc = variance_risk_contributions(R.cov(), w)
        # rc = shapley_sharpe(R, rf=rf, as_pct=as_pct)
        rc = sharpe_contributions(R, w, rf=rf, as_pct=as_pct)
        rc.name = periods[i].end_time  # stamp the as-of date

        # 2) HOLD through next period
        hold_px = prices.loc[periods[i].end_time : periods[i + 1].end_time]
        if len(hold_px) < 2:
            continue
        ret = np.dot((hold_px.iloc[-1] / hold_px.iloc[0] - 1), w.fillna(0.0))
        equity *= 1 + ret

        eq_curve.append((periods[i + 1].end_time, equity))
        w_log.append((periods[i + 1].end_time, w))
        rc_log.append(rc)  # store the attribution

    # ---- wrap up  -------------------------------------------------------
    equity_ser = pd.Series(
        [v for _, v in eq_curve],
        index=pd.DatetimeIndex([d for d, _ in eq_curve], name="date"),
        name=label,
    )
    weights_df = (
        pd.concat([w.rename(d) for d, w in w_log], axis=1)
        .T.sort_index()
        .rename_axis("date")
    )
    risk_df = (
        pd.concat(rc_log, axis=1).T.sort_index().rename_axis("date")
    )  # rows = rebalance dates

    return equity_ser, weights_df, risk_df
