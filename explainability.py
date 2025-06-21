# ──────────────────────────────────────────────────────────────────────────
# explainability.py
# Shapley-style risk attribution + Plotly visualisation
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Iterable, Literal

# --------------------------------------------------------------------- #
# 1.  Risk-contribution helpers                                         #
# --------------------------------------------------------------------- #


def _cvar_tail_mask(loss: np.ndarray, beta: float) -> np.ndarray:
    """Boolean mask of scenarios inside the CVaR tail (loss ≥ VaR_β)."""
    var = np.quantile(loss, beta)
    return loss >= var


def cvar_risk_contributions(
    R: pd.DataFrame | np.ndarray,
    weights: pd.Series | np.ndarray,
    beta: float = 0.95,
    as_pct: bool = True,
) -> pd.Series:
    """
    Shapley-equivalent attribution for portfolio CVaR_β.

    Parameters
    ----------
    R        : (T, N) returns, *positive = gain*.
    weights  : length-N weight vector matching R's columns order.
    beta     : CVaR confidence level ∈ (0, 1).
    as_pct   : return percentages summing to 100 % (else raw values).

    Returns
    -------
    pd.Series indexed by asset names with contributions ≥ 0.
    """
    R = R.values if isinstance(R, pd.DataFrame) else np.asarray(R)
    w = weights.values if isinstance(weights, pd.Series) else np.asarray(weights)
    loss = -(R @ w)  # portfolio loss per scenario
    mask = _cvar_tail_mask(loss, beta)  # tail scenarios only
    tail_exp_losses = -R[mask].mean(axis=0)  # E[loss_i | tail]
    contrib = w * tail_exp_losses  # RC_i
    if as_pct:
        contrib = 100 * contrib / contrib.sum()
    index = (
        weights.index
        if isinstance(weights, pd.Series)
        else getattr(getattr(R, "columns", None), "tolist", lambda: None)()
    )
    return pd.Series(contrib, index=index, name=f"CVaR {int(beta*100)}")


def variance_risk_contributions(
    cov: pd.DataFrame | np.ndarray,
    weights: pd.Series | np.ndarray,
    as_pct: bool = True,
) -> pd.Series:
    """
    Shapley (Euler) attribution for portfolio **variance** (σ²).

    cov      : (N, N) covariance matrix.
    weights  : length-N weights.
    """
    cov = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)
    w = weights.values if isinstance(weights, pd.Series) else np.asarray(weights)
    # Marginal contribution: Σ w   (gradient of wᵀΣw w.r.t w_i)
    mc = cov @ w
    contrib = w * mc  # RC_i = w_i ∂σ²/∂w_i
    if as_pct:
        contrib = 100 * contrib / contrib.sum()
    index = weights.index if isinstance(weights, pd.Series) else None
    return pd.Series(contrib, index=index, name="Mean-Variance")


# --------------------------------------------------------------------- #
# 2.  Plotly visualisation                                              #
# --------------------------------------------------------------------- #


def plot_risk_contributions(
    contrib_dict: Dict[str, pd.Series],
    chart: Literal["bar", "pie"] = "bar",
    title: str | None = None,
):
    """
    Parameters
    ----------
    contrib_dict : mapping  {strategy_name: pd.Series of risk % per asset}
                   All series will be re-indexed to the union of their columns.
    chart        : "bar" → stacked-bar (default), "pie" → pie per strategy.
    """
    # — normalise & align —
    df = pd.concat(contrib_dict, axis=1).fillna(0).T  # index=strategy
    if chart == "bar":
        fig = px.bar(
            df,
            x=df.index,
            y=df.columns,
            text_auto=".2f",
            title=title or "Risk attribution (Shapley-style)",
        )
        fig.update_layout(barmode="stack", yaxis_title="Contribution (%)")
    elif chart == "pie":
        # one big pie where slice colour = asset, sector = strategy
        df_long = df.reset_index().melt(
            id_vars="index", var_name="Asset", value_name="Contribution"
        )
        fig = px.pie(
            df_long,
            names="Asset",
            values="Contribution",
            facet_col="index",
            title=title or "Risk attribution (Shapley-style)",
            hole=0.35,
        )
        fig.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{percent:.1%}",
            hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
        )
    else:
        raise ValueError("chart must be 'bar' or 'pie'")
    fig.update_layout(legend_title_text="Asset")
    return fig


# --------------------------------------------------------------------- #
# 3.  Convenience driver for your existing workflow                     #
# --------------------------------------------------------------------- #


def explain_strategies(
    strategies: Iterable[dict],
    chart: Literal["bar", "pie"] = "bar",
    title: str | None = None,
):
    """
    High-level wrapper.

    Each *strategy* dict **must** expose:
        • 'name'            human-readable label,
        • 'weights'         pd.Series (index = asset tickers),
        • 'returns'         pd.DataFrame of scenario returns (same order),
    and EITHER:
        • 'beta'            for CVaR strategies           OR
        • 'cov'             for Mean-Variance strategy.
    """
    contribs: Dict[str, pd.Series] = {}
    for strat in strategies:
        if "beta" in strat:  # CVaR
            rc = cvar_risk_contributions(
                strat["returns"], strat["weights"], beta=strat["beta"]
            )
        else:  # Mean-Variance
            cov = strat.get("cov") or strat["returns"].cov()
            rc = variance_risk_contributions(cov, strat["weights"])
        contribs[strat["name"]] = rc
    fig = plot_risk_contributions(contribs, chart=chart, title=title)
    return fig
