#!/usr/bin/env python3
"""
CVaR portfolio optimiser
========================
• Loads *long-format* CSV (columns: date, price, company, …)
• Pivots to a prices matrix  (dates × tickers)
• Turns prices → log-returns
• Minimises  CVaR_β of portfolio loss (Rockafellar & Uryasev formulation)
• Prints the weights, expected return, VaR and CVaR

Adjust the CONFIG section to suit your slice, confidence level, constraints, …
-----------------------------------------------------------------------------
"""
from pathlib import Path
import pandas as pd
import numpy as np
import cvxpy as cp

# ───── CONFIG ────────────────────────────────────────────────────────────── #
DATA_PATH = Path("stocks_clean_aligned_v2.csv")  # the file you just uploaded
START, END = "2015-02-01", "2015-02-10"
BETA = 0.95  # confidence level for CVaR
TARGET_RETURN = None  # e.g. 0.0001 (=0.01 % per period) or None
SHORT_CAP = 0.2  # e.g. 0.2 caps any single short at −20 %
SHOW_ABS_GT = 1e-4  # print weights with |w| ≥ 0.0001 (0.01 %)
# ─────────────────────────────────────────────────────────────────────────── #


def load_prices(
    path: Path,
    date_col: str = "date",
    price_col: str = "price",
    ticker_col: str = "company",
) -> pd.DataFrame:
    """Read CSV/Excel in long **or** wide form and return a wide price matrix."""
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # strip Excel’s “Unnamed: 0” artefact, if present
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # ensure datetime
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    else:  # treat first column as date
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        date_col = df.columns[0]

    # long form → pivot; else assume already wide
    if {ticker_col, price_col}.issubset(df.columns):
        wide = df.pivot_table(index=date_col, columns=ticker_col, values=price_col)
    else:
        wide = df.set_index(date_col)

    wide = wide.sort_index().apply(pd.to_numeric, errors="coerce")
    return wide


def slice_period(
    prices: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Date-filter or take the last N rows."""
    return prices.loc[start:end]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """1-period log-returns, dropping rows with *any* NaNs."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def optimise_cvar(
    R: pd.DataFrame,
    beta: float = 0.95,
    target_return: float | None = None,
    short_cap: float | None = None,
    relax_if_needed: bool = True,
) -> dict:
    """Rockafellar-&-Uryasev linear CVaR minimisation."""
    T, N = R.shape
    r, mu = R.values, R.mean().values

    w = cp.Variable(N)  # weights
    alpha = cp.Variable()  # VaR
    zeta = cp.Variable(T, nonneg=True)
    loss = -r @ w
    CVaR = alpha + cp.sum(zeta) / ((1 - beta) * T)

    cons = [cp.sum(w) == 1, zeta >= loss - alpha]
    if short_cap is not None:
        cons += [w >= -short_cap, w <= 1 + short_cap]

    if target_return is not None:
        cons.append(mu @ w >= target_return)

    prob = cp.Problem(cp.Minimize(CVaR), cons)
    prob.solve(solver=cp.ECOS)

    # retry without unreachable target_return
    if prob.status == "infeasible" and target_return and relax_if_needed:
        print(
            f"⚠️  Target return {target_return:.6%} infeasible ⇒ retrying unconstrained."
        )
        return optimise_cvar(R, beta, None, short_cap, False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed: {prob.status}")

    w_series = pd.Series(w.value, index=R.columns, name="weight")
    return {
        "weights": w_series,
        "expected_return": float(mu @ w.value),
        "VaR": alpha.value,
        "CVaR": CVaR.value,
        "status": prob.status,
    }


def main() -> None:
    prices = load_prices(DATA_PATH)
    prices_ = slice_period(prices, START, END)
    returns = log_returns(prices_)

    if returns.empty:
        raise ValueError("No return rows after slicing - check dates.")

    res = optimise_cvar(returns, BETA, TARGET_RETURN, SHORT_CAP)

    # — results —
    print("\nOptimal weights (|w| ≥ {:.5f}):".format(SHOW_ABS_GT))
    print(res["weights"][res["weights"].abs() >= SHOW_ABS_GT].round(6).to_string())
    print("\nSummary:")
    print(f"  expected return : {res['expected_return']:.6%} per period")
    print(f"  VaR @ {BETA:.0%}        : {res['VaR']:.6f}")
    print(f"  CVaR @ {BETA:.0%}       : {res['CVaR']:.6f}")
    print("  solver status   :", res["status"])


if __name__ == "__main__":
    main()
