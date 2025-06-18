#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import cvxpy as cp

# ───── CONFIG ────────────────────────────────────────────────────────────── #
DATA_PATH = Path("stocks_clean_aligned_v2.csv")  # the file you just uploaded
START, END = "2015-02-01", "2019-12-31"
BETA = 0.95  # confidence level for CVaR
TARGET_RETURN = 0.0001  # e.g. 0.0001 (=0.01 % per period) or None
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


def simulate_top_n(
    prices: pd.DataFrame,
    n: int = 3,
    initial_cash: float = 1_000.0,
) -> pd.Series:
    """
    Run the Top-N momentum back-test and return a Series of portfolio values
    at each month-end tick (index = prices.index).
    """
    cash = initial_cash
    holdings: dict[str, float] = {}
    prev_px = prices.iloc[0]

    # create a Series to collect valuations
    vals = pd.Series(index=prices.index, dtype=float)
    vals.iloc[0] = initial_cash  # starting value at t0

    for idx in range(1, len(prices)):
        cur_px = prices.iloc[idx]

        # 1) Monthly returns
        monthly_ret = (cur_px / prev_px) - 1.0
        monthly_ret = monthly_ret.dropna()

        # 2) Pick Top-N
        top = monthly_ret.nlargest(n).index
        if len(top) == 0:
            vals.iloc[idx] = vals.iloc[idx - 1]  # carry forward
            prev_px = cur_px
            continue

        # 3) Liquidate
        if holdings:
            cash = sum(cur_px[t] * sh for t, sh in holdings.items())

        # 4) Re-invest equally
        allocation = cash / len(top)
        holdings = {t: allocation / cur_px[t] for t in top}

        # 5) Store valuation at this tick
        vals.iloc[idx] = sum(cur_px[t] * sh for t, sh in holdings.items())

        prev_px = cur_px

    return vals


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


def solve_cvxpy(objective, constraints, solver=None):
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=solver)
    return prob.status, prob.value, prob.variables()


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

    prob = solve_cvxpy(CVaR, cons, cp.ECOS)

    # retry without unreachable target_return
    if prob[0] == "infeasible" and target_return and relax_if_needed:
        print(
            f"⚠️  Target return {target_return:.6%} infeasible ⇒ retrying unconstrained."
        )
        return optimise_cvar(R, beta, None, short_cap, False)

    if prob[0] not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed: {prob[0]}")

    w_series = pd.Series(w.value, index=R.columns, name="weight")
    return {
        "weights": w_series,
        "expected_return": float(mu @ w.value),
        "VaR": alpha.value,
        "CVaR": CVaR.value,
        "status": prob[0],
    }


def optimize_mean_variance(
    R: pd.DataFrame,
    target_return: float | None = None,
    short_cap: float | None = None,
    relax_if_needed: bool = True,
) -> dict:
    """Mean-variance optimisation."""
    mu = R.mean().values
    Sigma = np.cov(R.values, rowvar=False)
    n = len(mu)

    w = cp.Variable(n)
    objective = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w) == 1]

    if short_cap is not None:
        constraints.append(w >= short_cap)

    if target_return is not None:
        target_constr = mu @ w >= target_return
        constraints.append(target_constr)

    try:
        status, _, _ = solve_cvxpy(objective, constraints)
        if w.value is None and relax_if_needed:
            # Relax target_return constraint if infeasible
            if target_return is not None:
                constraints = [c for c in constraints if c is not target_constr]
                status, _, _ = solve_cvxpy(objective, constraints)
        if w.value is None and relax_if_needed:
            # Relax short_cap constraint
            if short_cap is not None:
                constraints = [cp.sum(w) == 1]
                if target_return is not None:
                    constraints.append(mu @ w >= target_return)
                status, _, _ = solve_cvxpy(objective, constraints)
        if w.value is None:
            raise Exception(
                "Optimization failed: Problem infeasible even after relaxing constraints."
            )
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {e}")

    weights = np.array(w.value).flatten()
    exp_return = mu @ weights
    var = weights.T @ Sigma @ weights

    # Convert weights to pandas Series with the same index as input returns
    weights_series = pd.Series(weights, index=R.columns, name="weight")

    return {
        "weights": weights_series,
        "expected_return": float(exp_return),
        "variance": float(var),
        "status": status,
    }


def main() -> None:
    prices = load_prices(DATA_PATH)
    prices_ = slice_period(prices, START, END)
    returns = log_returns(prices_)

    if returns.empty:
        raise ValueError("No return rows after slicing - check dates.")

    res_cvar = optimise_cvar(returns, BETA, TARGET_RETURN, SHORT_CAP)
    res_mean_variance = optimize_mean_variance(returns, TARGET_RETURN, SHORT_CAP)

    # — results —
    print("CVaR Optimization Results:")
    print("--------------------------")
    print("\nOptimal weights (|w| ≥ {:.5f}):".format(SHOW_ABS_GT))
    print(
        res_cvar["weights"][res_cvar["weights"].abs() >= SHOW_ABS_GT]
        .round(6)
        .to_string()
    )
    print("\nSummary:")
    print(f"  expected return : {res_cvar['expected_return']:.6%} per period")
    print(f"  VaR @ {BETA:.0%}        : {res_cvar['VaR']:.6f}")
    print(f"  CVaR @ {BETA:.0%}       : {res_cvar['CVaR']:.6f}")
    print("  solver status   :", res_cvar["status"])

    print("\nMean-Variance Optimization Results:")
    print("-------------------------------------")
    print("\nOptimal weights (|w| ≥ {:.5f}):".format(SHOW_ABS_GT))
    print(
        res_mean_variance["weights"][res_mean_variance["weights"].abs() >= SHOW_ABS_GT]
        .round(6)
        .to_string()
    )
    print("\nSummary:")
    print(f"  expected return : {res_mean_variance['expected_return']:.6%} per period")
    print(f"  variance        : {res_mean_variance['variance']:.6f}")
    print("  solver status   :", res_mean_variance["status"])


if __name__ == "__main__":
    main()
