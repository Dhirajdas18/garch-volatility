"""Asymmetric GARCH across the point-in-time universe: is bad news worth more volatility than good
news, does it repeat, and does modelling it help on years the fit never saw?

Formation = calendar years `fit_years`, test = `test_years`. Stocks are the universe's top 200 on
the formation's last day with returns on at least `min_cover` of the formation days. Each is fitted
on the formation with Student-t shocks, then scored on the test days with every parameter frozen
(the variance recursion runs through the formation days first, so it is warmed up). Stocks that stop
trading are scored on the test days they have.
"""

import numpy as np
import pandas as pd

from gv.models import fit, predictive_loglik

MODELS = ("GARCH", "GJR", "EGARCH")


def pct_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns in percent between consecutive prices (gaps skipped, not filled)."""
    p = prices.dropna()
    return (100 * np.log(p / p.shift(1))).dropna()


def study_stock(r: pd.Series, form: tuple, test: tuple, min_test: int = 250) -> dict | None:
    """Formation fits of the three models, their out-of-sample log-likelihood per day, and a GJR
    refit on the test years (for stability) when there are enough test days."""
    a = r.loc[form[0] : form[1]]
    b = r.loc[test[0] : test[1]]
    if len(b) < min_test:
        return None
    both = pd.concat([a, b]).to_numpy()
    out = {}
    for m in MODELS:
        p = fit(a.to_numpy(), m, "t")
        ll = predictive_loglik(both, p, m, len(a))
        out[m] = p | {"oos_ll": float(ll.mean())}
    out["GJR_test"] = fit(b.to_numpy(), "GJR", "t")
    out["n_test"] = len(b)
    return out


def run(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    fit_years=(2014, 2019),
    test_years=(2020, 2026),
    n: int = 200,
    min_cover: float = 0.95,
) -> pd.DataFrame:
    """One row per stock: formation parameters and t-stats, out-of-sample log-likelihood per day for
    each model, and the test-years GJR gamma."""
    from gv.universe import members_on

    f0, f1 = pd.Timestamp(fit_years[0], 1, 1), pd.Timestamp(fit_years[1], 12, 31)
    t0, t1 = pd.Timestamp(test_years[0], 1, 1), pd.Timestamp(test_years[1], 12, 31)
    days = prices.loc[f0:f1].index
    rows = []
    for s in members_on(universe, f1, n):
        if s not in prices.columns:
            continue
        r = pct_returns(prices[s])
        if r.loc[f0:f1].size < min_cover * (len(days) - 1):
            continue
        res = study_stock(r, (f0, f1), (t0, t1))
        if res is None:
            continue
        row = {"symbol": s, "n_test": res["n_test"]}
        for m in MODELS:
            p = res[m]
            row |= {
                f"{m}_{k}": p[k]
                for k in (
                    "alpha",
                    "gamma",
                    "beta",
                    "nu",
                    "persistence",
                    "aic",
                    "bic",
                    "loglik",
                    "oos_ll",
                    "converged",
                )
            }
        row["GJR_gamma_t"] = res["GJR"]["gamma_t"]
        row["EGARCH_gamma_t"] = res["EGARCH"]["gamma_t"]
        row["GJR_gamma_test"] = res["GJR_test"]["gamma"]
        row["GJR_gamma_t_test"] = res["GJR_test"]["gamma_t"]
        rows.append(row)
    return pd.DataFrame(rows)
