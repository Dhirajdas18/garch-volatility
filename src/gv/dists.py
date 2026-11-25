"""Which shock distribution to put inside GARCH: normal, Student-t or skewed-t.

Same stocks, windows and frozen-parameter scoring as gv.asym (GJR model): each distribution is
fitted on the formation years and scored on the test years by the log-likelihood of each day's
return and by how often the return fell below the model's 1% (and 5%) one-day value at risk.
"""

import numpy as np
import pandas as pd

from gv.asym import pct_returns
from gv.models import conditional_variance, fit, predictive_density

DISTS = ("normal", "t", "skewt")


def study_stock(
    r: pd.Series, form: tuple, test: tuple, model: str = "GJR", min_test: int = 250
) -> dict | None:
    a, b = r.loc[form[0] : form[1]], r.loc[test[0] : test[1]]
    if len(b) < min_test:
        return None
    both = pd.concat([a, b]).to_numpy()
    out = {"n_test": len(b)}
    vols = {}
    for d in DISTS:
        p = fit(a.to_numpy(), model, d)
        ll, var1, y = predictive_density(both, p, model, len(a), 0.01)
        _, var5, _ = predictive_density(both, p, model, len(a), 0.05)
        vols[d] = np.sqrt(conditional_variance(both, p, model)[len(a) :])
        out[d] = {
            "nu": p["nu"],
            "lambda": p["lambda"],
            "oos_ll": float(ll.mean()),
            "breach1": float(np.mean(y < var1)),
            "breach5": float(np.mean(y < var5)),
            "converged": p["converged"],
            "persistence": p["persistence"],
        }
    out["vol_corr_t_normal"] = float(np.corrcoef(vols["t"], vols["normal"])[0, 1])
    out["vol_ratio_t_normal"] = float(np.median(vols["t"] / vols["normal"]))
    return out


def run(
    prices,
    universe,
    fit_years=(2014, 2019),
    test_years=(2020, 2026),
    n=200,
    min_cover=0.95,
    model="GJR",
) -> pd.DataFrame:
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
        res = study_stock(r, (f0, f1), (t0, t1), model)
        if res is None:
            continue
        row = {
            "symbol": s,
            "n_test": res["n_test"],
            "vol_corr_t_normal": res["vol_corr_t_normal"],
            "vol_ratio_t_normal": res["vol_ratio_t_normal"],
        }
        for d in DISTS:
            row |= {f"{d}_{k}": v for k, v in res[d].items()}
        rows.append(row)
    return pd.DataFrame(rows)
