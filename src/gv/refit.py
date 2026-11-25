"""Estimation in practice: how much fitted GARCH parameters move, how often to refit, and what to do
when a fit is unreliable.

For each test year Y (2020-2026), parameters are estimated from data before 1 January Y in several
ways and then used, frozen, through year Y (the variance recursion runs through the fitting window
first). Model: GJR with Student-t shocks.

- frozen: one fit on 2014-2019, used for every year.
- expanding: refit each year on everything from 2014.
- rolling 4y / rolling 2y: refit each year on the last 4 or 2 calendar years.
- common: the cross-sectional median of every rolling-4y parameter that year (alpha, gamma, beta,
  nu), with each stock's omega set so its long-run variance equals its own 4-year sample variance
  (variance targeting).
"""

import numpy as np
import pandas as pd

from gv.asym import pct_returns
from gv.models import fit, persistence, predictive_density

SCHEMES = ("frozen", "expanding", "rolling 4y", "rolling 2y", "common")


def _window(y0: int, y1: int) -> tuple:
    return pd.Timestamp(y0, 1, 1), pd.Timestamp(y1, 12, 31)


def unreliable(p: dict) -> bool:
    """A fit we wouldn't trust: not converged, or at an edge (beta below 0.5, persistence at or
    above 0.999, or Student-t degrees of freedom above 50, i.e. effectively normal)."""
    return (
        (not p["converged"])
        or p["beta"] < 0.5
        or p["persistence"] >= 0.999
        or (p["nu"] is not None and p["nu"] > 50)
    )


def score(r: pd.Series, p: dict, fit_win: tuple, year: int) -> dict | None:
    """Frozen-parameter log-likelihood and 1% VaR breaches over calendar `year`, with the recursion
    warmed up on `fit_win`."""
    a = r.loc[fit_win[0] : fit_win[1]]
    b = r.loc[pd.Timestamp(year, 1, 1) : pd.Timestamp(year, 12, 31)]
    if len(b) < 20 or len(a) < 250:
        return None
    ll, var, y = predictive_density(pd.concat([a, b]).to_numpy(), p, "GJR", len(a), 0.01)
    return {"ll": float(ll.sum()), "breaches": int(np.sum(y < var)), "days": len(b)}


def run(
    returns: dict[str, pd.Series], years=range(2020, 2027)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores, fits): one score row per (stock, year, scheme) and one fit row per (stock, year,
    window) with the parameters and whether the fit was unreliable."""
    fits, scores = [], []
    frozen = {
        s: fit(r.loc[_window(2014, 2019)[0] : _window(2014, 2019)[1]].to_numpy(), "GJR", "t")
        for s, r in returns.items()
    }
    for year in years:
        year_fits = {}
        for s, r in returns.items():
            wins = {
                "expanding": _window(2014, year - 1),
                "rolling 4y": _window(year - 4, year - 1),
                "rolling 2y": _window(year - 2, year - 1),
            }
            for name, w in wins.items():
                x = r.loc[w[0] : w[1]]
                if len(x) < 250:
                    continue
                p = fit(x.to_numpy(), "GJR", "t")
                year_fits[(s, name)] = (p, w)
                fits.append(
                    {"symbol": s, "year": year, "window": name, "unreliable": unreliable(p)}
                    | {
                        k: p[k]
                        for k in ("alpha", "gamma", "beta", "nu", "persistence", "converged")
                    }
                )
        four = [p for (s, n), (p, _) in year_fits.items() if n == "rolling 4y"]
        med = {k: float(np.median([p[k] for p in four])) for k in ("alpha", "gamma", "beta", "nu")}
        for s, r in returns.items():
            # frozen: 2014-2019 parameters, recursion warmed through every day before year Y
            plans = {"frozen": (frozen[s], _window(2014, year - 1))}
            for name in ("expanding", "rolling 4y", "rolling 2y"):
                if (s, name) in year_fits:
                    plans[name] = year_fits[(s, name)]
            if (s, "rolling 4y") in year_fits:
                own, w = year_fits[(s, "rolling 4y")]
                c = dict(own) | med
                c["persistence"] = persistence(c, "GJR")
                var4 = float(r.loc[w[0] : w[1]].var())
                c["omega"] = var4 * (1 - c["persistence"])
                plans["common"] = (c, w)
            for name, (p, w) in plans.items():
                sc = score(r, p, w, year)
                if sc:
                    scores.append({"symbol": s, "year": year, "scheme": name} | sc)
    return pd.DataFrame(scores), pd.DataFrame(fits)


def stock_returns(
    prices: pd.DataFrame, universe: pd.DataFrame, n: int = 200, min_cover: float = 0.95
) -> dict[str, pd.Series]:
    """The gv.asym stock set: top 200 on 31 Dec 2019 with returns on 95% of 2014-2019 days."""
    from gv.universe import members_on

    f0, f1 = _window(2014, 2019)
    days = prices.loc[f0:f1].index
    out = {}
    for s in members_on(universe, f1, n):
        if s in prices.columns:
            r = pct_returns(prices[s])
            if r.loc[f0:f1].size >= min_cover * (len(days) - 1) and r.loc["2020":].size >= 250:
                out[s] = r
    return out
