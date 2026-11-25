"""Market-wide volatility stress: how many stocks are far above their own normal volatility at
once, and whether that, India VIX or NIFTY's own GJR forecast best warns of next month's NIFTY
volatility.

Each January, GJR with Student-t shocks and common parameters (the cross-sectional median of
reliable fits on earlier data, as in gv.horizons) is set for every stock, variance-targeted to its
own history. Each day, a stock's ratio = its one-day conditional volatility ÷ its long-run
volatility. Stress breadth = the share of that day's top-200 universe with a ratio above
`threshold`; also the median ratio.
"""

import numpy as np
import pandas as pd

from gv.models import conditional_variance, dist_moments, fit, persistence
from gv.refit import unreliable


def daily_ratios(r_by_stock: dict[str, pd.Series], years=range(2017, 2027)) -> pd.DataFrame:
    """Day × stock ratio of conditional to long-run volatility, with parameters set each January
    from earlier data (common parameters for every stock)."""
    out = []
    for year in years:
        end = pd.Timestamp(year - 1, 12, 31)
        fits = []
        for r in r_by_stock.values():
            a = r.loc[:end]
            if len(a) >= 750:
                p = fit(a.to_numpy(), "GJR", "t")
                if not unreliable(p):
                    fits.append(p)
        med = {k: float(np.median([p[k] for p in fits])) for k in ("alpha", "gamma", "beta", "nu")}
        cols = {}
        for s, r in r_by_stock.items():
            a = r.loc[:end]
            if len(a) < 250:
                continue
            p = {"mu": float(a.mean()), "dist": "t", "lambda": None} | med
            p |= dist_moments("t", med["nu"])
            p["persistence"] = persistence(p, "GJR")
            lr = float(a.var())
            p["omega"] = lr * (1 - p["persistence"])
            upto = r.loc[: pd.Timestamp(year, 12, 31)]
            s2 = conditional_variance(upto.to_numpy(), p, "GJR")
            ratio = pd.Series(np.sqrt(s2 / lr), index=upto.index)
            cols[s] = ratio.loc[pd.Timestamp(year, 1, 1) :]
        out.append(pd.DataFrame(cols))
    return pd.concat(out).sort_index()


def breadth(
    ratios: pd.DataFrame, universe: pd.DataFrame, threshold: float = 1.5, n: int = 200
) -> pd.DataFrame:
    """Daily share of the top-`n` universe with a ratio above `threshold`, and the median ratio."""
    u = universe[universe["rank"] <= n]
    members = u.groupby("date")["symbol"].apply(set)
    rows = {}
    last = None
    for d in ratios.index:
        if d in members.index:
            last = members.loc[d]
        if last is None:
            continue
        x = ratios.loc[d, ratios.columns.intersection(list(last))].dropna()
        if len(x) >= 100:
            rows[d] = {
                "share": float((x > threshold).mean()),
                "median": float(x.median()),
                "stocks": len(x),
            }
    return pd.DataFrame(rows).T


def monthly_check(
    breadth_daily: pd.DataFrame,
    nifty: pd.Series,
    vix: pd.Series,
    forecasts: pd.DataFrame,
    days: int = 22,
) -> pd.DataFrame:
    """At each month-end: the predictors (NIFTY's GJR and 20-day forecasts, India VIX, stress share
    and median ratio) and what NIFTY did over the next `days` trading days (annualised volatility,
    return, worst fall from a running peak)."""
    r = np.log(nifty / nifty.shift(1)).dropna()
    rows = []
    for t in forecasts.index:
        if t not in r.index or t not in breadth_daily.index:
            continue
        k = r.index.get_loc(t)
        if k + days >= len(r):
            continue
        fut = r.iloc[k + 1 : k + 1 + days]
        path = np.exp(fut.cumsum())
        peak = np.maximum.accumulate(np.r_[1.0, path.to_numpy()])[1:]
        rows.append(
            {
                "date": t,
                "next vol": float(fut.std() * np.sqrt(252)),
                "next return": float(path.iloc[-1] - 1),
                "next worst fall": float((path.to_numpy() / peak - 1).min()),
                "NIFTY GJR": float(forecasts.at[t, "GJR"]),
                "NIFTY 20-day": float(forecasts.at[t, "20-day"]),
                "India VIX": float(vix.loc[:t].dropna().iloc[-1]),
                "stress share": float(breadth_daily.at[t, "share"]),
                "median ratio": float(breadth_daily.at[t, "median"]),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def partial_rank_corr(d: pd.DataFrame, x: str, y: str, given: str) -> float:
    """Rank correlation of x and y after removing the linear effect of `given` on both ranks."""
    rk = d[[x, y, given]].rank()
    g = np.column_stack([np.ones(len(rk)), rk[given].to_numpy()])

    def resid(v):
        b, *_ = np.linalg.lstsq(g, v, rcond=None)
        return v - g @ b

    return float(np.corrcoef(resid(rk[x].to_numpy()), resid(rk[y].to_numpy()))[0, 1])
