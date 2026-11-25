"""Volatility forecasts from one day to three months on the point-in-time universe.

Forecast origins: every month-end from January 2017. At each origin t the stocks are the universe's
top 200 that day. Each model forecasts the average daily return variance over the next h trading
days (h = 1, 5, 22, 63); the target is the realised average of squared daily returns over those days
(a target none of the models is fitted to). A stock that stops trading before t+h is skipped for
that horizon.

Models (parameters re-estimated every January on data before it, never after):
- 20-day: the sample variance of the last 20 returns, the same for every horizon.
- EWMA: RiskMetrics (lambda 0.94) on squared returns, the same for every horizon.
- GJR (own): GJR with Student-t shocks fitted on all of the stock's returns since 2013; if it has
  fewer than 750 or the fit is unreliable (gv.refit.unreliable), the common parameters below.
- GJR (common): the cross-sectional median alpha, gamma, beta, nu of that January's reliable own
  fits, with omega set so the long-run variance equals the stock's own variance since 2013.
- HAR: for each horizon, least squares of the realised average squared return over the next h days
  on the range-based daily variance, its 5-day and 22-day means (levels), fitted on windows that end
  before January; forecasts floored at a tenth of the stock's median variance.
"""

import numpy as np
import pandas as pd

from gv.har import daily_rv
from gv.models import dist_moments, fit, forecast_variance, har_features, persistence
from gv.refit import unreliable

HORIZONS = (1, 5, 22, 63)
MODELS = ("20-day", "EWMA", "GJR (own)", "GJR (common)", "HAR")


def _gjr_path(r: np.ndarray, p: dict) -> np.ndarray:
    """Conditional variance of each day (from the day before), vectorised enough for many stocks."""
    from gv.models import conditional_variance

    return conditional_variance(r, p, "GJR")


def _har_fits(rv: pd.Series, r2: pd.Series, end) -> dict:
    x = har_features(rv)
    out = {}
    for h in HORIZONS:
        y = r2.rolling(h).mean().shift(-h)  # mean of t+1..t+h, dated t
        d = pd.concat([x, y.rename("y")], axis=1).loc[:end].dropna()
        d = d.iloc[: max(len(d) - h, 0)]  # target windows must end before `end`
        if len(d) < 500:
            continue
        X = np.column_stack([np.ones(len(d)), d[["daily", "weekly", "monthly"]].to_numpy()])
        out[h], *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
    return out


def run(
    ohlc: dict, universe: pd.DataFrame, years=range(2017, 2027), start="2013-01-01", n: int = 200
) -> pd.DataFrame:
    """One row per (origin, stock, horizon, model) with the forecast and the realised target."""
    from gv.universe import members_on

    c = ohlc["adj_close"]
    month_ends = c.index.to_series().groupby(c.index.to_period("M")).max()
    rows = []
    members = {t: set(members_on(universe, t, n)) for t in month_ends if t.year in years}
    stocks = sorted(set().union(*members.values()))
    data = {}
    for s in stocks:
        cols = [ohlc[f][s].dropna() for f in ("adj_open", "adj_high", "adj_low", "adj_close")]
        idx = cols[0].index
        for col in cols[1:]:
            idx = idx.intersection(col.index)
        o, h, lo, cl = (col.loc[idx] for col in cols)
        r = (100 * np.log(cl / cl.shift(1))).dropna().loc[start:]
        rv = daily_rv(o, h, lo, cl).reindex(r.index)
        data[s] = (r, rv)

    for year in years:
        end = pd.Timestamp(year - 1, 12, 31)
        own, fallback = {}, []
        for s, (r, _) in data.items():
            a = r.loc[:end]
            if len(a) >= 750:
                p = fit(a.to_numpy(), "GJR", "t")
                if not unreliable(p):
                    own[s] = p
                    fallback.append(p)
        med = {
            k: float(np.median([p[k] for p in fallback])) for k in ("alpha", "gamma", "beta", "nu")
        }
        origins = [t for t in month_ends if t.year == year and t in members]
        if not origins:
            continue
        for s, (r, rv) in data.items():
            if not any(s in members[t] for t in origins):
                continue
            a = r.loc[:end]
            if len(a) < 250:
                continue
            common = {"mu": float(a.mean()), "dist": "t", "lambda": None} | med
            common |= dist_moments("t", med["nu"])
            common["persistence"] = persistence(common, "GJR")
            common["omega"] = float(a.var()) * (1 - common["persistence"])
            params = {"GJR (own)": own.get(s, common), "GJR (common)": common}
            upto = r.loc[: pd.Timestamp(year, 12, 31)]
            paths = {k: _gjr_path(upto.to_numpy(), p) for k, p in params.items()}
            har = _har_fits(rv.loc[:end], (r**2).loc[:end], end)
            feats = har_features(rv)
            r2 = r**2
            ewma = r2.ewm(alpha=0.06, adjust=False).mean()
            floor = 0.1 * float(rv.loc[:end].median())
            pos = {d: i for i, d in enumerate(upto.index)}
            for t in origins:
                if s not in members[t] or t not in pos:
                    continue
                i = pos[t]
                if i < 22:
                    continue
                flat = {
                    "20-day": float(r.iloc[: r.index.get_loc(t) + 1].iloc[-20:].var()),
                    "EWMA": float(ewma.loc[t]),
                }
                k = r.index.get_loc(t)
                for h in HORIZONS:
                    if k + h >= len(r):
                        continue
                    target = float(r2.iloc[k + 1 : k + 1 + h].mean())
                    fc = dict(flat)
                    for name, p in params.items():
                        state = {
                            "resid": float(upto.iloc[i] - p["mu"]),
                            "sigma2": float(paths[name][i]),
                        }
                        fc[name] = float(forecast_variance(state, p, "GJR", h).mean())
                    if h in har and not feats.loc[t].isna().any():
                        b = har[h]
                        fc["HAR"] = max(float(b[0] + feats.loc[t].to_numpy() @ b[1:]), floor)
                    for name, f in fc.items():
                        rows.append((t, s, h, name, f, target))
    return pd.DataFrame(rows, columns=["date", "symbol", "h", "model", "forecast", "target"])


def qlike(target, forecast, floor):
    y = np.maximum(target, floor)
    q = y / forecast
    return q - np.log(q) - 1


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Average QLIKE by horizon and model on the (origin, stock, horizon) cases every model
    forecast, its change against the 20-day variance, the Newey-West t-statistic of the monthly
    cross-sectional mean difference, and the share of months the model was better. The target is
    floored at 1% of that month's median target for the horizon."""
    w = df.pivot_table(index=["date", "symbol", "h"], columns="model", values="forecast").dropna()
    tgt = df.drop_duplicates(["date", "symbol", "h"]).set_index(["date", "symbol", "h"])["target"]
    tgt = tgt.loc[w.index]
    floor = tgt.groupby(level=["date", "h"]).transform(lambda x: 0.01 * x.median())
    y = np.maximum(tgt, floor)
    loss = pd.DataFrame({m: y / w[m] - np.log(y / w[m]) - 1 for m in w.columns})
    rows = []
    for h in sorted(loss.index.get_level_values("h").unique()):
        lh = loss.xs(h, level="h")
        per = lh.groupby(level="date").mean()
        for m in MODELS:
            d = (per[m] - per["20-day"]).to_numpy()
            lag = max(1, int(np.ceil(h / 22)))
            x = d - d.mean()
            s2 = x @ x / len(x)
            for k in range(1, lag + 1):
                s2 += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / len(x)
            t = d.mean() / np.sqrt(s2 / len(x)) if m != "20-day" else np.nan
            rows.append(
                {
                    "h": h,
                    "model": m,
                    "cases": len(lh),
                    "months": len(per),
                    "qlike": float(lh[m].mean()),
                    "vs 20-day": float(lh[m].mean() / lh["20-day"].mean() - 1),
                    "t": float(t),
                    "months better": float((d < 0).mean()),
                }
            )
    return pd.DataFrame(rows)
