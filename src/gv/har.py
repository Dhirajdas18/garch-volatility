"""HAR on range-based daily variance against GARCH-family and simple forecasts, one day ahead.

Daily "realised" variance without intraday data: the Garman-Klass range estimator for the trading
day plus the squared overnight gap, ln(open / previous close)^2, in percent squared. It is less
noisy than the squared return but runs above it on average (median 1.17 times for these stocks in
2020-2026), and HAR is fitted to it, so rankings against it favour HAR; rankings against the
squared return are the fairer test.

Models, all fitted on the formation years and frozen through the test years:
- HAR (log): log rv(t+1) = b0 + b1 log rv(t) + b2 log mean5(t) + b3 log mean22(t), by least squares;
  the level forecast adds half the residual variance (lognormal mean).
- GJR and FIGARCH with Student-t shocks (arch), one-day variance.
- EWMA (lambda 0.94) and the 20-day sample variance of returns: no fitting.
"""

import numpy as np
import pandas as pd

from gv.models import garman_klass, har_features

FORECASTS = ("HAR", "GJR", "FIGARCH", "EWMA", "20-day")


def daily_rv(o, h, lo, c) -> pd.Series:
    """Garman-Klass plus the squared overnight gap, percent squared, floored at a tiny positive
    level (flat days give 0)."""
    gap = np.log(o / c.shift(1))
    rv = 1e4 * (pd.Series(garman_klass(o, h, lo, c), index=c.index) + gap**2)
    floor = 1e-3 * rv.median()
    return rv.clip(lower=floor)


def har_fit(rv: pd.Series) -> dict:
    """Log-HAR coefficients and residual variance by OLS on `rv` (features at t, target at t+1)."""
    x = np.log(har_features(rv))
    y = np.log(rv).shift(-1)
    d = pd.concat([x, y.rename("y")], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d[["daily", "weekly", "monthly"]].to_numpy()])
    b, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
    resid = d["y"].to_numpy() - X @ b
    return {"b": b, "s2": float(resid.var(ddof=X.shape[1]))}


def har_forecast(rv: pd.Series, p: dict) -> pd.Series:
    """Forecast of rv(t+1) made at t, indexed by t+1."""
    x = np.log(har_features(rv))
    f = np.exp(p["b"][0] + x.to_numpy() @ p["b"][1:] + p["s2"] / 2)
    return pd.Series(f, index=rv.index).shift(1)


def arch_forecast(r: pd.Series, form_end, model: str) -> pd.Series:
    """One-day variance for each day, from a Student-t fit on returns up to `form_end`, parameters
    frozen and filtered through all of `r` (arch's fixed-parameter filter)."""
    from arch import arch_model

    a = r.loc[:form_end]
    kw = (
        {"vol": "FIGARCH", "p": 1, "q": 1}
        if model == "FIGARCH"
        else {"vol": "GARCH", "p": 1, "o": 1, "q": 1}
    )
    res = arch_model(a.to_numpy(), mean="Constant", dist="t", rescale=False, **kw).fit(
        disp="off", show_warning=False, options={"maxiter": 1000}
    )
    full = arch_model(r.to_numpy(), mean="Constant", dist="t", rescale=False, **kw)
    cv = full.fix(res.params.to_numpy()).conditional_volatility
    return pd.Series(np.asarray(cv) ** 2, index=r.index), bool(res.convergence_flag == 0)


def simple_forecasts(r: pd.Series) -> dict[str, pd.Series]:
    """EWMA(0.94) and 20-day variance of returns, each known at t and used for t+1."""
    ewma = (r**2).ewm(alpha=0.06, adjust=False).mean().shift(1)
    roll = r.rolling(20).var().shift(1)
    return {"EWMA": ewma, "20-day": roll}


def qlike(target: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Patton's QLIKE loss per day: target/forecast − log(target/forecast) − 1 (0 when perfect)."""
    q = target / forecast
    return q - np.log(q) - 1


def run(stocks: list[str], ohlc: dict, form_end, test_start) -> pd.DataFrame:
    """`study_stock` for each symbol with open, high, low and close on the same days."""
    rows = []
    for s in stocks:
        cols = [ohlc[f][s].dropna() for f in ("adj_open", "adj_high", "adj_low", "adj_close")]
        idx = cols[0].index
        for c_ in cols[1:]:
            idx = idx.intersection(c_.index)
        res = study_stock(*(c_.loc[idx] for c_ in cols), form_end, test_start)
        if res:
            rows.append({"symbol": s} | res)
    return pd.DataFrame(rows)


def study_stock(o, h, lo, c, form_end, test_start) -> dict | None:
    """One-day forecasts on the test days and their QLIKE against the range-based proxy and
    against the squared return."""
    r = (100 * np.log(c / c.shift(1))).dropna()
    rv = daily_rv(o, h, lo, c).reindex(r.index)
    keep = r.index[rv.notna()]
    r, rv = r.loc[keep], rv.loc[keep]
    if r.loc[test_start:].size < 250 or r.loc[:form_end].size < 1000:
        return None
    hp = har_fit(rv.loc[:form_end])
    fc = {"HAR": har_forecast(rv, hp)}
    fc["GJR"], ok1 = arch_forecast(r, form_end, "GJR")
    fc["FIGARCH"], ok2 = arch_forecast(r, form_end, "FIGARCH")
    fc |= simple_forecasts(r)
    t = r.loc[test_start:].index
    frame = pd.DataFrame({k: v.reindex(t) for k, v in fc.items()}).dropna()
    rvt, r2 = rv.loc[frame.index].to_numpy(), (r.loc[frame.index] ** 2).to_numpy()
    r2 = np.clip(r2, 1e-3 * np.median(rvt), None)
    out = {
        "proxy_over_r2": float(rvt.mean() / (r.loc[frame.index] ** 2).mean()),
        "days": len(frame),
        "converged": ok1 and ok2,
        "b_daily": hp["b"][1],
        "b_weekly": hp["b"][2],
        "b_monthly": hp["b"][3],
    }
    for k in FORECASTS:
        out[f"{k}_qlike_rv"] = float(qlike(rvt, frame[k].to_numpy()).mean())
        out[f"{k}_qlike_r2"] = float(qlike(r2, frame[k].to_numpy()).mean())
        out[f"{k}_bias"] = float(np.mean(frame[k].to_numpy()) / np.mean(rvt))
    return out
