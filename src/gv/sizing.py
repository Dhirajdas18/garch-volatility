"""Does a better volatility forecast pay? Two uses, monthly, after costs.

A. Sizing by risk: each month-end, hold that day's top-200 universe with weights in proportion to
   1 / forecast volatility (22-day forecasts from gv.horizons), against equal weights. A stock that
   stops trading during a month earns its return to its last price, then 0 (sold there). Costs:
   delivery charges and slippage per rupee traded on each side (`COST`), on the change from the
   drifted weights.
B. Volatility targeting: hold NIFTY (price index) at exposure min(cap, target / forecast volatility)
   for the next month, the rest in cash at 6.5% a year (exposure above 1 is financed at the same
   flat rate, with no spread); costs per rupee of exposure changed.
"""

import numpy as np
import pandas as pd

COST = (
    0.0017  # per side: STT 0.1%, stamp, exchange and GST (~0.12% buy / 0.10% sell) + 5 bp slippage
)
RF = 0.065


def monthly_returns(close: pd.DataFrame, month_ends: pd.Index) -> pd.DataFrame:
    """Return from each month-end to the next for every stock (0 if it has no next price)."""
    m = close.ffill().reindex(month_ends)
    last = close.apply(lambda c: c.last_valid_index())
    r = m.shift(-1) / m - 1
    for s in r.columns:  # after its last trade a stock is cash: 0, not a carried price
        if pd.notna(last[s]):
            r.loc[r.index >= last[s], s] = 0.0
    return r.fillna(0.0)


def inverse_vol(forecast: pd.DataFrame, rets: pd.DataFrame, cost: float = COST) -> dict:
    """Monthly net returns of weights ∝ 1/sqrt(forecast variance) (NaN forecast: not held), and
    the turnover. `forecast` is month-end × stock; equal weight if every forecast is 1."""
    dates = forecast.index.intersection(rets.index)
    w_prev = pd.Series(dtype=float)
    out, turn = [], []
    for t in dates:
        f = forecast.loc[t].dropna()
        f = f[f > 0]
        w = 1 / np.sqrt(f)
        w = w / w.sum()
        tv = float(w.subtract(w_prev, fill_value=0.0).abs().sum())
        r = rets.loc[t, w.index]
        g = float((w * r).sum())
        out.append(g - cost * tv)
        turn.append(tv)
        drifted = w * (1 + r)
        w_prev = drifted / drifted.sum() if drifted.sum() > 0 else drifted
    idx = rets.index[rets.index.get_indexer(dates) + 1]
    return {"net": pd.Series(out, index=idx), "turnover": pd.Series(turn, index=idx)}


def vol_target(
    r_daily: pd.Series,
    forecast_vol: pd.Series,
    target: float = 0.12,
    cap: float = 1.5,
    rf: float = RF,
    cost: float = 0.0005,
) -> pd.DataFrame:
    """Daily returns of NIFTY held at exposure min(cap, target / forecast) set at each month-end
    (annualised `forecast_vol`, decided at the close and held from the next day), cash at `rf`."""
    exp = (target / forecast_vol).clip(upper=cap)
    daily_exp = exp.reindex(r_daily.index).ffill().shift(1)
    days = r_daily.index.to_series().diff().dt.days.fillna(1).to_numpy()
    cash = rf * days / 365
    change = daily_exp.diff().abs().fillna(0.0)
    ret = daily_exp * r_daily + (1 - daily_exp) * cash - cost * change
    return pd.DataFrame({"ret": ret, "exposure": daily_exp}).dropna()


def nifty_forecasts(
    r_pct: pd.Series, month_ends: pd.Index, horizon: int = 22, years=range(2017, 2027)
) -> pd.DataFrame:
    """Annualised volatility forecasts of NIFTY over the next `horizon` days at each month-end:
    the 20-day standard deviation, EWMA (0.94), and GJR with Student-t shocks refitted each January
    on all returns before it (forecast = mean variance over the horizon)."""
    from gv.models import conditional_variance, fit, forecast_variance

    ann = np.sqrt(252) / 100
    rows = {}
    ewma = (r_pct**2).ewm(alpha=0.06, adjust=False).mean()
    for year in years:
        a = r_pct.loc[: pd.Timestamp(year - 1, 12, 31)]
        p = fit(a.to_numpy(), "GJR", "t")
        upto = r_pct.loc[: pd.Timestamp(year, 12, 31)]
        s2 = conditional_variance(upto.to_numpy(), p, "GJR")
        pos = {d: i for i, d in enumerate(upto.index)}
        for t in month_ends:
            if t.year != year or t not in pos:
                continue
            i = pos[t]
            state = {"resid": float(upto.iloc[i] - p["mu"]), "sigma2": float(s2[i])}
            gjr = float(forecast_variance(state, p, "GJR", horizon).mean())
            rows[t] = {
                "20-day": r_pct.loc[:t].iloc[-20:].std() * ann,
                "EWMA": np.sqrt(ewma.loc[t]) * ann,
                "GJR": np.sqrt(gjr) * ann,
            }
    return pd.DataFrame(rows).T
