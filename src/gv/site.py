"""Interactive dashboard: the GARCH lab for today's NIFTY 200 and NIFTY, as one static page.

Models are fitted here at build time (every stock, all three models, Student-t shocks, the last
five years); the page runs the recursions, forecasts and news-impact curves in the browser with
garch.js.
"""

import json
import re
import time
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from gv import models, plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 9h2l2-6 3 11 3-8 2 3h2%22 fill=%22none%22 stroke=%22%232a78d6%22"
    " stroke-width=%222%22 stroke-linejoin=%22round%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "garch.js").read_text(encoding="utf-8")
LAB = (HERE / "lab.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/garch-volatility"
SCALE = 1e4  # log returns stored as integers in units of 0.0001 (0.01%)
YEARS = 5
MIN_DAYS = 500  # about two years: fewer and the fit is not worth showing
DIST = "t"
KEEP = ["mu", "omega", "alpha", "gamma", "beta", "nu", "asym_weight", "abs_mean", "q01"]


def window_returns(adj: pd.DataFrame, years: int = YEARS) -> pd.DataFrame:
    """Daily log returns (fixes applied) over the last `years` years."""
    from gv.data import daily_returns

    r = np.log1p(daily_returns(adj))
    return r.loc[r.index[-1] - pd.DateOffset(years=years) :].iloc[1:]


def _round(v: float) -> float:
    return float(f"{v:.8g}")


def lab_data(adj: pd.DataFrame, years: int = YEARS) -> dict:
    """Integer log returns of NIFTY and each universe stock over the window, and each one's
    Student-t fit of all three models on exactly those (rounded) returns."""
    from gv.data import INDEX, SECTORS, SYMBOLS

    r = window_returns(adj, years)
    dates = r.index
    per_year = (len(dates) - 1) / ((dates[-1] - dates[0]).days / 365.25)
    data, dropped, failed = {}, [], []
    t0 = time.perf_counter()
    for s in [INDEX] + SYMBOLS:
        if s not in r or r[s].notna().sum() < MIN_DAYS:
            dropped.append(s)
            continue
        col = r[s]
        start = int(np.argmax(col.notna().to_numpy()))
        ints = np.round(col.iloc[start:].fillna(0.0).to_numpy() * SCALE).astype(int)
        pct = ints / (SCALE / 100)
        fits = {}
        for m in models.MODELS:
            f = models.fit(pct, m, DIST)
            fits[m] = {k: _round(f[k]) for k in KEEP} | {"converged": f["converged"]}
            if not f["converged"]:
                failed.append(f"{s} {m}")
        data[s] = {"start": start, "r": ints.tolist(), "fit": fits}
    seconds = time.perf_counter() - t0
    return {
        "symbols": list(data),
        "industries": {s: ("Index" if s == INDEX else SECTORS[s]) for s in data},
        "dates": list(dates.strftime("%Y-%m-%d")),
        "last": f"{dates[-1]:%d %b %Y}",
        "first": f"{dates[0]:%d %b %Y}",
        "scale": SCALE,
        "per_year": _round(per_year),
        "default": INDEX,
        "series": data,
        "dropped": dropped,
        "failed": failed,
        "fit_seconds": round(seconds, 1),
    }


def lab_section(d: dict) -> str:
    return LAB.replace("$js", JS).replace("$data", json.dumps(d, separators=(",", ":")))


def method_section(d: dict) -> str:
    dropped = ", ".join(d["dropped"]) or "none"
    failed = ", ".join(d["failed"]) or "none"
    return f"""<ul class="plain">
<li>Data: daily log returns of adjusted closes (dividends included for stocks; NIFTY is a price
index), {d["first"]} to {d["last"]}, with known Yahoo errors repaired. Stocks are today’s NIFTY 200
from Yahoo Finance; a stock listed later starts on its first day. Stocks with under
{MIN_DAYS} days in the window are left out ({dropped}).</li>
<li>Models: each has a constant mean and p = 1, q = 1. GARCH: tomorrow’s variance is a constant
plus α × today’s squared surprise plus β × today’s variance. GJR adds γ × the squared surprise on
down days only. EGARCH works on log variance, with γ times the signed surprise, so a fall and a
rise of the same size can move volatility differently.</li>
<li>Estimation: maximum likelihood with Student-t shocks scaled to unit variance, using the
<code>arch</code> package, on percent returns, refitted on every weekly rebuild. All parameters
are fitted on the same five years the charts show, so the charts are in-sample. Fits that did not
converge: {failed}.</li>
<li>Forecasts: GARCH and GJR forecasts are exact, with each later day moving toward the long-run
level at the rate of the persistence. GJR’s asymmetry term counts at half weight, because under a
symmetric t half of the shocks’ variance comes from down days. EGARCH has no closed form beyond
one day: the cone shows the exp of the expected log variance, which sits below the mean variance,
and more so at longer horizons.</li>
<li>Annualised volatility multiplies daily variance by {d["per_year"]:.0f}, the number of trading
days a year in this window, and takes the square root. The 20-day standard deviation is the plain
rolling sample standard deviation of the last 20 returns, the measure these models are meant to
improve on.</li>
<li>Value at risk: the loss on ₹1 lakh held for one day that the model expects to be exceeded on 1
day in 100, from tomorrow’s variance and the 1% quantile of the fitted unit-variance t
distribution.</li>
</ul>"""


SECTIONS = [
    (
        "GARCH lab",
        "Pick a stock, a model and a forecast horizon. The page runs the fitted model over the "
        "last five years of daily returns, then forecasts volatility forward. Research only, not "
        "investment advice: parameters are fitted on the same five years, so the charts are "
        "in-sample, and the stocks are today’s NIFTY 200.",
        lab_section,
    ),
    (
        "Method and caveats",
        "What the lab computes, and what it leaves out.",
        method_section,
    ),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(adj: pd.DataFrame, data: dict | None = None) -> str:
    d = data if data is not None else lab_data(adj)
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        body = make(d)
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    dk = plots.DARK
    return PAGE.substitute(
        icon=ICON,
        font=FONT,
        repo=REPO,
        updated=d["last"],
        surface=plots.SURFACE,
        text=plots.TEXT,
        text2=plots.TEXT_2,
        grid=plots.GRID,
        accent=plots.SERIES[0],
        accent2=plots.SERIES[1],
        muted=plots.MUTED,
        d_surface=dk[plots.SURFACE],
        d_text=dk[plots.TEXT],
        d_text2=dk[plots.TEXT_2],
        d_grid=dk[plots.GRID],
        d_accent=dk[plots.SERIES[0]],
        d_accent2=dk[plots.SERIES[1]],
        d_muted="#6b6a64",
        nav="\n".join(nav),
        sections="\n".join(parts),
    )
