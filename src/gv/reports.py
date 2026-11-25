"""README findings, one function per topic: `gv report <topic>` writes charts to out/."""

import os
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gv import plots


def prices() -> pd.DataFrame:
    """Adjusted closes of the universe, NIFTY and the ETF with known Yahoo errors repaired."""
    from gv.data import DATA_PATH, apply_fixes

    return apply_fixes(pd.read_parquet(DATA_PATH)["adj_close"])


def returns() -> pd.DataFrame:
    """Simple daily returns of adjusted closes, with known Yahoo errors repaired."""
    from gv.data import DATA_PATH, daily_returns

    return daily_returns(pd.read_parquet(DATA_PATH)["adj_close"])


def last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    return df.loc[df.index[-1] - pd.DateOffset(years=years) :]


def report_asymmetry(out_dir: Path) -> dict:
    """Asymmetric GARCH (GJR, EGARCH) against plain GARCH, Student-t shocks: fitted on 2014-2019 for
    the universe's top 200 at the end of 2019, scored on 2020-2026 with frozen parameters; NIFTY
    alongside. Needs GV_LAKE_DIR. Archive prices (no dividends)."""
    from scipy.stats import ttest_1samp

    from gv.asym import pct_returns, run, study_stock
    from gv.models import news_impact
    from gv.universe import load_lake, load_universe

    res = {}
    px, _ = load_lake()
    st = run(px, load_universe())
    conv = st[[f"{m}_converged" for m in ("GARCH", "GJR", "EGARCH")]].all(axis=1)
    res["stocks"] = (
        f"{len(st)} (top 200 on 31 Dec 2019 with returns on 95% of 2014-2019 days); all fits "
        f"converged: {bool(conv.all())}; test days per stock median {int(st['n_test'].median())}"
    )
    g, t = st["GJR_gamma"], st["GJR_gamma_t"]
    res["GJR, 2014-2019"] = (
        f"gamma median {g.median():.3f} (alpha {st['GJR_alpha'].median():.3f}); positive for "
        f"{(g > 0).mean():.0%}; t > 2 for {(t > 2).mean():.0%}, t < -2 for {(t < -2).mean():.0%}"
    )
    eg, egt = st["EGARCH_gamma"], st["EGARCH_gamma_t"]
    res["EGARCH, 2014-2019"] = (
        f"asymmetry term median {eg.median():.3f}; negative (bad news raises volatility more) for "
        f"{(eg < 0).mean():.0%}; t < -2 for {(egt < -2).mean():.0%}"
    )
    res["in-sample choice over GARCH"] = (
        f"AIC prefers GJR for {(st['GJR_aic'] < st['GARCH_aic']).mean():.0%}, EGARCH for "
        f"{(st['EGARCH_aic'] < st['GARCH_aic']).mean():.0%}; BIC GJR "
        f"{(st['GJR_bic'] < st['GARCH_bic']).mean():.0%}, EGARCH "
        f"{(st['EGARCH_bic'] < st['GARCH_bic']).mean():.0%}"
    )
    diffs = {}
    for m in ("GJR", "EGARCH"):
        d = st[f"{m}_oos_ll"] - st["GARCH_oos_ll"]
        diffs[m] = d
        tt = ttest_1samp(d, 0.0)
        res[f"2020-2026, {m} minus GARCH log-likelihood a day"] = (
            f"median {d.median():+.4f}, mean {d.mean():+.4f} (t {tt.statistic:.1f} across stocks); "
            f"better for {(d > 0).mean():.0%}; over a median {int(st['n_test'].median())} days "
            f"that is {d.median() * st['n_test'].median():+.1f} log-likelihood points"
        )
    ok = st["GJR_gamma_test"].notna()
    rho = st.loc[ok, "GJR_gamma"].corr(st.loc[ok, "GJR_gamma_test"], method="spearman")
    res["GJR refitted on 2020-2026"] = (
        f"gamma median {st['GJR_gamma_test'].median():.3f}, t > 2 for "
        f"{(st['GJR_gamma_t_test'] > 2).mean():.0%}; rank correlation with 2014-2019 {rho:.2f}"
    )

    market = pd.read_parquet(Path(os.environ["GV_LAKE_DIR"]) / "market.parquet")
    r = pct_returns(market.set_index("date")["nifty"])
    nf = study_stock(
        r,
        (pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-12-31")),
    )
    res["NIFTY, 2014-2019 fit"] = (
        f"GJR alpha {nf['GJR']['alpha']:.3f}, gamma {nf['GJR']['gamma']:.3f} (t "
        f"{nf['GJR']['gamma_t']:.1f}), beta {nf['GJR']['beta']:.3f}; refit on 2020-2026 gamma "
        f"{nf['GJR_test']['gamma']:.3f}"
    )
    v = np.sqrt(news_impact(nf["GJR"], "GJR", np.array([-4.0, 4.0])) * 247)
    res["NIFTY GJR news impact"] = (
        f"after a 4% fall tomorrow's volatility {v[0]:.1f}% a year, after a 4% rise {v[1]:.1f}%"
    )
    res["NIFTY, 2020-2026 log-likelihood a day"] = (
        f"GJR minus GARCH {nf['GJR']['oos_ll'] - nf['GARCH']['oos_ll']:+.4f}, EGARCH minus GARCH "
        f"{nf['EGARCH']['oos_ll'] - nf['GARCH']['oos_ll']:+.4f} over {nf['n_test']} days"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    shocks = np.linspace(-4, 4, 161)
    for m in ("GARCH", "GJR"):
        v = news_impact(nf[m], m, shocks)
        axes[0].plot(shocks, np.sqrt(v * 247), color=plots.COLORS[m], lw=2)
        plots.label_line_end(axes[0], shocks[-1], np.sqrt(v[-1] * 247), m, plots.COLORS[m])
    axes[0].set_xlabel("today's NIFTY move, % (from its mean)")
    axes[0].set_ylabel("tomorrow's volatility, % a year")
    axes[0].set_title("NIFTY, fitted on 2014-2019", fontsize=10, fontweight="normal", loc="left")
    axes[0].margins(x=0.15)
    bins = np.linspace(-3, 6, 37)
    axes[1].hist(t.clip(-3, 6), bins=bins, color=plots.COLORS["GJR"], rwidth=0.9)
    axes[1].axvline(2, color=plots.TEXT_2, lw=0.8, ls="--")
    axes[1].axvline(nf["GJR"]["gamma_t"], color=plots.COLORS["NIFTY"], lw=1.5)
    axes[1].text(
        nf["GJR"]["gamma_t"] + 0.1,
        axes[1].get_ylim()[1] * 0.9,
        "NIFTY",
        fontsize=8,
        color=plots.TEXT_2,
    )
    axes[1].set_xlabel("t-statistic of the bad-news term (GJR gamma)")
    axes[1].set_ylabel("stocks")
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].set_title(
        f"{len(st)} stocks: {(t > 2).mean():.0%} clearly asymmetric",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.suptitle(
        "Bad news raised volatility more than good news: strongly for NIFTY, weakly for "
        "most single stocks",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "asymmetry_news")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for m, x in (("GJR", 0), ("EGARCH", 1)):
        d = 1000 * diffs[m]
        axes[0].boxplot(
            d,
            positions=[x],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": plots.COLORS[m], "alpha": 0.5},
            medianprops={"color": plots.TEXT},
        )
        axes[0].text(
            x,
            np.percentile(d, 75) + 0.4,
            f"better for {(diffs[m] > 0).mean():.0%}",
            ha="center",
            fontsize=8,
            color=plots.TEXT,
        )
    axes[0].axhline(0, color=plots.TEXT_2, lw=0.8)
    axes[0].set_xticks([0, 1], ["GJR", "EGARCH"])
    axes[0].set_ylabel("log-likelihood a day vs GARCH, ×1000")
    axes[0].set_title(
        "2020-2026, parameters from 2014-2019", fontsize=10, fontweight="normal", loc="left"
    )
    axes[1].scatter(
        st.loc[ok, "GJR_gamma"],
        st.loc[ok, "GJR_gamma_test"],
        s=10,
        color=plots.COLORS["GJR"],
        alpha=0.6,
        linewidths=0,
    )
    lim = [-0.1, 0.35]
    axes[1].plot(lim, lim, color=plots.TEXT_2, lw=0.8, ls="--")
    axes[1].set_xlim(*lim)
    axes[1].set_ylim(*lim)
    axes[1].set_xlabel("GJR gamma, 2014-2019")
    axes[1].set_ylabel("GJR gamma, 2020-2026")
    axes[1].set_title(
        f"Same stock, two periods: rank correlation {rho:.2f}",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "asymmetry_oos")
    return res


def _kupiec_fails(rate: float, n: int, p: float, level: float = 0.05) -> bool:
    """Kupiec's unconditional coverage test: does a breach rate `rate` over `n` days reject the
    promised rate `p` at `level`?"""
    from scipy.stats import chi2

    x = round(rate * n)
    if x in (0, n):
        lr = -2 * (n * np.log(1 - p)) if x == 0 else -2 * (n * np.log(p))
    else:
        ph = x / n
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph)
        )
    return bool(lr > chi2.ppf(1 - level, 1))


def report_distributions(out_dir: Path) -> dict:
    """Normal, Student-t and skewed-t shocks inside GJR-GARCH: fitted on 2014-2019 for the same
    172 stocks as the asymmetry study, scored on 2020-2026 with frozen parameters by log-likelihood
    and by one-day 1% and 5% value-at-risk breaches; NIFTY alongside. Needs GV_LAKE_DIR."""
    from gv.asym import pct_returns
    from gv.dists import DISTS, run, study_stock
    from gv.universe import load_lake, load_universe

    res = {}
    px, _ = load_lake()
    st = run(px, load_universe())
    res["stocks"] = (
        f"{len(st)}; all fits converged: {bool(st[[f'{d}_converged' for d in DISTS]].all().all())}"
    )
    res["Student-t degrees of freedom"] = (
        f"median {st['t_nu'].median():.1f} (10th-90th percentile {st['t_nu'].quantile(0.1):.1f}-"
        f"{st['t_nu'].quantile(0.9):.1f}); a normal distribution is infinite"
    )
    res["skewed-t lambda"] = (
        f"median {st['skewt_lambda'].median():+.3f}; negative (longer left tail) for "
        f"{(st['skewt_lambda'] < 0).mean():.0%}"
    )
    fails = {}
    for d in DISTS:
        f1 = np.mean(
            [
                _kupiec_fails(b, n, 0.01)
                for b, n in zip(st[f"{d}_breach1"], st["n_test"], strict=True)
            ]
        )
        f5 = np.mean(
            [
                _kupiec_fails(b, n, 0.05)
                for b, n in zip(st[f"{d}_breach5"], st["n_test"], strict=True)
            ]
        )
        fails[d] = (f1, f5)
        res[f"{d}, 2020-2026"] = (
            f"log-likelihood a day {st[f'{d}_oos_ll'].mean():.4f}; 1% VaR breached on "
            f"{100 * st[f'{d}_breach1'].mean():.2f}% of days (median stock "
            f"{100 * st[f'{d}_breach1'].median():.2f}%), Kupiec rejects for {f1:.0%} of stocks; "
            f"5% VaR breached {100 * st[f'{d}_breach5'].mean():.2f}%, rejects {f5:.0%}"
        )
    d1 = st["t_oos_ll"] - st["normal_oos_ll"]
    d2 = st["skewt_oos_ll"] - st["t_oos_ll"]
    res["Student-t minus normal"] = (
        f"better for {(d1 > 0).mean():.0%} of stocks, median {d1.median():+.4f} a day"
    )
    res["skewed-t minus Student-t"] = (
        f"better for {(d2 > 0).mean():.0%}, median {d2.median():+.5f} a day"
    )
    res["volatility path, Student-t against normal"] = (
        f"median correlation {st['vol_corr_t_normal'].median():.3f}, median level ratio "
        f"{st['vol_ratio_t_normal'].median():.3f}"
    )
    market = pd.read_parquet(Path(os.environ["GV_LAKE_DIR"]) / "market.parquet")
    nf = study_stock(
        pct_returns(market.set_index("date")["nifty"]),
        (pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-12-31")),
    )
    for d in DISTS:
        extra = ""
        if nf[d]["nu"] is not None:
            extra += f", nu {nf[d]['nu']:.1f}"
        if nf[d]["lambda"] is not None:
            extra += f", lambda {nf[d]['lambda']:+.3f}"
        res[f"NIFTY, {d}"] = (
            f"log-likelihood a day {nf[d]['oos_ll']:.4f}; 1% VaR breached "
            f"{100 * nf[d]['breach1']:.2f}% of {nf['n_test']} days{extra}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].hist(
        st["t_nu"].clip(upper=12),
        bins=np.arange(2, 12.5, 0.5),
        color=plots.COLORS["GJR"],
        rwidth=0.9,
    )
    axes[0].axvline(nf["t"]["nu"], color=plots.COLORS["NIFTY"], lw=1.5)
    axes[0].text(
        nf["t"]["nu"] + 0.15, axes[0].get_ylim()[1] * 0.9, "NIFTY", fontsize=8, color=plots.TEXT_2
    )
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_xlabel("Student-t degrees of freedom (lower = fatter tails)")
    axes[0].set_ylabel("stocks")
    axes[0].set_title(
        f"{len(st)} stocks, fitted on 2014-2019", fontsize=10, fontweight="normal", loc="left"
    )
    labels = {"normal": "normal", "t": "Student-t", "skewt": "skewed-t"}
    xs = np.arange(3)
    v = [100 * st[f"{d}_breach1"].mean() for d in DISTS]
    w = [100 * nf[d]["breach1"] for d in DISTS]
    axes[1].bar(xs - 0.2, v, 0.38, color=plots.COLORS["GJR"], label="stocks (average)")
    axes[1].bar(xs + 0.2, w, 0.38, color=plots.COLORS["NIFTY"], label="NIFTY")
    for i in range(3):
        axes[1].text(i - 0.2, v[i] + 0.03, f"{v[i]:.2f}", ha="center", fontsize=8, color=plots.TEXT)
        axes[1].text(i + 0.2, w[i] + 0.03, f"{w[i]:.2f}", ha="center", fontsize=8, color=plots.TEXT)
    axes[1].axhline(1.0, color=plots.TEXT_2, lw=0.8, ls="--")
    axes[1].set_xticks(xs, [labels[d] for d in DISTS])
    axes[1].set_ylabel("% of 2020-2026 days below the 1% VaR")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(
        "Promised 1% (dashed); frozen 2014-2019 parameters",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.suptitle(
        "Fat tails everywhere: a normal GARCH understated the 1% loss; Student-t hit it",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "distributions")
    return res


def report_refit(out_dir: Path) -> dict:
    """Estimation in practice for GJR with Student-t shocks on the 172 stocks of the asymmetry
    study: five ways to set parameters before each test year 2020-2026 (frozen 2014-2019 fit,
    yearly refits on expanding, 4-year and 2-year windows, and common cross-sectional parameters
    with each stock's own variance level), scored with frozen parameters through each year; how much
    fits move and how often they are unreliable. Needs GV_LAKE_DIR."""
    from gv.refit import SCHEMES, run, stock_returns
    from gv.universe import load_lake, load_universe

    res = {}
    px, _ = load_lake()
    sc, fi = run(stock_returns(px, load_universe()))
    n_schemes = sc.groupby(["symbol", "year"])["scheme"].transform("nunique")
    sc = sc[n_schemes == len(SCHEMES)]
    res["scored"] = (
        f"{sc['symbol'].nunique()} stocks, {sc.groupby(['symbol', 'year']).ngroups} stock-years "
        f"with all five schemes, 2020-2026"
    )
    tot = sc.groupby("scheme")[["ll", "days", "breaches"]].sum()
    per = sc.groupby(["scheme", "symbol"])[["ll", "days", "breaches"]].sum()
    out = {}
    for k in SCHEMES:
        row = tot.loc[k]
        st = per.loc[k]
        fails = np.mean(
            [_kupiec_fails(b / d, d, 0.01) for b, d in zip(st["breaches"], st["days"], strict=True)]
        )
        better = (
            st["ll"] / st["days"] > per.loc["frozen"]["ll"] / per.loc["frozen"]["days"]
        ).mean()
        out[k] = {
            "ll": row["ll"] / row["days"],
            "breach": row["breaches"] / row["days"],
            "fails": fails,
            "better": better,
        }
        res[k] = (
            f"log-likelihood a day {out[k]['ll']:.4f}; 1% VaR breached on "
            f"{100 * out[k]['breach']:.2f}% of days; Kupiec rejects {fails:.0%} of stocks"
            + ("" if k == "frozen" else f"; better than frozen for {better:.0%} of stocks")
        )
    for w in ("expanding", "rolling 4y", "rolling 2y"):
        f = fi[fi["window"] == w]
        res[f"fits on {w} windows"] = (
            f"unreliable {f['unreliable'].mean():.0%} "
            f"(not converged {(~f['converged']).mean():.1%}, "
            f"beta < 0.5 {(f['beta'] < 0.5).mean():.0%}, persistence >= 0.999 "
            f"{(f['persistence'] >= 0.999).mean():.0%}, nu > 50 {(f['nu'] > 50).mean():.0%}); "
            f"persistence median {f['persistence'].median():.3f}"
        )
        pv = f.pivot(index="year", columns="symbol", values="persistence")
        bv = f.pivot(index="year", columns="symbol", values="beta")
        res[f"year-to-year change, {w}"] = (
            f"persistence median {pv.diff().abs().stack().median():.3f}, beta "
            f"{bv.diff().abs().stack().median():.3f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = list(SCHEMES)
    xs = np.arange(len(names))
    base = out["frozen"]["ll"]
    v = [1000 * (out[k]["ll"] - base) for k in names]
    axes[0].bar(xs, v, color=[plots.SERIES[0] if x >= 0 else plots.MUTED for x in v], width=0.55)
    for i, x in enumerate(v):
        axes[0].text(
            i,
            x + (0.15 if x >= 0 else -0.15),
            f"{x:+.1f}",
            ha="center",
            va="bottom" if x >= 0 else "top",
            fontsize=8,
            color=plots.TEXT,
        )
    axes[0].axhline(0, color=plots.TEXT_2, lw=0.8)
    axes[0].set_xticks(
        xs, ["frozen\n2014-19", "expanding", "rolling\n4 years", "rolling\n2 years", "common"]
    )
    axes[0].set_ylabel("log-likelihood a day vs frozen, ×1000")
    axes[0].set_title(
        "2020-2026, refitted each January", fontsize=10, fontweight="normal", loc="left"
    )
    ws = ["expanding", "rolling 4y", "rolling 2y"]
    u = [100 * fi[fi["window"] == w]["unreliable"].mean() for w in ws]
    axes[1].bar(range(3), u, color=plots.SERIES[1], width=0.55)
    for i, x in enumerate(u):
        axes[1].text(i, x + 0.6, f"{x:.0f}%", ha="center", fontsize=9, color=plots.TEXT)
    axes[1].set_xticks(range(3), ["expanding\n(6-12 years)", "4 years", "2 years"])
    axes[1].set_ylabel("% of fits unreliable")
    axes[1].set_title(
        "Not converged, beta < 0.5, persistence ≥ 0.999 or ν > 50",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.suptitle(
        "More data beat fresher data; short windows gave unreliable fits", x=0.02, ha="left"
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "refit")
    return res


def report_har(out_dir: Path) -> dict:
    """One-day volatility forecasts on 2020-2026 for the 172 stocks of the asymmetry study, every
    model fitted on 2014-2019 and frozen: log-HAR on range-based daily variance, GJR-t, FIGARCH-t
    (long memory), EWMA and the 20-day variance; scored by QLIKE against the range-based proxy and
    against the squared return. Needs GV_LAKE_DIR."""
    from gv.har import FORECASTS, run
    from gv.refit import stock_returns
    from gv.universe import lake_ohlc, load_lake, load_universe

    res = {}
    px, _ = load_lake()
    stocks = list(stock_returns(px, load_universe()))
    st = run(stocks, lake_ohlc(), pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"))
    res["stocks"] = (
        f"{len(st)}; median {int(st['days'].median())} test days; GJR and FIGARCH converged for "
        f"{st['converged'].mean():.0%}"
    )
    res["HAR weights (median, log daily / weekly / monthly)"] = (
        f"{st['b_daily'].median():.2f} / {st['b_weekly'].median():.2f} / "
        f"{st['b_monthly'].median():.2f}"
    )
    q = {}
    for tgt, name in (("rv", "range-based variance"), ("r2", "squared return")):
        t = st[[f"{k}_qlike_{tgt}" for k in FORECASTS]]
        t.columns = list(FORECASTS)
        q[tgt] = t
        best = t.idxmin(axis=1).value_counts(normalize=True)
        res[f"QLIKE against {name}"] = ", ".join(
            f"{k} {t[k].mean():.3f} (best for {best.get(k, 0):.0%}, beats 20-day for "
            f"{(t[k] < t['20-day']).mean():.0%})"
            for k in FORECASTS
        )
    for tgt, name in (("rv", "range-based variance"), ("r2", "squared return")):
        base = q[tgt]["20-day"].mean()
        res[f"QLIKE vs the 20-day variance, {name}"] = ", ".join(
            f"{k} {100 * (q[tgt][k].mean() / base - 1):+.0f}%" for k in FORECASTS[:-1]
        )
    res["forecast level / proxy level (median)"] = ", ".join(
        f"{k} {st[f'{k}_bias'].median():.2f}" for k in FORECASTS
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(FORECASTS))
    for j, (tgt, c, lab) in enumerate(
        (
            ("rv", plots.SERIES[0], "against range-based variance"),
            ("r2", plots.MUTED, "against the squared return"),
        )
    ):
        base = q[tgt]["20-day"].mean()
        v = [100 * (q[tgt][k].mean() / base - 1) for k in FORECASTS]
        ax.bar(xs + (j - 0.5) * 0.38, v, 0.38, color=c, label=lab)
        for i, x in enumerate(v):
            ax.text(
                i + (j - 0.5) * 0.38,
                x - 1.2,
                f"{x:.0f}%",
                ha="center",
                va="top",
                fontsize=8,
                color=plots.TEXT,
            )
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(xs, ["HAR (range)", "GJR-t", "FIGARCH-t", "EWMA", "20-day"])
    ax.set_ylabel("average QLIKE vs the 20-day variance, %")
    ax.set_ylim(-45, 5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(
        f"{len(st)} stocks, one day ahead, 2020-2026, fitted on 2014-2019 (lower is better)",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "har")
    return res


def report_horizons(out_dir: Path) -> dict:
    """Forecasts of average daily variance over the next 1, 5, 22 and 63 trading days, made at every
    month-end from January 2017 for that day's top-200 universe (delisted stocks included), by the
    20-day variance, EWMA, GJR-t with each stock's own expanding fit, GJR-t with common parameters,
    and HAR; parameters set each January from earlier data; scored by QLIKE against realised squared
    returns. Needs GV_LAKE_DIR."""
    from gv.horizons import MODELS, run, score
    from gv.universe import lake_ohlc, load_universe

    res = {}
    df = run(lake_ohlc(), load_universe())
    sc = score(df)
    res["coverage"] = (
        f"{df['date'].nunique()} month-ends {df['date'].min():%b %Y}-{df['date'].max():%b %Y}, "
        f"{df['symbol'].nunique()} different stocks"
    )
    for h, g in sc.groupby("h"):
        g = g.set_index("model")
        res[f"{h}-day horizon ({int(g['cases'].iloc[0])} cases)"] = "; ".join(
            f"{m} {100 * g.at[m, 'vs 20-day']:+.0f}% (t {g.at[m, 't']:.1f}, better in "
            f"{g.at[m, 'months better']:.0%} of months)"
            for m in MODELS
            if m != "20-day"
        )

    fig, ax = plt.subplots(figsize=(10, 4.4))
    hs = sorted(sc["h"].unique())
    models = [m for m in MODELS if m != "20-day"]
    colors = {
        "EWMA": plots.MUTED,
        "GJR (own)": plots.COLORS["GJR"],
        "GJR (common)": plots.SERIES[3],
        "HAR": plots.SERIES[0],
    }
    w = 0.2
    for j, m in enumerate(models):
        v = [100 * sc[(sc["h"] == h) & (sc["model"] == m)]["vs 20-day"].iloc[0] for h in hs]
        x = np.arange(len(hs)) + (j - 1.5) * w
        ax.bar(x, v, w, color=colors[m], label=m)
        for xi, vi in zip(x, v, strict=True):
            ax.text(xi, vi - 1, f"{vi:.0f}", ha="center", va="top", fontsize=7, color=plots.TEXT)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(range(len(hs)), [f"{h} day" + ("s" if h > 1 else "") for h in hs])
    ax.set_ylabel("average QLIKE vs the 20-day variance, %")
    ax.set_ylim(-60, 5)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.set_title(
        f"Top 200 each month-end, {df['date'].min():%Y}-{df['date'].max():%Y}, against "
        "realised squared returns (lower is better)",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.suptitle("The further ahead, the more the 20-day variance lost", x=0.02, ha="left")
    fig.tight_layout()
    plots.save(fig, out_dir, "horizons")
    return res


def report_sizing(out_dir: Path) -> dict:
    """Does a better volatility forecast pay? (A) Inverse-volatility weights across each month's
    top-200 universe from the 22-day forecasts of the horizons study, against equal weights,
    after delivery costs; (B) a 12% volatility target on NIFTY (exposure capped at 1.5, cash
    6.5%) with the 20-day, EWMA and GJR-t forecasts, January 2017 onward. Needs GV_LAKE_DIR. No
    dividends."""
    from gv.horizons import run
    from gv.sizing import inverse_vol, monthly_returns, nifty_forecasts, vol_target
    from gv.universe import lake_ohlc, load_universe

    res = {}
    ohlc = lake_ohlc()
    df = run(ohlc, load_universe())
    d = df[df["h"] == 22]
    c = ohlc["adj_close"]
    me = pd.Index(c.index.to_series().groupby(c.index.to_period("M")).max().to_numpy())
    rets = monthly_returns(c, me)
    fc = {
        m: d[d["model"] == m].pivot_table(index="date", columns="symbol", values="forecast")
        for m in d["model"].unique()
    }
    both = None
    for f in fc.values():
        both = f.notna() if both is None else both & f.notna()

    def stats(x, per_year):
        eq = (1 + x).cumprod()
        yrs = (x.index[-1] - x.index[0]).days / 365.25 + 1 / per_year
        vol = x.std() * np.sqrt(per_year)
        return {
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "vol": vol,
            "sharpe": (x.mean() * per_year - 0.065) / vol,
            "dd": float((eq / eq.cummax() - 1).min()),
        }

    a = {"equal weight": inverse_vol(fc["20-day"].where(both) * 0 + 1, rets)}
    for m in ("20-day", "EWMA", "GJR (own)", "GJR (common)", "HAR"):
        a[m] = inverse_vol(fc[m].where(both), rets)
    sa = {}
    for k, v in a.items():
        sa[k] = stats(v["net"], 12) | {"turnover": float(v["turnover"].iloc[1:].mean() * 12)}
        res[f"A, {k}"] = (
            f"{100 * sa[k]['cagr']:.1f}% a year after costs, volatility {100 * sa[k]['vol']:.1f}%, "
            f"Sharpe {sa[k]['sharpe']:.2f}, worst drawdown {100 * sa[k]['dd']:.1f}%, traded "
            f"{sa[k]['turnover']:.2f}x the portfolio a year"
        )
    res["A, period"] = (
        f"{a['equal weight']['net'].index[0]:%b %Y}-{a['equal weight']['net'].index[-1]:%b %Y}"
    )

    market = pd.read_parquet(Path(os.environ["GV_LAKE_DIR"]) / "market.parquet")
    nifty = market.set_index("date")["nifty"].dropna()
    r = (100 * np.log(nifty / nifty.shift(1))).dropna()
    nme = pd.Index(nifty.index.to_series().groupby(nifty.index.to_period("M")).max().to_numpy())
    nf = nifty_forecasts(r, nme)
    rd = nifty.pct_change().dropna().loc[nf.index[0] :]
    paths = {"buy and hold": pd.DataFrame({"ret": rd, "exposure": 1.0})}
    for k in nf.columns:
        paths[f"target, {k}"] = vol_target(rd, nf[k])
    sb, mvol = {}, {}
    for k, v in paths.items():
        x = v["ret"]
        mv = x.groupby(x.index.to_period("M")).std() * np.sqrt(252)
        mvol[k] = mv
        sb[k] = stats(x, 252) | {
            "miss": float(np.sqrt(((mv - 0.12) ** 2).mean())),
            "hot": float((mv > 0.18).mean()),
        }
        res[f"B, {k}"] = (
            f"{100 * sb[k]['cagr']:.1f}% a year, volatility {100 * sb[k]['vol']:.1f}%, Sharpe "
            f"{sb[k]['sharpe']:.2f}, worst drawdown {100 * sb[k]['dd']:.1f}%; monthly volatility "
            f"off 12% by {100 * sb[k]['miss']:.1f} points (root mean square), above 18% in "
            f"{sb[k]['hot']:.0%} of months"
        )
    res["B, forecast spread (sd of the monthly forecasts)"] = ", ".join(
        f"{k} {100 * nf[k].std():.1f} points" for k in nf.columns
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = ["equal weight", "20-day", "EWMA", "GJR (own)", "GJR (common)", "HAR"]
    col = [
        plots.MUTED
        if k == "equal weight"
        else plots.COLORS["20-day"]
        if k == "20-day"
        else plots.SERIES[0]
        for k in names
    ]
    for ax, key, lab in (
        (axes[0], "turnover", "traded, times the portfolio a year"),
        (axes[1], "dd", "worst drawdown, %"),
    ):
        v = [sa[k][key] * (100 if key == "dd" else 1) for k in names]
        ax.barh(names[::-1], v[::-1], color=col[::-1], height=0.55)
        for i, x in enumerate(v[::-1]):
            if key == "dd":
                ax.text(
                    x - 0.5, i, f"{x:.0f}%", va="center", ha="right", fontsize=8, color=plots.TEXT
                )
            else:
                ax.text(x + 0.05, i, f"{x:.1f}", va="center", fontsize=8, color=plots.TEXT)
        ax.set_xlabel(lab)
    axes[1].set_xlim(-52, 0)
    axes[1].set_yticks([])
    fig.suptitle(
        "Sizing by risk: every forecast cut drawdowns alike; GJR and HAR traded half as "
        "much as the 20-day",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "sizing_inverse_vol")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    keys = ["buy and hold", "target, 20-day", "target, EWMA", "target, GJR"]
    data = [100 * mvol[k].dropna().to_numpy() for k in keys]
    bp = ax.boxplot(
        data,
        widths=0.5,
        showfliers=True,
        patch_artist=True,
        flierprops={
            "markersize": 3,
            "markerfacecolor": plots.MUTED,
            "markeredgecolor": plots.MUTED,
        },
        medianprops={"color": plots.TEXT},
    )
    for patch, k in zip(bp["boxes"], keys, strict=True):
        patch.set_facecolor(plots.SERIES[0] if k.endswith("GJR") else plots.MUTED)
        patch.set_alpha(0.6)
    ax.axhline(12, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_xticks(
        range(1, 5),
        ["NIFTY held", "target with\n20-day", "target with\nEWMA", "target with\nGJR-t"],
    )
    ax.set_ylabel("each month's volatility, % a year")
    ax.set_ylim(0, 60)
    ax.set_title(
        f"NIFTY, aiming at 12% (dashed), {rd.index[0]:%b %Y}-{rd.index[-1]:%b %Y}",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.suptitle(
        "Volatility targeting: GJR came closest to the target and cut the worst fall most",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "sizing_target")
    return res


def report_regime(out_dir: Path) -> dict:
    """Market-wide volatility stress, 2017-2026: the daily share of the top-200 universe whose GJR
    volatility (common parameters) is above 1.5 times its own long-run level, and the median ratio;
    at each month-end, how well it, India VIX, NIFTY's GJR and 20-day forecasts ranked NIFTY's
    volatility and returns over the next 22 trading days. Needs GV_LAKE_DIR."""
    from scipy.stats import spearmanr

    from gv.regime import breadth, daily_ratios, monthly_check, partial_rank_corr
    from gv.sizing import nifty_forecasts
    from gv.universe import lake_ohlc, load_universe

    res = {}
    ohlc = lake_ohlc()
    uni = load_universe()
    c = ohlc["adj_close"]
    syms = sorted(set(uni[(uni["date"] >= "2017-01-01") & (uni["rank"] <= 200)]["symbol"]))
    rets = {s: (100 * np.log(c[s] / c[s].shift(1))).dropna() for s in syms if s in c.columns}
    b = breadth(daily_ratios(rets), uni)
    market = pd.read_parquet(Path(os.environ["GV_LAKE_DIR"]) / "market.parquet").set_index("date")
    nifty = market["nifty"].dropna()
    r = (100 * np.log(nifty / nifty.shift(1))).dropna()
    me = pd.Index(nifty.index.to_series().groupby(nifty.index.to_period("M")).max().to_numpy())
    d = monthly_check(b, nifty, market["vix"], nifty_forecasts(r, me))
    res["daily stress share"] = (
        f"{len(b)} days {b.index[0]:%b %Y}-{b.index[-1]:%b %Y}; "
        f"median {100 * b['share'].median():.1f}% "
        f"of stocks above 1.5x their normal volatility, 90th percentile "
        f"{100 * b['share'].quantile(0.9):.1f}%; peak {100 * b['share'].max():.1f}% on "
        f"{b['share'].idxmax():%d %b %Y}"
    )
    for label, a, z in (
        ("March-April 2020", "2020-03-01", "2020-04-30"),
        ("January-March 2025", "2025-01-01", "2025-03-31"),
        ("June 2024", "2024-06-01", "2024-06-30"),
    ):
        w = b.loc[a:z]
        res[f"peak stress share, {label}"] = (
            f"{100 * w['share'].max():.1f}% on {w['share'].idxmax():%d %b %Y}"
        )
    preds = ["India VIX", "NIFTY 20-day", "median ratio", "NIFTY GJR", "stress share"]
    rho = {}
    for k in preds:
        rho[k] = {
            y: spearmanr(d[k], d[y]).statistic
            for y in ("next vol", "next return", "next worst fall")
        }
        res[f"{k}, rank correlation with the next 22 days"] = (
            f"volatility {rho[k]['next vol']:.2f}, return {rho[k]['next return']:+.2f}, worst "
            f"fall {rho[k]['next worst fall']:+.2f}"
        )
    part = {
        k: partial_rank_corr(d, k, "next vol", "NIFTY GJR")
        for k in ("India VIX", "median ratio", "stress share")
    }
    res["beyond NIFTY's GJR (partial rank correlation with next volatility)"] = ", ".join(
        f"{k} {v:.2f}" for k, v in part.items()
    )
    d["tercile"] = pd.qcut(d["median ratio"], 3, labels=["calm", "middle", "stressed"])
    g = d.groupby("tercile", observed=True)
    for name, row in g.agg(
        v=("next vol", "mean"),
        r=("next return", "mean"),
        f=("next worst fall", "mean"),
        n=("next return", "size"),
    ).iterrows():
        res[f"next 22 days after a {name} month-end (median ratio terciles)"] = (
            f"{row['n']:.0f} months: NIFTY volatility {100 * row['v']:.1f}%, return "
            f"{100 * row['r']:+.1f}%, worst fall {100 * row['f']:.1f}%"
        )
    res["months"] = f"{len(d)} month-ends {d.index[0]:%b %Y}-{d.index[-1]:%b %Y}"

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, height_ratios=[1.2, 1])
    axes[0].fill_between(b.index, 0, 100 * b["share"], color=plots.SERIES[1], alpha=0.5, lw=0)
    axes[0].plot(b.index, 100 * b["share"], color=plots.SERIES[1], lw=0.8)
    axes[0].set_ylabel("% of top 200 above\n1.5x their normal")
    axes[0].set_title(
        "Stress breadth: stocks whose GJR volatility is far above their own normal",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    n = nifty.loc[b.index[0] :]
    dd = 100 * (n / n.cummax() - 1)
    axes[1].fill_between(dd.index, dd, 0, color=plots.COLORS["NIFTY"], alpha=0.4, lw=0)
    axes[1].set_ylabel("NIFTY below its\nrunning peak, %")
    fig.suptitle(
        "March 2020 lifted almost every stock at once; later sell-offs touched far fewer",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "regime_timeline")

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(preds))
    v1 = [rho[k]["next vol"] for k in preds]
    v2 = [part.get(k, np.nan) for k in preds]
    ax.bar(
        xs - 0.2,
        v1,
        0.38,
        color=plots.SERIES[0],
        label="rank correlation with next month's NIFTY volatility",
    )
    ax.bar(
        xs + 0.2,
        [0 if np.isnan(x) else x for x in v2],
        0.38,
        color=plots.SERIES[2],
        label="the same, beyond NIFTY's own GJR forecast",
    )
    for i, (a, bb) in enumerate(zip(v1, v2, strict=True)):
        ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8, color=plots.TEXT)
        if not np.isnan(bb):
            ax.text(i + 0.2, bb + 0.01, f"{bb:.2f}", ha="center", fontsize=8, color=plots.TEXT)
    ax.set_xticks(xs, preds)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        f"{len(d)} month-ends, {d.index[0]:%Y}-{d.index[-1]:%Y}",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.suptitle(
        "India VIX warned best; stock stress breadth added something NIFTY's own model missed",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "regime_warning")
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "asymmetry": report_asymmetry,
    "distributions": report_distributions,
    "refit": report_refit,
    "har": report_har,
    "horizons": report_horizons,
    "sizing": report_sizing,
    "regime": report_regime,
}
