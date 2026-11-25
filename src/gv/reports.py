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


REPORTS: dict[str, Callable[[Path], dict]] = {
    "asymmetry": report_asymmetry,
    "distributions": report_distributions,
}
