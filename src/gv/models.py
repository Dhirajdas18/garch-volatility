"""Volatility models: GARCH(1,1), GJR-GARCH(1,1,1) and EGARCH(1,1,1) with a constant mean.

Estimation uses Kevin Sheppard's `arch` package; the recursions, forecasts and news-impact curves
are plain numpy so they can be checked line by line and mirrored in the browser (garch.js).

Returns are in percent (a 1.2% day is 1.2), which keeps the optimiser well scaled. With
e_t = r_t - mu the residual and z_t = e_t / sigma_t the standardised shock:

    GARCH   s2_t = omega + alpha e2_{t-1} + beta s2_{t-1}
    GJR     s2_t = omega + (alpha + gamma 1[e_{t-1} < 0]) e2_{t-1} + beta s2_{t-1}
    EGARCH  ln s2_t = omega + alpha (|z_{t-1}| - sqrt(2/pi)) + gamma z_{t-1} + beta ln s2_{t-1}

The constant sqrt(2/pi) in EGARCH is E|z| for a normal shock; arch uses it whatever the
distribution, and so do we.
"""

import math

import numpy as np
import pandas as pd

MODELS = ("GARCH", "GJR", "EGARCH")
DISTS = ("normal", "t", "skewt")
SQRT2_OV_PI = math.sqrt(2 / math.pi)
_ARCH_DIST = {"normal": "normal", "t": "t", "skewt": "skewt"}


def _check(model: str, dist: str | None = None) -> None:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {model!r}")
    if dist is not None and dist not in DISTS:
        raise ValueError(f"dist must be one of {DISTS}, not {dist!r}")


def backcast(returns_pct: np.ndarray) -> float:
    """The variance assumed before the first day, as arch computes it: an exponentially weighted
    (0.94) average of the first 75 squared deviations from the sample mean."""
    r = np.asarray(returns_pct, dtype=float)
    e = r - r.mean()
    tau = min(75, len(e))
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(e[:tau] ** 2 * w))


def dist_moments(dist: str, nu: float | None = None, lam: float | None = None) -> dict:
    """Moments of the unit-variance shock that the forecasts and VaR need:
    asym_weight = E[z^2 1(z<0)] (0.5 when symmetric), abs_mean = E|z|, q01 = the 1% quantile."""
    from arch.univariate import Normal, SkewStudent, StudentsT

    d, p = {
        "normal": (Normal(), []),
        "t": (StudentsT(), [nu]),
        "skewt": (SkewStudent(), [nu, lam]),
    }[dist]
    return {
        "asym_weight": float(d.partial_moment(2, 0.0, p)),
        "abs_mean": float(-2 * d.partial_moment(1, 0.0, p)),  # E z = 0, so E|z| = -2 E[z 1(z<0)]
        "q01": float(d.ppf(0.01, p)),
    }


def persistence(params: dict, model: str) -> float:
    """How much of today's variance shock is left tomorrow: alpha + gamma E[z^2 1(z<0)] + beta for
    GARCH and GJR (gamma is 0 for GARCH), beta for EGARCH (on log variance)."""
    _check(model)
    if model == "EGARCH":
        return params["beta"]
    return params["alpha"] + params.get("gamma", 0.0) * params["asym_weight"] + params["beta"]


def half_life(params: dict, model: str) -> float:
    """Days for a variance shock to halve: ln 0.5 / ln persistence (inf if persistence >= 1)."""
    p = persistence(params, model)
    return math.log(0.5) / math.log(p) if 0 < p < 1 else math.inf


def long_run_variance(params: dict, model: str) -> float:
    """The level forecasts revert to: omega / (1 - persistence) for GARCH and GJR (nan if not
    stationary); for EGARCH exp of the long-run mean of ln s2, consistent with forecast_variance."""
    p = persistence(params, model)
    if p >= 1:
        return math.nan
    if model == "EGARCH":
        drift = params["omega"] + params["alpha"] * (params["abs_mean"] - SQRT2_OV_PI)
        return math.exp(drift / (1 - p))
    return params["omega"] / (1 - p)


def fit(returns_pct, model: str = "GARCH", dist: str = "t") -> dict:
    """Fit a constant-mean GARCH, GJR or EGARCH (p=1, q=1, o=1 when asymmetric) by maximum
    likelihood with arch. Returns plain floats: mu, omega, alpha, gamma (0 for GARCH), beta,
    nu and lambda (None when the distribution lacks them), loglik, aic, bic, persistence,
    converged, n, and the shock moments of dist_moments."""
    from arch import arch_model

    _check(model, dist)
    y = np.asarray(returns_pct, dtype=float)
    am = arch_model(
        y,
        mean="Constant",
        vol="EGARCH" if model == "EGARCH" else "GARCH",
        p=1,
        o=0 if model == "GARCH" else 1,
        q=1,
        dist=_ARCH_DIST[dist],
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 1000})
    p = res.params
    out = {
        "model": model,
        "dist": dist,
        "mu": float(p["mu"]),
        "omega": float(p["omega"]),
        "alpha": float(p["alpha[1]"]),
        "gamma": float(p["gamma[1]"]) if "gamma[1]" in p else 0.0,
        "beta": float(p["beta[1]"]),
        "nu": float(p["eta"] if "eta" in p else p["nu"]) if dist != "normal" else None,
        "lambda": float(p["lambda"]) if "lambda" in p else None,
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "converged": bool(res.convergence_flag == 0),
        "n": len(y),
    }
    out.update(dist_moments(dist, out["nu"], out["lambda"]))
    out["persistence"] = persistence(out, model)
    return out


def conditional_variance(returns_pct, params: dict, model: str) -> np.ndarray:
    """s2_t for every day t, from information up to t-1 (percent squared). Starts from
    backcast(), as arch does; arch's variance bounds are not applied (they bind only for
    degenerate parameters)."""
    _check(model)
    r = np.asarray(returns_pct, dtype=float)
    e = r - params["mu"]
    n = len(e)
    s2 = np.empty(n)
    bc = backcast(r)
    om, a, g, b = params["omega"], params["alpha"], params.get("gamma", 0.0), params["beta"]
    if model == "EGARCH":
        lns = om + b * math.log(bc)  # the shock terms are left out on the first day, as in arch
        for t in range(n):
            if t:
                z = e[t - 1] / math.sqrt(s2[t - 1])
                lns = om + a * (abs(z) - SQRT2_OV_PI) + g * z + b * lns
            s2[t] = math.exp(lns)
        return s2
    s2[0] = om + a * bc + g * 0.5 * bc + b * bc
    for t in range(1, n):
        e2 = e[t - 1] ** 2
        s2[t] = om + (a + (g if e[t - 1] < 0 else 0.0)) * e2 + b * s2[t - 1]
    return s2


def last_state(returns_pct, params: dict, model: str) -> dict:
    """The last day's residual and conditional variance, which is all a forecast needs."""
    r = np.asarray(returns_pct, dtype=float)
    s2 = conditional_variance(r, params, model)
    return {"resid": float(r[-1] - params["mu"]), "sigma2": float(s2[-1])}


def forecast_variance(state: dict, params: dict, model: str, horizon: int) -> np.ndarray:
    """Variance forecasts for each of the next `horizon` days (day 1 is tomorrow).

    GARCH and GJR are exact: after day 1, s2_{h} = omega + persistence * s2_{h-1}, where GJR's
    asymmetry enters through E[z^2 1(z<0)] of the fitted distribution (0.5 if symmetric; the
    skewed-t value otherwise, as arch's analytic forecast uses).

    EGARCH has no closed form for the expected variance beyond day 1 (with Student-t shocks it
    need not even be finite). This returns exp(E[ln s2_h]), with
    E[ln s2_h] = omega + alpha (E|z| - sqrt(2/pi)) + beta E[ln s2_{h-1}]: the geometric-mean path,
    which sits below the mean variance by a Jensen gap that grows with the horizon.
    """
    _check(model)
    e, s2 = state["resid"], state["sigma2"]
    om, a, g, b = params["omega"], params["alpha"], params.get("gamma", 0.0), params["beta"]
    out = np.empty(horizon)
    if model == "EGARCH":
        z = e / math.sqrt(s2)
        lns = om + a * (abs(z) - SQRT2_OV_PI) + g * z + b * math.log(s2)
        drift = om + a * (params["abs_mean"] - SQRT2_OV_PI)
        for h in range(horizon):
            if h:
                lns = drift + b * lns
            out[h] = math.exp(lns)
        return out
    out[0] = om + (a + (g if e < 0 else 0.0)) * e * e + b * s2
    p = persistence(params, model)
    for h in range(1, horizon):
        out[h] = om + p * out[h - 1]
    return out


def news_impact(params: dict, model: str, shocks) -> np.ndarray:
    """Tomorrow's variance as a function of today's residual (percent), with today's variance at
    its long-run level."""
    _check(model)
    e = np.asarray(shocks, dtype=float)
    lr = long_run_variance(params, model)
    om, a, g, b = params["omega"], params["alpha"], params.get("gamma", 0.0), params["beta"]
    if model == "EGARCH":
        z = e / math.sqrt(lr)
        return np.exp(om + a * (np.abs(z) - SQRT2_OV_PI) + g * z + b * math.log(lr))
    return om + (a + g * (e < 0)) * e**2 + b * lr


def value_at_risk(mu: float, sigma2: float, q01: float, amount: float = 1e5) -> float:
    """One-day 99% VaR in rupees on `amount`, for log returns in percent: the loss at the 1%
    quantile mu + sigma q01 of the model's distribution."""
    r = (mu + math.sqrt(sigma2) * q01) / 100
    return amount * (1 - math.exp(r))


def parkinson(high, low):
    """Parkinson's daily variance from the high-low range: (ln H/L)^2 / (4 ln 2)."""
    return np.log(np.asarray(high, float) / np.asarray(low, float)) ** 2 / (4 * math.log(2))


def garman_klass(open_, high, low, close):
    """Garman-Klass daily variance: 0.5 (ln H/L)^2 - (2 ln 2 - 1) (ln C/O)^2 (no overnight gap)."""
    hl = np.log(np.asarray(high, float) / np.asarray(low, float))
    co = np.log(np.asarray(close, float) / np.asarray(open_, float))
    return 0.5 * hl**2 - (2 * math.log(2) - 1) * co**2


def har_features(daily_var: pd.Series) -> pd.DataFrame:
    """HAR regressors for forecasting day t+1: the day's variance, and its means over the last 5
    and 22 days, each ending at day t (so row t uses data up to t only)."""
    v = pd.Series(daily_var, dtype=float)
    return pd.DataFrame(
        {
            "daily": v,
            "weekly": v.rolling(5).mean(),
            "monthly": v.rolling(22).mean(),
        }
    )
