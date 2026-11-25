import math

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from gv import models as m
from gv.data import DATA_PATH


def _simulate(n=2500, seed=0, model="GJR"):
    """Percent returns from a GJR-GARCH (or EGARCH) with unit-variance Student-t shocks."""
    rng = np.random.default_rng(seed)
    nu = 6.0
    z = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    r, s2 = np.empty(n), 1.5
    for t in range(n):
        r[t] = 0.05 + math.sqrt(s2) * z[t]
        e = r[t] - 0.05
        if model == "EGARCH":
            zt = e / math.sqrt(s2)
            ln = -0.01 + 0.15 * (abs(zt) - m.SQRT2_OV_PI) - 0.08 * zt + 0.96 * math.log(s2)
            s2 = math.exp(ln)
        else:
            s2 = 0.04 + (0.04 + 0.1 * (e < 0)) * e * e + 0.9 * s2
    return r


def _arch(y, model, dist):
    return arch_model(
        y,
        mean="Constant",
        vol="EGARCH" if model == "EGARCH" else "GARCH",
        p=1,
        o=0 if model == "GARCH" else 1,
        q=1,
        dist=dist,
        rescale=False,
    ).fit(disp="off", show_warning=False, options={"maxiter": 1000})


def _real_returns():
    from gv.data import daily_returns

    px = pd.read_parquet(DATA_PATH)["adj_close"][["RELIANCE"]]
    r = np.log1p(daily_returns(px)["RELIANCE"]).dropna()
    return 100 * r.iloc[-1250:].to_numpy()


@pytest.mark.parametrize("model", m.MODELS)
@pytest.mark.parametrize("dist", m.DISTS)
def test_recursion_matches_arch_on_simulated_data(model, dist):
    y = _simulate(model="EGARCH" if model == "EGARCH" else "GJR")
    p = m.fit(y, model, dist)
    res = _arch(y, model, dist)
    s = np.sqrt(m.conditional_variance(y, p, model))
    assert s[20:] == pytest.approx(res.conditional_volatility[20:], rel=1e-6)
    assert p["converged"] and p["loglik"] == pytest.approx(res.loglikelihood)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="needs data/prices.parquet")
@pytest.mark.parametrize("model", m.MODELS)
def test_recursion_matches_arch_on_a_real_stock(model):
    y = _real_returns()
    p = m.fit(y, model, "t")
    res = _arch(y, model, "t")
    s = np.sqrt(m.conditional_variance(y, p, model))
    assert s[20:] == pytest.approx(res.conditional_volatility[20:], rel=1e-6)


@pytest.mark.parametrize(
    "model,dist", [("GARCH", "normal"), ("GARCH", "t"), ("GJR", "t"), ("GJR", "skewt")]
)
def test_forecasts_match_arch_analytic(model, dist):
    y = _simulate(seed=1)
    p = m.fit(y, model, dist)
    want = _arch(y, model, dist).forecast(horizon=63, reindex=False).variance.to_numpy()[-1]
    got = m.forecast_variance(m.last_state(y, p, model), p, model, 63)
    assert got == pytest.approx(want, rel=1e-9)
    far = m.forecast_variance(m.last_state(y, p, model), p, model, 5000)[-1]
    assert far == pytest.approx(m.long_run_variance(p, model), rel=1e-6)


def test_skewed_t_asymmetry_weight_is_not_a_half():
    w = m.dist_moments("skewt", 6.0, -0.2)["asym_weight"]
    assert w > 0.5  # left skew puts more variance below zero
    assert m.dist_moments("t", 6.0)["asym_weight"] == pytest.approx(0.5)
    assert m.dist_moments("normal")["abs_mean"] == pytest.approx(m.SQRT2_OV_PI)


def test_egarch_forecast_day_one_exact_and_log_mean_path_after():
    y = _simulate(seed=2, model="EGARCH")
    p = m.fit(y, "EGARCH", "t")
    res = _arch(y, "EGARCH", "t")
    state = m.last_state(y, p, "EGARCH")
    got = m.forecast_variance(state, p, "EGARCH", 21)
    one = res.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0]
    assert got[0] == pytest.approx(one, rel=1e-9)
    sim = res.forecast(
        horizon=21,
        method="simulation",
        simulations=20000,
        reindex=False,
        random_state=np.random.RandomState(0),
    )
    paths = sim.simulations.residual_variances[-1]  # simulations x horizon
    mean_log = np.log(paths).mean(axis=0)
    assert np.log(got) == pytest.approx(mean_log, abs=0.02)


def test_news_impact_is_asymmetric_and_matches_a_one_day_forecast():
    shocks = np.array([-3.0, 0.0, 3.0])
    g = dict(omega=0.05, alpha=0.04, gamma=0.1, beta=0.88, asym_weight=0.5, abs_mean=0.78)
    ni = m.news_impact(g, "GJR", shocks)
    assert ni[0] > ni[2] > ni[1]
    lr = m.long_run_variance(g, "GJR")
    assert ni[1] == pytest.approx(0.05 + 0.88 * lr)
    one = m.forecast_variance({"resid": -3.0, "sigma2": lr}, g, "GJR", 1)[0]
    assert ni[0] == pytest.approx(one)
    sym = m.news_impact(dict(g, gamma=0.0), "GARCH", shocks)
    assert sym[0] == pytest.approx(sym[2])
    e = dict(omega=-0.01, alpha=0.15, gamma=-0.08, beta=0.96, abs_mean=0.78)
    ne = m.news_impact(e, "EGARCH", shocks)
    assert ne[0] > ne[2] > ne[1]
    one = m.forecast_variance(
        {"resid": 3.0, "sigma2": m.long_run_variance(e, "EGARCH")}, e, "EGARCH", 1
    )[0]
    assert ne[2] == pytest.approx(one)


def test_persistence_half_life_and_long_run():
    g = dict(omega=0.05, alpha=0.05, gamma=0.1, beta=0.85, asym_weight=0.5)
    assert m.persistence(g, "GJR") == pytest.approx(0.95)
    assert m.half_life(g, "GJR") == pytest.approx(math.log(0.5) / math.log(0.95))
    assert m.long_run_variance(g, "GJR") == pytest.approx(1.0)
    assert math.isnan(m.long_run_variance(dict(g, beta=0.95), "GJR"))
    with pytest.raises(ValueError):
        m.persistence(g, "APARCH")


def test_value_at_risk_on_a_lakh():
    var = m.value_at_risk(0.0, 1.0, -2.326, 1e5)  # 1% daily sigma
    assert var == pytest.approx(1e5 * (1 - math.exp(-0.02326)))


def test_range_estimators():
    h, lo = math.exp(0.02), 1.0
    assert m.parkinson([h], [lo])[0] == pytest.approx(0.0004 / (4 * math.log(2)))
    assert m.garman_klass([1.0], [h], [lo], [1.0])[0] == pytest.approx(0.5 * 0.0004)
    # Brownian paths with 1% daily volatility: both average close to 1e-4 (a path sampled 1000
    # times a day slightly understates the true range, so the tolerance is 10%)
    rng = np.random.default_rng(0)
    paths = np.cumsum(rng.normal(0, 0.01 / math.sqrt(1000), (3000, 1000)), axis=1)
    paths = np.hstack([np.zeros((3000, 1)), paths])
    o, c = np.exp(paths[:, 0]), np.exp(paths[:, -1])
    hi, low = np.exp(paths.max(axis=1)), np.exp(paths.min(axis=1))
    assert m.parkinson(hi, low).mean() == pytest.approx(1e-4, rel=0.1)
    assert m.garman_klass(o, hi, low, c).mean() == pytest.approx(1e-4, rel=0.1)


def test_har_features_use_data_up_to_the_day_only():
    v = pd.Series(np.arange(1.0, 41.0))
    f = m.har_features(v)
    assert list(f.columns) == ["daily", "weekly", "monthly"]
    assert f["weekly"].iloc[:4].isna().all() and f["monthly"].iloc[:21].isna().all()
    assert f["weekly"].iloc[10] == pytest.approx(np.mean(np.arange(7.0, 12.0)))
    assert f["monthly"].iloc[30] == pytest.approx(np.mean(np.arange(10.0, 32.0)))
    future = v.copy()
    future.iloc[25:] = 1e6
    g = m.har_features(future)
    pd.testing.assert_frame_equal(f.iloc[:25], g.iloc[:25])


def test_predictive_loglik_matches_arch_in_sample_for_t():
    import numpy as np

    from gv.models import fit, predictive_loglik

    rng = np.random.default_rng(5)
    y = rng.standard_t(6, 1500) * 1.2
    p = fit(y, "GJR", "t")
    ll = predictive_loglik(y, p, "GJR", 0).sum()
    assert np.isclose(ll, p["loglik"], rtol=1e-6)
    assert p["gamma_t"] is not None


def test_predictive_density_matches_arch_loglik_for_every_distribution():
    import numpy as np

    from gv.models import fit, predictive_density

    rng = np.random.default_rng(9)
    y = rng.standard_t(5, 1500)
    for dist in ("normal", "t", "skewt"):
        p = fit(y, "GJR", dist)
        ll, var, r = predictive_density(y, p, "GJR", 0)
        assert np.isclose(ll.sum(), p["loglik"], rtol=1e-6)
        assert 0.003 < np.mean(r < var) < 0.03


def test_har_fit_recovers_a_known_log_har():
    import numpy as np
    import pandas as pd

    from gv.har import har_fit, har_forecast, qlike

    rng = np.random.default_rng(3)
    n = 3000
    lv = np.zeros(n)
    for t in range(22, n):
        lv[t] = (
            0.1
            + 0.3 * lv[t - 1]
            + 0.3 * lv[t - 5 : t].mean()
            + 0.3 * lv[t - 22 : t].mean()
            + rng.normal(0, 0.3)
        )
    rv = pd.Series(np.exp(lv))
    p = har_fit(rv)
    assert p["b"][1] > 0 and abs(p["b"][1:].sum() - 0.9) < 0.1
    f = har_forecast(rv, p)
    assert np.isnan(f.iloc[0]) and qlike(np.array([1.0]), np.array([1.0]))[0] == 0


def test_tvalue_returns_none_when_the_hessian_is_singular():
    import numpy as np
    import pandas as pd

    from gv.models import _tvalue

    class Singular:
        params = pd.Series({"gamma[1]": 0.0})

        @property
        def tvalues(self):
            raise np.linalg.LinAlgError("Singular matrix")

    assert _tvalue(Singular(), "gamma[1]") is None
    assert _tvalue(Singular(), "missing") is None


def test_sane_rejects_degenerate_fits_and_common_params_target_the_variance():
    import numpy as np

    from gv.models import common_params, long_run_variance, sane

    rng = np.random.default_rng(4)
    y = rng.standard_t(6, 1500)
    good = {
        "model": "GJR",
        "dist": "t",
        "mu": 0.05,
        "omega": 0.05,
        "alpha": 0.05,
        "gamma": 0.05,
        "beta": 0.88,
        "nu": 6.0,
        "lambda": None,
        "converged": True,
        "asym_weight": 0.5,
        "abs_mean": 0.75,
    }
    from gv.models import persistence

    good["persistence"] = persistence(good, "GJR")
    assert sane(good, "GJR", float(y.var()))
    bad = dict(good, mu=-34.0)
    assert not sane(bad, "GJR", float(y.var()))
    egarch = {
        "model": "EGARCH",
        "dist": "t",
        "mu": 0.0,
        "omega": 50.0,
        "alpha": -25.5,
        "gamma": 110.6,
        "beta": 0.0,
        "nu": 2.05,
        "lambda": None,
        "converged": True,
        "asym_weight": 0.5,
        "abs_mean": 0.9,
        "persistence": 0.0,
    }
    assert not sane(egarch, "EGARCH", float(y.var()))
    for model in ("GARCH", "GJR", "EGARCH"):
        c = common_params([good, dict(good, alpha=0.07)], model, y)
        assert np.isclose(long_run_variance(c, model), y.var(), rtol=1e-9)
