import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gv import models as m

JS = Path(__file__).resolve().parents[1] / "src" / "gv" / "garch.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(expr: str):
    # The script goes in on stdin: Linux caps a single command-line argument at 128 KB.
    script = f"const g = require({json.dumps(str(JS))});console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _returns(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    r, s2 = np.empty(n), 1.2
    for t in range(n):
        r[t] = 0.04 + math.sqrt(s2) * rng.standard_t(6) * math.sqrt(4 / 6)
        e = r[t] - 0.04
        s2 = 0.03 + (0.04 + 0.12 * (e < 0)) * e * e + 0.88 * s2
    return r


@pytest.fixture(scope="module")
def fits():
    r = _returns()
    return r, {(mod, d): m.fit(r, mod, d) for mod in m.MODELS for d in ("t", "skewt")}


def _p(p: dict) -> str:
    return json.dumps({k: v for k, v in p.items() if isinstance(v, int | float)})


@pytest.mark.parametrize("model", m.MODELS)
@pytest.mark.parametrize("dist", ["t", "skewt"])
def test_browser_recursion_forecast_and_news_impact_match_python(fits, model, dist):
    r, all_fits = fits
    p = all_fits[(model, dist)]
    rr, pp = json.dumps(r.tolist()), _p(p)
    shocks = np.linspace(-5, 5, 41)
    js = _node(
        f"(() => {{ const r = {rr}, p = {pp}, s = g.lastState(r, p, {json.dumps(model)});"
        f"return {{ bc: g.backcast(r), s2: g.conditionalVariance(r, p, {json.dumps(model)}),"
        f"f: g.forecastVariance(s, p, {json.dumps(model)}, 63),"
        f"ni: g.newsImpact(p, {json.dumps(model)}, {json.dumps(shocks.tolist())}),"
        f"pers: g.persistence(p, {json.dumps(model)}), hl: g.halfLife(p, {json.dumps(model)}),"
        f"lr: g.longRunVariance(p, {json.dumps(model)}),"
        f"var: g.valueAtRisk(p.mu, s.sigma2, p.q01) }}; }})()"
    )
    state = m.last_state(r, p, model)
    assert js["bc"] == pytest.approx(m.backcast(r), rel=1e-9)
    assert js["s2"] == pytest.approx(m.conditional_variance(r, p, model).tolist(), rel=1e-9)
    assert js["f"] == pytest.approx(m.forecast_variance(state, p, model, 63).tolist(), rel=1e-9)
    assert js["ni"] == pytest.approx(m.news_impact(p, model, shocks).tolist(), rel=1e-9)
    assert js["pers"] == pytest.approx(m.persistence(p, model), rel=1e-12)
    assert js["hl"] == pytest.approx(m.half_life(p, model), rel=1e-9)
    assert js["lr"] == pytest.approx(m.long_run_variance(p, model), rel=1e-9)
    assert js["var"] == pytest.approx(m.value_at_risk(p["mu"], state["sigma2"], p["q01"]), rel=1e-9)


def test_browser_rolling_std_matches_pandas():
    r = _returns(seed=3, n=200)
    js = _node(f"g.rollingStd({json.dumps(r.tolist())}, 20)")
    py = pd.Series(r).rolling(20).std()
    assert all(v is None for v in js[:19])
    assert js[19:] == pytest.approx(py.iloc[19:].tolist(), rel=1e-9)


def test_browser_rejects_unknown_models():
    out = _node(
        "(() => { try { g.persistence({}, 'APARCH'); return 'no'; } catch (e) { return 'ok'; } })()"
    )
    assert out == "ok"
