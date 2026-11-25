import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from gv.data import SYMBOLS
from gv.site import SCALE, build_html, lab_data

FEW = SYMBOLS[:4]


@pytest.fixture(scope="module")
def data(monkeypatch_module):
    return lab_data(_prices())


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.setattr("gv.data.SYMBOLS", FEW)  # four stocks keep the fits quick
    yield mp
    mp.undo()


def _prices():
    idx = pd.bdate_range("2019-01-01", "2026-11-25")
    cols = FEW + ["NIFTY", "NIFTYBEES"]
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0003, 0.015, (len(idx), len(cols)))
    px = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)
    px.iloc[: len(idx) - 900, 0] = np.nan  # listed about 3.5 years ago: starts late
    px.iloc[: len(idx) - 300, 1] = np.nan  # listed about a year ago: too short, dropped
    return px


def test_lab_data_is_compact_and_fits_every_model(data):
    assert data["symbols"][0] == "NIFTY" and FEW[1] in data["dropped"]
    assert set(data["symbols"]) == {"NIFTY", FEW[0], FEW[2], FEW[3]}
    first = data["series"][FEW[0]]
    assert first["start"] > 0 and len(first["r"]) == len(data["dates"]) - first["start"]
    assert all(isinstance(v, int) for v in first["r"])
    assert 240 < data["per_year"] < 270
    for s in data["symbols"]:
        assert set(data["series"][s]["fit"]) == {"GARCH", "GJR", "EGARCH"}
        assert data["series"][s]["fit"]["GJR"]["nu"] > 2
    assert json.loads(json.dumps(data)) == data
    assert data["dates"][-1] == "2026-11-25"


def test_returns_round_trip_to_a_hundredth_of_a_percent(data):
    r = np.log(_prices()["NIFTY"]).diff()
    got = np.array(data["series"]["NIFTY"]["r"]) / SCALE
    assert got == pytest.approx(r.iloc[-len(got) :].to_numpy(), abs=0.5 / SCALE + 1e-12)


def test_page_builds_without_placeholders_and_states_the_caveats(data):
    html = build_html(_prices(), data)
    assert "function conditionalVariance" in html and 'id="garch-lab"' in html
    assert "$" + "js" not in html and "$" + "data" not in html
    for words in ("not investment advice", "in-sample", "today’s NIFTY 200", "dividends"):
        assert words in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_scripts_parse(data):
    html = build_html(_prices(), data)
    for i, src in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        out = subprocess.run(["node", "--check", "-"], input=src, capture_output=True, text=True)
        assert out.returncode == 0, (i, out.stderr[:500])
