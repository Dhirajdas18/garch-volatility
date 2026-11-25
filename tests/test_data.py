import numpy as np
import pandas as pd
import pytest

from gv.data import (
    ETFS,
    INDICES,
    SECTORS,
    SYMBOLS,
    add_adjusted_ohlc,
    classify_drops,
    daily_returns,
    missing_symbols,
    tickers,
)


def test_universe_snapshot_is_nifty200_with_industries():
    assert len(SYMBOLS) >= 190 and len(set(SYMBOLS)) == len(SYMBOLS)
    assert set(SECTORS) == set(SYMBOLS) and all(SECTORS.values())
    assert not (set(INDICES) | set(ETFS)) & set(SYMBOLS)


def test_yahoo_tickers():
    t = tickers(["NIFTY", "NIFTYBEES", "M&M"])
    assert t == {"^NSEI": "NIFTY", "NIFTYBEES.NS": "NIFTYBEES", "M&M.NS": "M&M"}


def test_short_history_is_not_missing():
    idx = pd.bdate_range("2020-01-01", periods=4)
    adj = pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [np.nan, np.nan, 5, 6], "C": np.nan}, index=idx)
    table = pd.concat({"adj_close": adj}, axis=1)
    assert missing_symbols(table, ["A", "B", "C", "D"]) == ["C", "D"]


def test_adjusted_ohlc_scales_by_the_days_dividend_factor():
    idx = pd.bdate_range("2020-01-01", periods=2)
    fields = {
        "open": [100.0, 98.0],
        "high": [104.0, 101.0],
        "low": [99.0, 95.0],
        "close": [102.0, 100.0],
        "adj_close": [51.0, 80.0],  # factors 0.5 and 0.8
    }
    table = pd.concat({f: pd.DataFrame({"X": v}, index=idx) for f, v in fields.items()}, axis=1)
    out = add_adjusted_ohlc(table)
    assert out[("adj_open", "X")].tolist() == pytest.approx([50.0, 78.4])
    assert out[("adj_high", "X")].tolist() == pytest.approx([52.0, 80.8])
    assert out[("adj_low", "X")].tolist() == pytest.approx([49.5, 76.0])
    assert out.columns.names == ["field", "symbol"]


def _fixes(tmp_path, rows):
    p = tmp_path / "fixes.csv"
    pd.DataFrame(rows, columns=["symbol", "date", "action", "reason"]).to_csv(p, index=False)
    return p


def test_zero_blank_and_start_fixes(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame(
        {
            "A": [100.0, 50.0, 51.0, 52.0, 53.0],  # day 1: unadjusted split -> zero
            "B": [10.0, 1.0, 10.5, 10.6, 10.7],  # day 1: bad print -> blank
            "C": [99.0, 98.0, 5.0, 5.1, 5.2],  # real history starts on day 2
        },
        index=idx,
    )
    fixes = _fixes(
        tmp_path,
        [
            ["A", idx[1].date(), "zero", ""],
            ["B", idx[1].date(), "blank", ""],
            ["C", idx[2].date(), "start", ""],
        ],
    )
    r = daily_returns(px, fixes)
    assert r.loc[idx[1], "A"] == 0.0
    assert r.loc[idx[1], "B"] == 0.0 and r.loc[idx[2], "B"] == pytest.approx(0.05)
    assert r.loc[: idx[2], "C"].isna().all() and r.loc[idx[3], "C"] == pytest.approx(0.02)
    assert np.isfinite(r.loc[idx[3]]).all()


def test_classify_drops(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=4)
    adj = pd.DataFrame(
        {
            "A": [100.0, 50, 50, 50],  # fixed on day 1
            "B": [100.0, 100, 70, 70],  # day 2: the index fell 6%
            "C": [100.0, 100, 100, 75],  # day 3: alone
            "D": [100.0, 70, 70, 70],  # day 1, before its start row
        },
        index=idx,
    )
    index = pd.Series([100.0, 100, 94, 94], index=idx)
    fixes = _fixes(tmp_path, [["A", idx[1].date(), "zero", ""], ["D", idx[2].date(), "start", ""]])
    out = classify_drops(adj, index, fixes)
    assert sorted(out["fixed"]) == [("A", idx[1]), ("D", idx[1])]
    assert out["market"] == [("B", idx[2])]
    assert out["isolated"] == [("C", idx[3])]
