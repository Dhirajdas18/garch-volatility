import numpy as np
import pandas as pd

from gv.sizing import inverse_vol, vol_target


def test_vol_target_uses_the_month_end_exposure_from_the_next_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.Series(0.01, index=idx)
    me = pd.DatetimeIndex([pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")])
    fc = pd.Series([0.24, 0.06], index=me)  # exposure 0.5, then 1.5 (cap)
    out = vol_target(r, fc, target=0.12, cap=1.5, rf=0.0, cost=0.0)
    assert (
        pd.Timestamp("2020-01-31") not in out.index
    )  # decided at that close, held from the next day
    assert out.loc["2020-02-03", "exposure"] == 0.5
    assert out.loc["2020-03-02", "exposure"] == 1.5


def test_inverse_vol_weights_and_next_month_dating():
    me = pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"])
    rets = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=me)
    fc = pd.DataFrame({"A": [1.0, 1.0], "B": [4.0, 4.0]}, index=me[:2])  # sd 1 and 2
    out = inverse_vol(fc, rets, cost=0.0)
    assert out["net"].index[0] == me[1]
    assert np.isclose(out["net"].iloc[0], (2 / 3) * 0.10)
