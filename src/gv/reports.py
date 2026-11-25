"""README findings, one function per topic: `gv report <topic>` writes charts to out/."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd


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


REPORTS: dict[str, Callable[[Path], dict]] = {}
