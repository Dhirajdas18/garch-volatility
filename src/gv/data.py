"""Daily OHLC and adjusted closes from Yahoo Finance for today's NIFTY 200 stocks, NIFTY and the
NIFTY ETF.

The universe (docs/data/nifty200.csv) is a snapshot of NSE's NIFTY 200 list with each stock's
industry. Using today's list over past years is survivorship-biased: stocks that fell out of the
index are missing. The findings use a point-in-time universe instead (see universe.py).

Columns are a (field, symbol) MultiIndex. Fields: open, high, low, close, adj_close, and
adj_open, adj_high, adj_low. Yahoo's open/high/low/close are split-adjusted only; adj_close also
takes out dividends. The adjusted open/high/low scale each day's open/high/low by that day's
adj_close / close, so a day's range and its close are on the same basis.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONSTITUENTS = ROOT / "docs" / "data" / "nifty200.csv"
_UNIVERSE = pd.read_csv(CONSTITUENTS)
SYMBOLS = _UNIVERSE["symbol"].tolist()
SECTORS = dict(_UNIVERSE.set_index("symbol")["industry"])
INDEX = "NIFTY"
INDICES = {"NIFTY": "^NSEI"}  # price index, no dividends
ETFS = ["NIFTYBEES"]  # NIFTY 50 ETF, with dividends
START = "2007-01-01"  # includes 2008
DATA_PATH = ROOT / "data" / "prices.parquet"
FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close"}
FIELDS_OHLC = ("open", "high", "low", "close")


def everything() -> list[str]:
    """Every symbol we download: the stocks, the index and the ETF."""
    return SYMBOLS + list(INDICES) + ETFS


def tickers(symbols: list[str] | None = None) -> dict[str, str]:
    """Yahoo ticker -> our symbol, for `symbols` (default: everything)."""
    want = symbols if symbols is not None else everything()
    return {INDICES.get(s, f"{s}.NS"): s for s in want}


def add_adjusted_ohlc(table: pd.DataFrame) -> pd.DataFrame:
    """Add adj_open, adj_high and adj_low: open/high/low times the day's adj_close / close."""
    factor = table["adj_close"] / table["close"]
    adj = {f"adj_{f}": table[f] * factor for f in ("open", "high", "low")}
    out = pd.concat([table, pd.concat(adj, axis=1, names=["field"])], axis=1)
    out.columns.names = ["field", "symbol"]
    return out.sort_index(axis=1)


def fetch_yfinance(
    symbols: list[str] | None = None, start: str = START, end: str | None = None
) -> pd.DataFrame:
    """Download `symbols` (default: all), with the adjusted open/high/low added."""
    import yfinance as yf

    names = tickers(symbols)
    # threads=False: parallel downloads can lock yfinance's sqlite cache and drop a ticker
    raw = yf.download(
        list(names), start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    raw = raw[list(FIELDS)].rename(columns=FIELDS, level=0).rename(columns=names, level=1)
    raw.columns.names = ["field", "symbol"]
    raw.index = pd.DatetimeIndex(raw.index.tz_localize(None), name="date")
    return add_adjusted_ohlc(raw)


def missing_symbols(table: pd.DataFrame, expected: list[str] | None = None) -> list[str]:
    """Symbols with no adjusted close at all: an all-empty column or one Yahoo left out.

    Short history (a recent listing) is not missing. `expected` defaults to every symbol we
    download (the stocks, the index and the ETF).
    """
    adj = table["adj_close"] if "adj_close" in table.columns.get_level_values(0) else pd.DataFrame()
    want = expected if expected is not None else everything()
    return [s for s in want if s not in adj.columns or adj[s].isna().all()]


def load(field: str, symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """One field as a wide date x symbol frame (all rows; gaps stay NaN)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    table = pd.read_parquet(path)
    if field not in table.columns.get_level_values("field"):
        raise KeyError(f"field not in price table: {field}")
    df = table[field]
    if symbols is not None:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise KeyError(f"symbols not in price table: {missing}")
        df = df[symbols]
    df.columns.name = None
    return df.sort_index()


def load_bars(symbol: str, path: Path = DATA_PATH) -> pd.DataFrame:
    """Adjusted open, high, low and close for one symbol, on days with all four."""
    fields = {f: load(f"adj_{f}", [symbol], path=path)[symbol] for f in FIELDS_OHLC}
    return pd.concat(fields, axis=1).dropna()


def flag_adjusted_drops(
    adj: pd.DataFrame, threshold: float = 0.2
) -> list[tuple[str, pd.Timestamp]]:
    """(symbol, date) pairs where an adjusted price falls more than `threshold` in a day.

    Large caps rarely do that for real; each hit is worth checking for an unadjusted split or bonus.
    """
    moves = adj.pct_change(fill_method=None).stack()
    hits = moves[moves < -threshold]
    return [(sym, date) for date, sym in hits.index]


FIXES = ROOT / "docs" / "data" / "yahoo_fixes.csv"


def classify_drops(
    adj: pd.DataFrame, index: pd.Series, fixes: Path = FIXES, market_move: float = -0.03
) -> dict[str, list[tuple[str, pd.Timestamp]]]:
    """Sort flag_adjusted_drops into three groups:
    - fixed: the fixes file repairs that day, or a `start` row drops it from the stock's history;
    - market: the index fell by `market_move` or more that day, or three or more stocks were
      flagged on it (a market-wide fall, not a data error);
    - isolated: everything else, to review against the exchange's own prices.
    """
    table = pd.read_csv(fixes, parse_dates=["date"])
    days = set(zip(table["symbol"], table["date"], strict=True))
    starts = dict(zip(*table[table["action"] == "start"][["symbol", "date"]].T.values, strict=True))
    flags = flag_adjusted_drops(adj)
    per_day = pd.Series([d for _, d in flags]).value_counts()
    index_ret = index.pct_change(fill_method=None)
    out = {"fixed": [], "market": [], "isolated": []}
    for sym, day in flags:
        if (sym, day) in days or (sym in starts and day < starts[sym]):
            out["fixed"].append((sym, day))
        elif index_ret.get(day, 0.0) <= market_move or per_day[day] >= 3:
            out["market"].append((sym, day))
        else:
            out["isolated"].append((sym, day))
    return out


def apply_fixes(prices: pd.DataFrame, fixes: Path = FIXES) -> pd.DataFrame:
    """Adjusted closes with `start` and `blank` rows of the fixes file applied, gaps inside a
    stock's own history carried forward (`zero` rows act on returns; see daily_returns)."""
    px = prices.copy()
    table = pd.read_csv(fixes, parse_dates=["date"])
    for row in table[table["action"] == "start"].itertuples():
        if row.symbol in px.columns:
            px.loc[px.index < row.date, row.symbol] = float("nan")
    for row in table[table["action"] == "blank"].itertuples():
        if row.symbol in px.columns and row.date in px.index:
            px.loc[row.date, row.symbol] = float("nan")
    started = px.notna().cummax()  # forward-fill only inside a stock's own history
    return px.ffill().where(started)


def daily_returns(prices: pd.DataFrame, fixes: Path = FIXES) -> pd.DataFrame:
    """Simple daily returns of adjusted closes, with known Yahoo errors repaired from
    docs/data/yahoo_fixes.csv (each with its source):
    - zero: a corporate action Yahoo did not adjust for; that day's return is set to 0.
    - blank: a bad print; the price is dropped and the previous price carried forward.
    - start: the series is not this company before the date; earlier prices are dropped.
    """
    r = apply_fixes(prices, fixes).pct_change(fill_method=None)
    table = pd.read_csv(fixes, parse_dates=["date"])
    for row in table[table["action"] == "zero"].itertuples():
        if row.symbol in r.columns and row.date in r.index:
            r.loc[row.date, row.symbol] = 0.0
    return r
