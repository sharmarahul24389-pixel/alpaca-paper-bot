"""
Shared Alpaca market data client.
Replaces yfinance for all intraday and daily bar fetches.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

logger = logging.getLogger(__name__)
_ET    = pytz.timezone("America/New_York")

_client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)

_TF = {
    "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "30m": TimeFrame(30, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
    "1d":  TimeFrame(1,  TimeFrameUnit.Day),
}


def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(_ET)
    else:
        df.index = df.index.tz_convert(_ET)
    return df


def _process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _to_et(df)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"}, inplace=True)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna()


def get_bars(ticker: str, timeframe: str = "1d", days: int = 60) -> pd.DataFrame | None:
    """
    Fetch bars for a single ticker. Returns DataFrame with ET-timezone index
    and columns Open/High/Low/Close/Volume. Returns None on failure.
    """
    cal_days = max(days * 2 + 10, 30)   # buffer for weekends / holidays
    start    = datetime.now(pytz.utc) - timedelta(days=cal_days)
    try:
        req  = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=_TF[timeframe],
            start=start,
            adjustment="all",
            feed="iex",
        )
        raw = _client.get_stock_bars(req).df
        if raw.empty:
            return None
        syms = raw.index.get_level_values(0).unique()
        if ticker not in syms:
            return None
        df = raw.xs(ticker, level=0)
        df = _process(df)
        if df.empty:
            return None
        # keep only the requested trading-day window
        cutoff = pd.Timestamp.now(tz=_ET) - pd.Timedelta(days=days + 5)
        df = df[df.index >= cutoff]
        return df if not df.empty else None
    except Exception as exc:
        logger.warning(f"Alpaca bars failed ({ticker} {timeframe} {days}d): {exc}")
        return None


def get_bars_multi(tickers: list[str], timeframe: str = "1d",
                   days: int = 60) -> dict[str, pd.DataFrame]:
    """
    Fetch bars for multiple tickers in one request.
    Returns {ticker: DataFrame}. Missing tickers are omitted.
    """
    if not tickers:
        return {}
    cal_days = max(days * 2 + 10, 30)
    start    = datetime.now(pytz.utc) - timedelta(days=cal_days)
    try:
        req  = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=_TF[timeframe],
            start=start,
            adjustment="all",
            feed="iex",
        )
        raw = _client.get_stock_bars(req).df
        if raw.empty:
            return {}
        result: dict[str, pd.DataFrame] = {}
        for sym in raw.index.get_level_values(0).unique():
            try:
                df = raw.xs(sym, level=0)
                df = _process(df)
                if not df.empty:
                    cutoff = pd.Timestamp.now(tz=_ET) - pd.Timedelta(days=days + 5)
                    df = df[df.index >= cutoff]
                    if not df.empty:
                        result[str(sym)] = df
            except Exception:
                pass
        return result
    except Exception as exc:
        logger.warning(f"Alpaca multi-bars failed ({timeframe} {days}d): {exc}")
        return {}
