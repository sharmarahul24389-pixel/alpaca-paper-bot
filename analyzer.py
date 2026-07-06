import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from alpaca_data import get_bars

logger = logging.getLogger(__name__)
_ET    = pytz.timezone("America/New_York")



def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP that resets at the start of each trading day."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    out = pd.Series(index=df.index, dtype=float)
    idx = df.index.tz_convert("America/New_York") if df.index.tzinfo else df.index
    for d in pd.unique(idx.date):
        m      = idx.date == d
        cum_tp = (tp[m] * df["Volume"][m]).cumsum()
        cum_v  = df["Volume"][m].cumsum()
        out[m] = (cum_tp / cum_v.replace(0, float("nan"))).values
    return out


def _vwap_bands(df: pd.DataFrame) -> dict:
    """Today's VWAP with +-1sigma and +-2sigma standard deviation bands."""
    today = datetime.now(_ET).date()
    idx   = df.index.tz_convert(_ET) if df.index.tzinfo else df.index
    tod   = df[idx.date == today].copy()

    if len(tod) < 2:
        return {}

    tp      = (tod["High"] + tod["Low"] + tod["Close"]) / 3
    cum_tpv = (tp * tod["Volume"]).cumsum()
    cum_vol = tod["Volume"].cumsum().replace(0, np.nan)
    vwap    = cum_tpv / cum_vol
    dev     = tp - vwap
    cum_var = (dev ** 2 * tod["Volume"]).cumsum() / cum_vol
    std     = np.sqrt(cum_var)

    vwap_now = float(vwap.iloc[-1])
    std_now  = float(std.iloc[-1])

    if np.isnan(vwap_now) or np.isnan(std_now) or std_now < 0.01:
        return {}

    return {
        "vwap":   round(vwap_now, 2),
        "upper1": round(vwap_now + std_now,     2),
        "upper2": round(vwap_now + 2 * std_now, 2),
        "lower1": round(vwap_now - std_now,     2),
        "lower2": round(vwap_now - 2 * std_now, 2),
        "std":    round(std_now, 2),
    }


def _orb_from_df(df: pd.DataFrame) -> dict | None:
    """Extract the Opening Range (9:30 AM 30-min bar) from the DataFrame."""
    today = datetime.now(_ET).date()
    idx   = df.index.tz_convert(_ET) if df.index.tzinfo else df.index

    mask = (idx.date == today) & (idx.hour == 9) & (idx.minute == 30)
    bar  = df[mask]
    if bar.empty:
        return None

    h = float(bar["High"].iloc[0])
    l = float(bar["Low"].iloc[0])
    return {
        "high":  round(h, 2),
        "low":   round(l, 2),
        "range": round(h - l, 2),
        "mid":   round((h + l) / 2, 2),
    }


def _pdh_pdl_from_df(df: pd.DataFrame) -> dict:
    """Previous trading day High / Low / Close from 30-min bars."""
    today = datetime.now(_ET).date()
    idx   = df.index.tz_convert(_ET) if df.index.tzinfo else df.index

    prev_day = None
    for d in sorted(set(idx.date), reverse=True):
        if d < today:
            prev_day = d
            break

    if prev_day is None:
        return {}

    prev = df[idx.date == prev_day]
    if prev.empty:
        return {}

    return {
        "pdh": round(float(prev["High"].max()), 2),
        "pdl": round(float(prev["Low"].min()), 2),
        "pdc": round(float(prev["Close"].iloc[-1]), 2),
    }


def fetch_30min(ticker: str, days: int = 5) -> pd.DataFrame | None:
    return get_bars(ticker, "30m", days=days + 4)   # +4 for weekend buffer


def fetch_5min(ticker: str) -> pd.DataFrame | None:
    return get_bars(ticker, "5m", days=3)


_orb15_cache: dict = {}   # ticker -> {date, high, low, range, mid}


def _orb_15min(ticker: str, df_5m: pd.DataFrame) -> dict | None:
    """
    15-min ORB: high/low of the first 3 x 5-min bars (9:30, 9:35, 9:40 AM ET).
    Fires at 9:45 AM — 15 minutes earlier than the classic 30-min ORB.
    """
    now_et = datetime.now(_ET)
    today  = now_et.date()

    # Not enough time has passed for all 3 bars to close
    if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 44):
        return None

    cached = _orb15_cache.get(ticker)
    if cached and cached.get("date") == today:
        return cached

    idx = df_5m.index.tz_convert(_ET) if df_5m.index.tzinfo else df_5m.index
    orb_bars = df_5m[
        (idx.date == today) &
        (idx.hour == 9) &
        (idx.minute.isin([30, 35, 40]))
    ]

    if len(orb_bars) < 2:   # need at least 2 of 3 bars
        return None

    h = float(orb_bars["High"].max())
    l = float(orb_bars["Low"].min())
    result = {
        "date":  today,
        "high":  round(h, 2),
        "low":   round(l, 2),
        "range": round(h - l, 2),
        "mid":   round((h + l) / 2, 2),
    }
    _orb15_cache[ticker] = result
    return result


# -- Indicator helpers ---------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# -- SPY relative strength (cached 5 min) -------------------------------------

_spy_cache: dict = {"pct": 0.0, "ts": 0.0}


def get_spy_day_pct() -> float:
    """SPY % change vs yesterday's close. Cached 5 min."""
    if time.time() - _spy_cache["ts"] < 300:
        return _spy_cache["pct"]
    try:
        df = get_bars("SPY", "1d", days=5)
        if df is not None and len(df) >= 2:
            pct = float(
                (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
            )
            _spy_cache["pct"] = round(pct, 2)
            _spy_cache["ts"]  = time.time()
            return _spy_cache["pct"]
    except Exception as exc:
        logger.warning(f"SPY day_pct failed: {exc}")
    return 0.0


# -- 4-Hour EMA trend (cached 60 min) -----------------------------------------

_h4_cache: dict = {}   # {ticker: {"trend": str, "ts": float}}


def _4h_ema_trend(ticker: str) -> str:
    """4-hour EMA20 vs EMA50 trend. BULLISH / BEARISH / NEUTRAL. Cached 60 min."""
    cached = _h4_cache.get(ticker)
    if cached and time.time() - cached["ts"] < 3600:
        return cached["trend"]

    try:
        df = get_bars(ticker, "1h", days=60)

        if df is None or len(df) < 40:
            result = "NEUTRAL"
        else:
            df4 = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last",  "Volume": "sum",
            }).dropna()

            if len(df4) < 20:
                result = "NEUTRAL"
            else:
                close = df4["Close"]
                ema20 = close.ewm(span=20, adjust=False).mean()
                ema50 = close.ewm(span=50, adjust=False).mean()
                last  = float(close.iloc[-1])
                e20   = float(ema20.iloc[-1])
                e50   = float(ema50.iloc[-1])
                if last > e20 > e50:
                    result = "BULLISH"
                elif last < e20 < e50:
                    result = "BEARISH"
                else:
                    result = "NEUTRAL"
    except Exception as exc:
        logger.debug(f"4H EMA failed ({ticker}): {exc}")
        result = "NEUTRAL"

    _h4_cache[ticker] = {"trend": result, "ts": time.time()}
    return result


# -- Earnings proximity (cached 4 h) ------------------------------------------

_earnings_cache: dict = {}   # {ticker: {"days": int|None, "ts": float}}


def get_days_to_earnings(ticker: str) -> int | None:
    """
    Returns calendar days until next earnings, or None if unavailable.
    Cached 4 hours per ticker.
    """
    cached = _earnings_cache.get(ticker)
    if cached and time.time() - cached["ts"] < 4 * 3600:
        return cached["days"]

    result = None
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        cal = tkr.calendar
        if cal is not None and not (hasattr(cal, "empty") and cal.empty):
            earn_dt = None
            if hasattr(cal, "index") and "Earnings Date" in cal.index:
                earn_dt = pd.Timestamp(cal.loc["Earnings Date"].iloc[0])
            elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                earn_dt = pd.Timestamp(cal["Earnings Date"].iloc[0])

            if earn_dt is not None and not pd.isna(earn_dt):
                if earn_dt.tzinfo is None:
                    earn_dt = earn_dt.tz_localize("America/New_York")
                else:
                    earn_dt = earn_dt.tz_convert("America/New_York")
                days   = (earn_dt.date() - datetime.now(_ET).date()).days
                result = max(0, days)
    except Exception as exc:
        logger.debug(f"Earnings fetch failed ({ticker}): {exc}")

    _earnings_cache[ticker] = {"days": result, "ts": time.time()}
    return result


# -- Market structure (HH/HL or LH/LL on today's 30-min bars) ----------------

def _market_structure(df: pd.DataFrame) -> dict:
    """
    Check today's 30-min bars for HH+HL (bullish) or LH+LL (bearish) structure.
    Uses last 3 bars of today's session. Returns {"bullish": bool, "bearish": bool}.
    """
    today = datetime.now(_ET).date()
    idx   = df.index.tz_convert(_ET) if df.index.tzinfo else df.index
    tod   = df[idx.date == today]

    if len(tod) < 3:
        return {"bullish": False, "bearish": False}

    highs = tod["High"].values[-3:]
    lows  = tod["Low"].values[-3:]

    bullish = highs[-1] > highs[-2] and lows[-1] > lows[-2]
    bearish = highs[-1] < highs[-2] and lows[-1] < lows[-2]

    return {"bullish": bool(bullish), "bearish": bool(bearish)}


# -- Public API ----------------------------------------------------------------

def analyze(ticker: str) -> dict | None:
    df = fetch_30min(ticker)
    if df is None or len(df) < 52:
        logger.warning(f"{ticker}: only {len(df) if df is not None else 0} bars -- skipping")
        return None

    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(_ET)
    else:
        df.index = df.index.tz_convert(_ET)

    # 15-min ORB (9:45 AM signal) takes priority over 30-min ORB (10:00 AM signal)
    now_et = datetime.now(_ET)
    if now_et.hour == 9 and now_et.minute >= 44 or now_et.hour >= 10:
        df_5m = fetch_5min(ticker)
        orb_override = _orb_15min(ticker, df_5m) if df_5m is not None else None
    else:
        orb_override = None

    close  = df["Close"]
    volume = df["Volume"]

    rsi                          = _rsi(close)
    macd_line, signal_line, hist = _macd(close)
    ema20                        = close.ewm(span=20, adjust=False).mean()
    ema50                        = close.ewm(span=50, adjust=False).mean()
    atr                          = _atr(df)
    vol_avg20                    = volume.rolling(20).mean()
    vwap_series                  = _vwap(df)

    def s(series, i=-1):
        return float(series.iloc[i])

    last_close     = s(close)
    last_vwap      = s(vwap_series)
    last_vol_ratio = float(volume.iloc[-1] / vol_avg20.iloc[-1]) if vol_avg20.iloc[-1] > 0 else 1.0

    orb_data   = orb_override if orb_override is not None else _orb_from_df(df)
    pdh_pdl    = _pdh_pdl_from_df(df)
    vwap_b     = _vwap_bands(df)
    mkt_struct = _market_structure(df)
    pdc        = pdh_pdl.get("pdc")
    day_pct    = round((last_close - pdc) / pdc * 100, 2) if pdc else 0.0

    try:
        from sector_filter import get_sector_trend, get_sector_etf
        sector_trend = get_sector_trend(ticker)
        sector_etf   = get_sector_etf(ticker)
    except Exception:
        sector_trend = "NEUTRAL"
        sector_etf   = "?"

    h4_trend         = _4h_ema_trend(ticker)
    days_to_earnings = get_days_to_earnings(ticker)

    return {
        "ticker":             ticker,
        "price":              last_close,
        "rsi":                s(rsi),
        "rsi_prev":           s(rsi, -2),
        "rsi_rising":         s(rsi) > s(rsi, -2),
        "macd":               s(macd_line),
        "macd_signal":        s(signal_line),
        "macd_hist":          s(hist),
        "macd_hist_prev":     s(hist, -2),
        "macd_bullish_cross": s(macd_line, -2) < s(signal_line, -2) and s(macd_line) > s(signal_line),
        "macd_bearish_cross": s(macd_line, -2) > s(signal_line, -2) and s(macd_line) < s(signal_line),
        "ema20":              s(ema20),
        "ema50":              s(ema50),
        "price_above_ema20":  last_close > s(ema20),
        "price_above_ema50":  last_close > s(ema50),
        "ema20_above_ema50":  s(ema20) > s(ema50),
        "atr":                s(atr),
        "volume_ratio":       last_vol_ratio,
        "vwap":               last_vwap,
        "price_above_vwap":   last_close > last_vwap,
        "bar_bullish":        float(df["Close"].iloc[-1]) > float(df["Open"].iloc[-1]),
        "orb":                orb_data,
        "pdh":                pdh_pdl.get("pdh"),
        "pdl":                pdh_pdl.get("pdl"),
        "pdc":                pdc,
        "day_pct":            day_pct,
        "vwap_bands":         vwap_b,
        "sector_trend":       sector_trend,
        "sector_etf":         sector_etf,
        "h4_trend":           h4_trend,
        "days_to_earnings":   days_to_earnings,
        "market_structure":   mkt_struct,
    }
