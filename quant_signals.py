"""
Quant Signals v10 — Statistical Edge Layer

Two complementary modules:

  1. Momentum Ranking
     Ranks tickers by 20-day price return.  Call once per trading day
     (refreshed automatically); returns top-N long candidates and
     bottom-N short candidates.  Used to pre-filter the quant scan
     universe so only sustained leaders / laggards are examined.

  2. Z-Score Mean Reversion
     Detects when price is >= 2.0 standard deviations from its 20-day
     rolling mean — a statistically stretched condition that tends to
     snap back.  Fires 10:30 AM – 2:00 PM ET, after the ORB window
     closes.  Grade B at |z| >= 2.5, Grade C at |z| >= 2.0.

These signals are additive to the ORB-momentum signals and share the
same daily quota / dedup system.
"""
import logging
from datetime import datetime

import pandas as pd
import pytz
import yfinance as yf

from analyzer import _flatten_columns, _atr
from config import TIMEZONE
from position_sizer import calculate_position
from signal_generator import Signal

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

_Z_LOOKBACK  = 20    # days for rolling mean / std
_Z_STRONG    = 2.0   # |z| >= this → signal fires
_Z_HIGH_CONV = 2.5   # |z| >= this → Grade B (else Grade C)
_MOM_PERIOD  = 20    # days for momentum return ranking
_TOP_N       = 12    # top + bottom N returned by rank_momentum
_ATR_SL_MULT = 1.5   # stop = 1.5 × ATR (tighter than ORB 2.0×)
_QT_RR       = 2.0   # fixed 2:1 reward-to-risk for mean reversion


# ---------------------------------------------------------------------------
# 1. MOMENTUM RANKING
# ---------------------------------------------------------------------------

def rank_momentum(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Batch-download 20-day returns for all tickers.
    Returns (top_N_longs, bottom_N_shorts) sorted by momentum strength.
    Falls back to first/last N of the input list on any failure.
    """
    try:
        raw = yf.download(
            tickers,
            period="30d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        # Multi-ticker download returns MultiIndex columns; flatten to Close
        if hasattr(raw.columns, "levels"):
            close = raw["Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw

        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0] if len(tickers) == 1 else "?")

        returns: dict[str, float] = {}
        for ticker in close.columns:
            series = close[str(ticker)].dropna()
            if len(series) >= _MOM_PERIOD:
                start = float(series.iloc[-_MOM_PERIOD])
                end   = float(series.iloc[-1])
                if start > 0:
                    returns[str(ticker)] = round((end - start) / start * 100, 2)

        if not returns:
            logger.warning("rank_momentum: no valid return data — using raw order")
            return tickers[:_TOP_N], tickers[-_TOP_N:]

        ranked = sorted(returns, key=lambda t: returns[t], reverse=True)
        longs  = ranked[:_TOP_N]
        shorts = ranked[-_TOP_N:]

        logger.info(
            f"Momentum rank — top longs: {longs[:3]}  "
            f"({returns.get(longs[0], 0):+.1f}% .. {returns.get(longs[-1], 0):+.1f}%)  |  "
            f"top shorts: {shorts[:3]}  "
            f"({returns.get(shorts[0], 0):+.1f}% .. {returns.get(shorts[-1], 0):+.1f}%)"
        )
        return longs, shorts

    except Exception as exc:
        logger.error(f"rank_momentum failed: {exc}")
        return tickers[:_TOP_N], tickers[-_TOP_N:]


# ---------------------------------------------------------------------------
# 2. BATCH Z-SCORE COMPUTATION
# ---------------------------------------------------------------------------

def batch_z_scores(tickers: list[str]) -> dict[str, dict]:
    """
    Download 60 days of daily bars for all tickers in a single call.
    Computes 20-day rolling z-score for each.
    Returns only tickers whose |z| >= _Z_STRONG — these are the candidates
    that run_quant_scan will build signals from.

    Return format: { "AAPL": {"z_score": -2.31, "price": 182.5, "mean": 190.1, "std": 3.3} }
    """
    try:
        raw = yf.download(
            tickers,
            period="60d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if hasattr(raw.columns, "levels"):
            close = raw["Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw

        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0] if len(tickers) == 1 else "?")

        results: dict[str, dict] = {}
        for ticker in close.columns:
            series = close[str(ticker)].dropna()
            if len(series) < _Z_LOOKBACK + 2:
                continue
            roll_mean = series.rolling(_Z_LOOKBACK).mean()
            roll_std  = series.rolling(_Z_LOOKBACK).std()
            price = float(series.iloc[-1])
            mean  = float(roll_mean.iloc[-1])
            std   = float(roll_std.iloc[-1])
            if std <= 0 or price <= 0:
                continue
            z = (price - mean) / std
            if abs(z) < _Z_STRONG:
                continue  # not stretched enough
            results[str(ticker)] = {
                "z_score": round(z, 2),
                "price":   round(price, 2),
                "mean":    round(mean, 2),
                "std":     round(std, 2),
            }

        logger.info(
            f"batch_z_scores: {len(results)}/{len(tickers)} tickers "
            f"with |z| >= {_Z_STRONG}"
        )
        return results

    except Exception as exc:
        logger.error(f"batch_z_scores failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# 3. QUANT SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_quant_signal(
    ticker: str,
    z_data: dict | None = None,
    rank: int | None = None,
) -> Signal | None:
    """
    Build a z-score mean-reversion Signal.

    Parameters
    ----------
    ticker  : Stock symbol
    z_data  : Pre-computed dict from batch_z_scores; fetched on-demand if None.
    rank    : Optional momentum rank (1 = strongest; shown in signal reasons).

    Returns None if:
      - Outside the 10:30 AM – 2:00 PM ET window
      - |z| < _Z_STRONG
      - ATR fetch fails
    """
    now_et = datetime.now(_ET)
    mins   = now_et.hour * 60 + now_et.minute
    # 10:30 AM = 630 min; 2:00 PM = 840 min
    if not (630 <= mins < 840):
        return None

    # Compute z-score if not provided from batch
    if z_data is None:
        try:
            raw = yf.download(
                ticker, period="60d", interval="1d",
                auto_adjust=True, progress=False,
            )
            raw = _flatten_columns(raw)
            raw.dropna(inplace=True)
            if len(raw) < _Z_LOOKBACK + 2:
                return None
            close     = raw["Close"]
            roll_mean = close.rolling(_Z_LOOKBACK).mean()
            roll_std  = close.rolling(_Z_LOOKBACK).std()
            price = float(close.iloc[-1])
            mean  = float(roll_mean.iloc[-1])
            std   = float(roll_std.iloc[-1])
            if std <= 0 or price <= 0:
                return None
            z = (price - mean) / std
            if abs(z) < _Z_STRONG:
                return None
            z_data = {
                "z_score": round(z, 2),
                "price":   round(price, 2),
                "mean":    round(mean, 2),
                "std":     round(std, 2),
            }
        except Exception as exc:
            logger.error(f"Z-score compute failed ({ticker}): {exc}")
            return None

    z     = z_data["z_score"]
    price = z_data["price"]

    # ATR from 30-min bars — used for stop sizing
    try:
        df30 = yf.download(
            ticker, period="5d", interval="30m",
            auto_adjust=True, progress=False,
        )
        df30 = _flatten_columns(df30)
        df30.dropna(inplace=True)
        if df30.empty:
            return None
        atr_val = float(_atr(df30).iloc[-1])
        if atr_val <= 0:
            return None
    except Exception as exc:
        logger.error(f"ATR fetch failed ({ticker}): {exc}")
        return None

    sl_dist = atr_val * _ATR_SL_MULT
    grade   = "B" if abs(z) >= _Z_HIGH_CONV else "C"
    conf    = min(100, int(abs(z) / 3.0 * 100))

    rank_note = f"  (momentum rank #{rank})" if rank is not None else ""

    if z <= -_Z_STRONG:   # oversold → BUY
        entry     = price
        stop_loss = round(entry - sl_dist, 2)
        target    = round(entry + sl_dist * _QT_RR, 2)
        reasons   = [
            f"Z-score {z:.2f} — {abs(z):.1f}σ below 20d mean ${z_data['mean']:.2f}{rank_note}",
            f"Mean reversion BUY — price statistically stretched to downside",
        ]
        return Signal(
            ticker=ticker, direction="BUY", entry=entry,
            stop_loss=stop_loss, target=target, rr=_QT_RR,
            confidence=conf, grade=grade, reasons=reasons,
            position=calculate_position(entry, stop_loss, grade),
        )

    if z >= _Z_STRONG:    # overbought → SELL
        entry     = price
        stop_loss = round(entry + sl_dist, 2)
        target    = round(entry - sl_dist * _QT_RR, 2)
        reasons   = [
            f"Z-score {z:.2f} — {abs(z):.1f}σ above 20d mean ${z_data['mean']:.2f}{rank_note}",
            f"Mean reversion SELL — price statistically stretched to upside",
        ]
        return Signal(
            ticker=ticker, direction="SELL", entry=entry,
            stop_loss=stop_loss, target=target, rr=_QT_RR,
            confidence=conf, grade=grade, reasons=reasons,
            position=calculate_position(entry, stop_loss, grade),
        )

    return None
