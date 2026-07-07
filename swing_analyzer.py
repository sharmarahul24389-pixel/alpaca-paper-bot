import logging

from alpaca_data import get_bars
from analyzer import _rsi, _macd, _atr
from config import (
    ACCOUNT_SIZE,
    MIN_SIGNAL_SCORE,
    SWING_ATR_SL,
    SWING_RISK_PCT,
    SWING_RR_RATIO,
)
from signal_generator import Signal

logger = logging.getLogger(__name__)

_SWING_MAX_SCORE = 13  # RSI(3)+MACD(3)+EMA(2)+Weekly(2)+Vol(1)+RS(1)+Options(1)


def _weekly_trend(ticker: str) -> str:
    try:
        df = get_bars(ticker, "1d", days=180)
        if df is None:
            return "NEUTRAL"
        df = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        if len(df) < 20:
            return "NEUTRAL"
        close = df["Close"]
        ema10 = close.ewm(span=10, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        last  = float(close.iloc[-1])
        e10   = float(ema10.iloc[-1])
        e20   = float(ema20.iloc[-1])
        if last > e10 > e20:
            return "BULLISH"
        if last < e10 < e20:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def analyze_swing(ticker: str) -> dict | None:
    try:
        df = get_bars(ticker, "1d", days=120)
        if df is None:
            return None
        df = df.dropna()
    except Exception as exc:
        logger.error(f"{ticker}: swing fetch failed — {exc}")
        return None

    if len(df) < 52:
        logger.warning(f"{ticker}: only {len(df)} daily bars — skipping swing")
        return None

    close  = df["Close"]
    volume = df["Volume"]

    rsi        = _rsi(close)
    macd_line, signal_line, histogram = _macd(close)
    ema20      = close.ewm(span=20, adjust=False).mean()
    ema50      = close.ewm(span=50, adjust=False).mean()
    atr        = _atr(df)
    vol_avg20  = volume.rolling(20).mean()

    def s(series, idx=-1):
        return float(series.iloc[idx])

    return {
        "ticker":            ticker,
        "price":             s(close),
        "rsi":               s(rsi),
        "rsi_prev":          s(rsi, -2),
        "rsi_rising":        s(rsi) > s(rsi, -2),
        "macd_hist":         s(histogram),
        "macd_hist_prev":    s(histogram, -2),
        "macd_bullish_cross": s(macd_line, -2) < s(signal_line, -2) and s(macd_line) > s(signal_line),
        "macd_bearish_cross": s(macd_line, -2) > s(signal_line, -2) and s(macd_line) < s(signal_line),
        "price_above_ema20": s(close) > s(ema20),
        "price_above_ema50": s(close) > s(ema50),
        "ema20_above_ema50": s(ema20) > s(ema50),
        "atr":               s(atr),
        "volume_ratio":      float(volume.iloc[-1] / vol_avg20.iloc[-1]) if vol_avg20.iloc[-1] > 0 else 1.0,
        "weekly_trend":      _weekly_trend(ticker),
    }


def generate_swing_signal(analysis: dict, events_warning: str = "") -> Signal:
    ticker    = analysis["ticker"]
    price     = analysis["price"]
    atr       = analysis["atr"]
    rsi       = analysis["rsi"]
    vol_ratio = analysis["volume_ratio"]

    buy_score  = 0
    sell_score = 0
    buy_reasons:  list[str] = []
    sell_reasons: list[str] = []

    # RSI daily (max 3 pts)
    if rsi <= 35:
        buy_score += 3
        buy_reasons.append(f"Daily RSI {rsi:.0f} oversold")
    elif 35 < rsi < 55 and analysis["rsi_rising"]:
        buy_score += 2
        buy_reasons.append(f"Daily RSI {rsi:.0f} rising")

    if rsi >= 65:
        sell_score += 3
        sell_reasons.append(f"Daily RSI {rsi:.0f} overbought")
    elif 45 < rsi < 65 and not analysis["rsi_rising"]:
        sell_score += 2
        sell_reasons.append(f"Daily RSI {rsi:.0f} falling")

    # MACD daily (max 3 pts)
    if analysis["macd_bullish_cross"]:
        buy_score += 3
        buy_reasons.append("Daily MACD bullish crossover")
    elif analysis["macd_hist"] > 0 and analysis["macd_hist"] > analysis["macd_hist_prev"]:
        buy_score += 1
        buy_reasons.append("Daily MACD expanding bullish")

    if analysis["macd_bearish_cross"]:
        sell_score += 3
        sell_reasons.append("Daily MACD bearish crossover")
    elif analysis["macd_hist"] < 0 and analysis["macd_hist"] < analysis["macd_hist_prev"]:
        sell_score += 1
        sell_reasons.append("Daily MACD expanding bearish")

    # EMA trend daily (max 2 pts)
    if analysis["price_above_ema20"] and analysis["ema20_above_ema50"]:
        buy_score += 2
        buy_reasons.append("Daily: Price > EMA20 > EMA50")
    elif not analysis["price_above_ema20"] and not analysis["ema20_above_ema50"]:
        sell_score += 2
        sell_reasons.append("Daily: Price < EMA20 < EMA50")

    # Weekly trend (max 2 pts)
    weekly = analysis["weekly_trend"]
    if weekly == "BULLISH":
        buy_score += 2
        buy_reasons.append("Weekly trend: BULLISH")
    elif weekly == "BEARISH":
        sell_score += 2
        sell_reasons.append("Weekly trend: BEARISH")

    # Volume surge (max 1 pt)
    if vol_ratio >= 1.5:
        if buy_score >= sell_score:
            buy_score += 1
            buy_reasons.append(f"Volume surge {vol_ratio:.1f}x avg")
        else:
            sell_score += 1
            sell_reasons.append(f"Volume surge {vol_ratio:.1f}x avg")

    # Position size for swing
    sl_dist     = atr * SWING_ATR_SL
    risk_amount = ACCOUNT_SIZE * SWING_RISK_PCT

    def _position(entry: float, sl: float) -> dict:
        risk_unit = abs(entry - sl)
        units     = max(1, int(risk_amount / risk_unit)) if risk_unit > 0.01 else 1
        return {
            "units":           units,
            "position_value":  round(units * entry, 2),
            "risk_amount":     round(risk_amount, 2),
            "target_pnl":      round(risk_amount * SWING_RR_RATIO, 2),
            "pct_of_account":  round(units * entry / ACCOUNT_SIZE * 100, 1),
        }

    if events_warning:
        if buy_score >= sell_score:
            buy_reasons.append(events_warning)
        else:
            sell_reasons.append(events_warning)

    if buy_score >= MIN_SIGNAL_SCORE and buy_score > sell_score:
        entry     = price
        stop_loss = round(entry - sl_dist, 2)
        target    = round(entry + sl_dist * SWING_RR_RATIO, 2)
        return Signal(
            ticker=ticker, direction="BUY", entry=entry,
            stop_loss=stop_loss, target=target, rr=SWING_RR_RATIO,
            confidence=min(100, int(buy_score / _SWING_MAX_SCORE * 100)),
            reasons=buy_reasons[:5], position=_position(entry, stop_loss),
        )

    if sell_score >= MIN_SIGNAL_SCORE and sell_score > buy_score:
        entry     = price
        stop_loss = round(entry + sl_dist, 2)
        target    = round(entry - sl_dist * SWING_RR_RATIO, 2)
        return Signal(
            ticker=ticker, direction="SELL", entry=entry,
            stop_loss=stop_loss, target=target, rr=SWING_RR_RATIO,
            confidence=min(100, int(sell_score / _SWING_MAX_SCORE * 100)),
            reasons=sell_reasons[:5], position=_position(entry, stop_loss),
        )

    return Signal(
        ticker=ticker, direction="WAIT", entry=price,
        stop_loss=0.0, target=0.0, rr=0.0, confidence=0,
        reasons=[], position={},
    )
