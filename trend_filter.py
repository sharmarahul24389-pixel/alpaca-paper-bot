import logging
from alpaca_data import get_bars, get_bars_multi

logger = logging.getLogger(__name__)


def get_daily_trend(ticker: str) -> str:
    try:
        df = get_bars(ticker, "1d", days=60)
        if df is None or len(df) < 50:
            return "NEUTRAL"

        close      = df["Close"]
        ema20      = close.ewm(span=20, adjust=False).mean()
        ema50      = close.ewm(span=50, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema20 = float(ema20.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])

        if last_close > last_ema20 > last_ema50:
            return "BULLISH"
        elif last_close < last_ema20 < last_ema50:
            return "BEARISH"
        return "NEUTRAL"
    except Exception as exc:
        logger.warning(f"Daily trend failed ({ticker}): {exc}")
        return "NEUTRAL"


def get_market_regime() -> str:
    return get_daily_trend("SPY")


def get_relative_strength(ticker: str, benchmark: str = "SPY", days: int = 5) -> float:
    try:
        data = get_bars_multi([ticker, benchmark], "1d", days=days + 10)
        if ticker not in data or benchmark not in data:
            return 1.0
        ticker_ret = float(data[ticker]["Close"].dropna().pct_change(days).iloc[-1])
        bench_ret  = float(data[benchmark]["Close"].dropna().pct_change(days).iloc[-1])
        if bench_ret == 0:
            return 1.0
        return round(ticker_ret / bench_ret, 2)
    except Exception as exc:
        logger.warning(f"RS failed ({ticker}): {exc}")
        return 1.0
