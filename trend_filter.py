import logging
import yfinance as yf
from analyzer import _flatten_columns

logger = logging.getLogger(__name__)


def get_daily_trend(ticker: str) -> str:
    try:
        df = yf.download(ticker, period="60d", interval="1d",
                         auto_adjust=True, progress=False)
        df = _flatten_columns(df)
        df.dropna(inplace=True)
        if len(df) < 50:
            return "NEUTRAL"

        close = df["Close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

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
        data = yf.download([ticker, benchmark], period=f"{days + 5}d",
                           interval="1d", auto_adjust=True, progress=False)
        close = data["Close"]
        if ticker not in close.columns or benchmark not in close.columns:
            return 1.0
        ticker_ret = float(close[ticker].dropna().pct_change(days).iloc[-1])
        bench_ret = float(close[benchmark].dropna().pct_change(days).iloc[-1])
        if bench_ret == 0:
            return 1.0
        return round(ticker_ret / bench_ret, 2)
    except Exception as exc:
        logger.warning(f"RS failed ({ticker}): {exc}")
        return 1.0
