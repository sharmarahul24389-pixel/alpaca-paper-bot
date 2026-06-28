import logging
import yfinance as yf

logger = logging.getLogger(__name__)

# Put/Call ratio thresholds
BULLISH_PC = 0.7   # Low puts vs calls = market expects upside
BEARISH_PC = 1.3   # High puts vs calls = market expects downside


def get_options_sentiment(ticker: str) -> str:
    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return "NEUTRAL"

        chain = t.option_chain(dates[0])
        call_vol = float(chain.calls["volume"].fillna(0).sum())
        put_vol  = float(chain.puts["volume"].fillna(0).sum())

        if call_vol + put_vol < 100:
            return "NEUTRAL"

        pc_ratio = put_vol / call_vol if call_vol > 0 else 1.0

        if pc_ratio < BULLISH_PC:
            return "BULLISH"
        elif pc_ratio > BEARISH_PC:
            return "BEARISH"
        return "NEUTRAL"
    except Exception as exc:
        logger.warning(f"Options flow failed ({ticker}): {exc}")
        return "NEUTRAL"
