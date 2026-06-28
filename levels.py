import logging
import yfinance as yf
from analyzer import _flatten_columns

logger = logging.getLogger(__name__)

PROXIMITY_PCT = 0.005  # Within 0.5% of a level


def get_pivot_levels(ticker: str) -> dict:
    try:
        df = yf.download(ticker, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        df = _flatten_columns(df)
        df.dropna(inplace=True)
        if len(df) < 2:
            return {}

        prev = df.iloc[-2]
        high = float(prev["High"])
        low = float(prev["Low"])
        close = float(prev["Close"])

        P  = (high + low + close) / 3
        R1 = 2 * P - low
        R2 = P + (high - low)
        S1 = 2 * P - high
        S2 = P - (high - low)

        return {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}
    except Exception as exc:
        logger.warning(f"Pivot levels failed ({ticker}): {exc}")
        return {}


def check_near_level(price: float, levels: dict, direction: str) -> tuple:
    if not levels:
        return False, ""

    targets = ["S1", "S2", "P"] if direction == "BUY" else ["R1", "R2", "P"]
    for name in targets:
        level = levels.get(name, 0)
        if level > 0 and abs(price - level) / level <= PROXIMITY_PCT:
            return True, name
    return False, ""
