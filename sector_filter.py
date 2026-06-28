"""
Sector ETF alignment filter.
Maps each ticker to its sector ETF and checks if the sector
is trending in the same direction as the proposed trade.
Bearish sector + BUY signal = penalised heavily.
"""
import logging
from datetime import datetime

import pytz
import yfinance as yf

from analyzer import _flatten_columns

logger = logging.getLogger(__name__)
_ET    = pytz.timezone("America/New_York")

SECTOR_MAP = {
    # Technology
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AMD":"XLK","AVGO":"XLK",
    "QCOM":"XLK","TXN":"XLK","ORCL":"XLK","CRM":"XLK","CSCO":"XLK",
    "ADBE":"XLK","INTU":"XLK","IBM":"XLK","AMAT":"XLK","LRCX":"XLK",
    "KLAC":"XLK","MCHP":"XLK","SNPS":"XLK","CDNS":"XLK","NXPI":"XLK",
    "ADI":"XLK","ASML":"XLK","MU":"XLK","PANW":"XLK","ANET":"XLK",
    "MRVL":"XLK","CRWD":"XLK","FTNT":"XLK","ZS":"XLK","ON":"XLK",
    "ARM":"XLK","SMCI":"XLK","TTD":"XLK","WDAY":"XLK","ANSS":"XLK",
    "PLTR":"XLK","APP":"XLK","CDW":"XLK","PAYX":"XLK","TEAM":"XLK",
    "MSTR":"XLK","ROP":"XLI",
    # Communication Services
    "GOOGL":"XLC","GOOG":"XLC","META":"XLC","NFLX":"XLC",
    "DIS":"XLC","T":"XLC","VZ":"XLC","CHTR":"XLC",
    # Consumer Discretionary
    "AMZN":"XLY","TSLA":"XLY","NKE":"XLY","MCD":"XLY","BKNG":"XLY",
    "SBUX":"XLY","TJX":"XLY","ROST":"XLY","GM":"XLY","F":"XLY",
    "RIVN":"XLY","LCID":"XLY","MELI":"XLY","UBER":"XLY","DASH":"XLY",
    "ABNB":"XLY",
    # Consumer Staples
    "PG":"XLP","KO":"XLP","PEP":"XLP","PM":"XLP","MO":"XLP",
    "COST":"XLP","WMT":"XLP","MDLZ":"XLP","CL":"XLP","KDP":"XLP",
    "MNST":"XLP",
    # Financials
    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","C":"XLF",
    "V":"XLF","MA":"XLF","AXP":"XLF","BLK":"XLF","SCHW":"XLF",
    "SPGI":"XLF","CME":"XLF","ICE":"XLF","CB":"XLF","MMC":"XLF",
    "MCO":"XLF","PNC":"XLF","USB":"XLF","COF":"XLF","COIN":"XLF",
    "NU":"XLF","VRSK":"XLF","PYPL":"XLF",
    # Healthcare
    "UNH":"XLV","JNJ":"XLV","LLY":"XLV","ABBV":"XLV","MRK":"XLV",
    "TMO":"XLV","ABT":"XLV","DHR":"XLV","BMY":"XLV","AMGN":"XLV",
    "GILD":"XLV","VRTX":"XLV","REGN":"XLV","ISRG":"XLV","BSX":"XLV",
    "BDX":"XLV","SYK":"XLV","ZTS":"XLV","ELV":"XLV","CI":"XLV",
    "HCA":"XLV","HUM":"XLV","DXCM":"XLV","IDXX":"XLV","GEHC":"XLV",
    "BIIB":"XLV",
    # Energy
    "XOM":"XLE","CVX":"XLE","COP":"XLE","EOG":"XLE","SLB":"XLE",
    "MPC":"XLE","PSX":"XLE","VLO":"XLE","KMI":"XLE","WMB":"XLE",
    "OKE":"XLE","HAL":"XLE","LNG":"XLE","DVN":"XLE","FANG":"XLE",
    "FCX":"XLB","NEM":"XLB",
    # Industrials
    "GE":"XLI","CAT":"XLI","DE":"XLI","HON":"XLI","UNP":"XLI",
    "RTX":"XLI","GD":"XLI","NOC":"XLI","ETN":"XLI","EMR":"XLI",
    "ITW":"XLI","FDX":"XLI","NSC":"XLI","CARR":"XLI","PCAR":"XLI",
    "ODFL":"XLI","FAST":"XLI","CTAS":"XLI","GEV":"XLI","CPRT":"XLI",
    # Utilities / Real Estate
    "NEE":"XLU","SO":"XLU","CEG":"XLU","PLD":"XLRE",
    # Thematic semiconductor ETFs
    "DRAM":"XLK","SMH":"XLK","SOXX":"XLK",
    # Broad (skip sector filter)
    "SPY":"SPY","QQQ":"SPY","IWM":"SPY",
}

_cache: dict = {}   # {etf: {trend, pct, ts}}


def get_sector_etf(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "SPY")


def get_sector_trend(ticker: str) -> str:
    """Returns 'BULLISH', 'BEARISH', or 'NEUTRAL' for the ticker's sector."""
    etf = get_sector_etf(ticker)
    if etf == "SPY":
        return "NEUTRAL"

    now   = datetime.now(_ET)
    entry = _cache.get(etf)
    if entry and (now - entry["ts"]).seconds < 1800:
        return entry["trend"]

    try:
        df = yf.download(etf, period="5d", interval="1h",
                         auto_adjust=True, progress=False)
        df = _flatten_columns(df)
        df.dropna(inplace=True)

        if len(df) < 5:
            return "NEUTRAL"

        close = df["Close"]
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        price = float(close.iloc[-1])
        e9    = float(ema9.iloc[-1])
        e21   = float(ema21.iloc[-1])
        pct   = (price - float(close.iloc[-5])) / float(close.iloc[-5]) * 100

        if price > e9 > e21 and pct > 0.15:
            trend = "BULLISH"
        elif price < e9 < e21 and pct < -0.15:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        _cache[etf] = {"trend": trend, "pct": round(pct, 2), "ts": now}
        logger.info(f"Sector {etf}: {trend} ({pct:+.2f}%)")
        return trend

    except Exception as exc:
        logger.warning(f"Sector filter ({etf}): {exc}")
        return "NEUTRAL"
