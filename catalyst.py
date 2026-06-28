"""
catalyst.py — Real-time catalyst detection.

Monitors:
  - Trump / Truth Social posts (sector sentiment)
  - Congress / Senate trades (stock direction)
  - SEC 8-K filings (material events — skip on filing day)
  - Sector ETF alignment (fight the sector = lower edge)
  - 20-day MA trend alignment (biggest single edge booster)

All results are cached to avoid hammering APIs.
"""
import logging
from datetime import datetime, date, timedelta

import feedparser
import pytz
import requests
import yfinance as yf

log = logging.getLogger(__name__)
_ET = pytz.timezone("America/New_York")

# ── Sector ETF map ────────────────────────────────────────────────────────────
STOCK_TO_SECTOR: dict[str, str] = {
    # Technology
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AMD":"XLK","INTC":"XLK",
    "QCOM":"XLK","AVGO":"XLK","MU":"XLK","AMAT":"XLK","LRCX":"XLK",
    "KLAC":"XLK","MRVL":"XLK","SMCI":"XLK","ARM":"XLK","TXN":"XLK",
    "CRM":"XLK","NOW":"XLK","ORCL":"XLK","ADSK":"XLK","INTU":"XLK",
    "SNOW":"XLK","DDOG":"XLK","MDB":"XLK","NET":"XLK","ZS":"XLK",
    "PANW":"XLK","CRWD":"XLK","OKTA":"XLK","PLTR":"XLK","S":"XLK",
    # Communication / Social
    "META":"XLC","GOOGL":"XLC","NFLX":"XLC","SPOT":"XLC",
    "PINS":"XLC","SNAP":"XLC","ROKU":"XLC",
    # Consumer / EV / Auto
    "AMZN":"XLY","TSLA":"XLY","SHOP":"XLY","UBER":"XLY","LYFT":"XLY",
    "F":"XLY","GM":"XLY","RIVN":"XLY","NIO":"XLY",
    # Finance / Crypto
    "JPM":"XLF","GS":"XLF","MS":"XLF","BAC":"XLF","C":"XLF",
    "V":"XLF","MA":"XLF","PYPL":"XLF","COIN":"XLF","HOOD":"XLF","MARA":"XLF",
    # Healthcare / Biotech
    "UNH":"XLV","LLY":"XLV","JNJ":"XLV","PFE":"XLV",
    "ABBV":"XLV","MRNA":"XLV","BNTX":"XLV",
    # Energy
    "XOM":"XLE","CVX":"XLE","OXY":"XLE","SLB":"XLE","COP":"XLE",
    # Defense (Trump-sensitive)
    "LMT":"XAR","RTX":"XAR","NOC":"XAR","GD":"XAR","BA":"XAR",
}

# Trump post keyword clusters
_TRUMP_ENERGY_BULL  = ["drill","oil","lng","pipeline","fossil","coal","deregulat","energy dominan"]
_TRUMP_TECH_BEAR    = ["tiktok","china tech","ban app","sanction","tariff tech"]
_TRUMP_DEFENSE_BULL = ["military","defense","nato","border","wall","troop","weapon"]
_TRUMP_TARIFF_BEAR  = ["tariff","trade war","import tax","china tariff","mexico tariff",
                        "reciprocal","levy on"]
_TRUMP_MARKET_BULL  = ["market is great","economy is booming","stock market","best economy",
                        "tremendous","beautiful numbers"]

# ── Caches ────────────────────────────────────────────────────────────────────
_trump_cache: dict    = {"ts": None, "result": None}
_congress_cache: dict = {"ts": None, "result": {}}
_sec_cache: dict      = {}     # ticker → {"ts": datetime, "has_8k": bool}
_sector_cache: dict   = {}     # etf → {"ts": datetime, "pct": float}
_trend_cache: dict    = {}     # "TICKER_DIR" → {"ts": datetime, "ok": bool, "reason": str}


# ── Trump / Truth Social ──────────────────────────────────────────────────────

def get_trump_catalyst() -> dict:
    """
    Return:
      active       bool
      sentiment    BULLISH | BEARISH | NEUTRAL
      affected_sectors  list[str]
      affected_tickers  list[str]
      summary      str
    Cached 15 min.
    """
    global _trump_cache
    now = datetime.now(_ET)
    if _trump_cache["ts"] and (now - _trump_cache["ts"]).seconds < 900:
        return _trump_cache["result"]

    result = {"active": False, "sentiment": "NEUTRAL",
              "affected_sectors": [], "affected_tickers": [], "summary": ""}
    try:
        feed = feedparser.parse("https://truthsocial.com/@realDonaldTrump.rss")
        cutoff = now - timedelta(hours=4)
        recent = []
        for e in feed.entries[:15]:
            try:
                pub = datetime(*e.published_parsed[:6], tzinfo=pytz.utc).astimezone(_ET)
                if pub >= cutoff:
                    recent.append((e.title + " " + getattr(e, "summary", "")).lower())
            except Exception:
                pass
        if not recent:
            _trump_cache = {"ts": now, "result": result}
            return result

        text = " ".join(recent)
        notes = []

        if any(k in text for k in _TRUMP_TARIFF_BEAR):
            result["active"] = True
            result["affected_sectors"] += ["XLY", "XLK", "XLF"]
            result["sentiment"] = "BEARISH"
            notes.append("tariff post → consumer/tech/finance bearish")

        if any(k in text for k in _TRUMP_ENERGY_BULL):
            result["active"] = True
            result["affected_sectors"].append("XLE")
            result["affected_tickers"] += ["XOM", "CVX", "OXY"]
            if result["sentiment"] != "BEARISH":
                result["sentiment"] = "BULLISH"
            notes.append("energy drill post → XLE/OXY bullish")

        if any(k in text for k in _TRUMP_DEFENSE_BULL):
            result["active"] = True
            result["affected_tickers"] += ["LMT", "RTX", "NOC", "GD"]
            notes.append("defense/military post → defense stocks bullish")

        if any(k in text for k in _TRUMP_TECH_BEAR):
            result["active"] = True
            result["affected_sectors"].append("XLK")
            result["sentiment"] = "BEARISH"
            notes.append("tech ban post → XLK bearish")

        if any(k in text for k in _TRUMP_MARKET_BULL) and result["sentiment"] == "NEUTRAL":
            result["active"] = True
            result["sentiment"] = "BULLISH"
            notes.append("Trump bullish on market")

        result["summary"] = "Trump: " + " | ".join(notes) if notes else ""
        if result["active"]:
            log.info(f"Trump catalyst active: {result['summary']}")

    except Exception as e:
        log.debug(f"Trump RSS failed: {e}")

    _trump_cache = {"ts": now, "result": result}
    return result


# ── Congress / Senate trades ──────────────────────────────────────────────────

def get_congress_buys() -> dict[str, str]:
    """
    Return {ticker: "BUY"|"SELL"} for stocks with net 3+ congress buys/sells in last 30 days.
    Uses housestockwatcher.com free public API.
    Cached 1 hour.
    """
    global _congress_cache
    now = datetime.now(_ET)
    if _congress_cache["ts"] and (now - _congress_cache["ts"]).seconds < 3600:
        return _congress_cache["result"]

    result: dict[str, str] = {}
    try:
        r = requests.get("https://housestockwatcher.com/api",
                         timeout=12, headers={"User-Agent": "alpaca-paper-bot/1.0"})
        trades = r.json()
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        counts: dict[str, int] = {}
        for t in trades:
            if t.get("transaction_date", "") < cutoff:
                continue
            ticker = (t.get("ticker") or "").strip().upper()
            if not ticker or len(ticker) > 5 or ticker == "N/A":
                continue
            tx = (t.get("type") or "").upper()
            if "PURCHASE" in tx:
                counts[ticker] = counts.get(ticker, 0) + 1
            elif "SALE" in tx:
                counts[ticker] = counts.get(ticker, 0) - 1

        for ticker, score in counts.items():
            if score >= 3:
                result[ticker] = "BUY"
            elif score <= -3:
                result[ticker] = "SELL"

        if result:
            log.info(f"Congress catalyst tickers: {list(result.keys())[:8]}")
    except Exception as e:
        log.debug(f"Congress API failed: {e}")

    _congress_cache = {"ts": now, "result": result}
    return result


# ── SEC 8-K material events ───────────────────────────────────────────────────

def check_sec_8k(ticker: str) -> bool:
    """Return True if ticker filed an 8-K today (material event = higher uncertainty)."""
    global _sec_cache
    now = datetime.now(_ET)
    if ticker in _sec_cache and (now - _sec_cache[ticker]["ts"]).seconds < 1800:
        return _sec_cache[ticker]["has_8k"]

    has_8k = False
    try:
        today = date.today().isoformat()
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{ticker}%22&forms=8-K"
            f"&dateRange=custom&startdt={today}&enddt={today}"
        )
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "alpaca-paper-bot contact@example.com"})
        data = r.json()
        has_8k = int(data.get("total", {}).get("value", 0)) > 0
        if has_8k:
            log.info(f"SEC 8-K filed today for {ticker}")
    except Exception as e:
        log.debug(f"SEC EDGAR check failed for {ticker}: {e}")

    _sec_cache[ticker] = {"ts": now, "has_8k": has_8k}
    return has_8k


# ── Sector ETF alignment ──────────────────────────────────────────────────────

def get_sector_pct(etf: str) -> float:
    """Today's % change for a sector ETF. Cached 10 min."""
    now = datetime.now(_ET)
    if etf in _sector_cache and (now - _sector_cache[etf]["ts"]).seconds < 600:
        return _sector_cache[etf]["pct"]
    try:
        df = yf.download(etf, period="2d", interval="1d",
                         auto_adjust=True, progress=False)
        pct = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100
    except Exception:
        pct = 0.0
    _sector_cache[etf] = {"ts": now, "pct": round(pct, 2)}
    return pct


# ── 20-day MA trend alignment ─────────────────────────────────────────────────

def check_trend_alignment(ticker: str, direction: str) -> tuple[bool, str]:
    """
    BUY requires price >= SMA20 × 0.99 (within 1% below is ok — close to reclaim).
    SELL requires price <= SMA20 × 1.01.
    Cached 30 min.
    """
    now = datetime.now(_ET)
    key = f"{ticker}_{direction}"
    if key in _trend_cache and (now - _trend_cache[key]["ts"]).seconds < 1800:
        c = _trend_cache[key]
        return c["ok"], c["reason"]

    try:
        df = yf.download(ticker, period="30d", interval="1d",
                         auto_adjust=True, progress=False)
        close = float(df["Close"].iloc[-1])
        sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
        pct   = (close - sma20) / sma20 * 100

        if direction == "BUY":
            ok     = close >= sma20 * 0.99
            reason = (f"Trend OK: ${close:.2f} vs SMA20 ${sma20:.2f} ({pct:+.1f}%)"
                      if ok else
                      f"Trend miss: BUY but ${close:.2f} below SMA20 ${sma20:.2f} ({pct:+.1f}%)")
        else:
            ok     = close <= sma20 * 1.01
            reason = (f"Trend OK: ${close:.2f} vs SMA20 ${sma20:.2f} ({pct:+.1f}%)"
                      if ok else
                      f"Trend miss: SELL but ${close:.2f} above SMA20 ${sma20:.2f} ({pct:+.1f}%)")
    except Exception as e:
        log.debug(f"Trend check failed {ticker}: {e}")
        ok, reason = True, "Trend data unavailable — allowed"

    _trend_cache[key] = {"ts": now, "ok": ok, "reason": reason}
    return ok, reason


# ── Composite catalyst score ──────────────────────────────────────────────────

def get_catalyst_score(ticker: str, direction: str) -> tuple[int, list[str]]:
    """
    Returns (score, reasons).
      score >= +2  : strong confirmation — boost confidence
      score == 0   : neutral
      score <= -2  : opposing catalyst — skip or require Grade A
      score <= -4  : hard skip (e.g. tariff post + SEC 8-K)
    """
    score   = 0
    reasons: list[str] = []

    # ── Trump catalyst ────────────────────────────────────────────────────────
    trump = get_trump_catalyst()
    if trump["active"]:
        sector_etf = STOCK_TO_SECTOR.get(ticker, "")
        ticker_hit  = ticker in trump.get("affected_tickers", [])
        sector_hit  = sector_etf and sector_etf in trump.get("affected_sectors", [])
        if ticker_hit or sector_hit:
            sent = trump["sentiment"]
            if direction == "BUY" and sent == "BULLISH":
                score += 2; reasons.append("Trump post confirms BUY")
            elif direction == "SELL" and sent == "BEARISH":
                score += 2; reasons.append("Trump post confirms SELL")
            elif direction == "BUY" and sent == "BEARISH":
                score -= 3; reasons.append("Trump post OPPOSES BUY — risky")
            elif direction == "SELL" and sent == "BULLISH":
                score -= 3; reasons.append("Trump post OPPOSES SELL — risky")

    # ── Congress trades ───────────────────────────────────────────────────────
    congress = get_congress_buys()
    if ticker in congress:
        cdir = congress[ticker]
        if cdir == direction:
            score += 2; reasons.append(f"Congress net-buying {ticker}" if cdir=="BUY" else f"Congress net-selling {ticker}")
        else:
            score -= 1; reasons.append(f"Congress going {cdir} vs our {direction}")

    # ── SEC 8-K today ─────────────────────────────────────────────────────────
    if check_sec_8k(ticker):
        score -= 2; reasons.append("SEC 8-K filed today — skip or require A grade")

    # ── Sector ETF alignment ──────────────────────────────────────────────────
    sector_etf = STOCK_TO_SECTOR.get(ticker)
    if sector_etf:
        spct = get_sector_pct(sector_etf)
        if direction == "BUY":
            if spct >= 0.4:
                score += 1; reasons.append(f"Sector {sector_etf} {spct:+.1f}% — aligned")
            elif spct <= -0.6:
                score -= 2; reasons.append(f"Sector {sector_etf} {spct:+.1f}% — fighting sector")
        else:
            if spct <= -0.4:
                score += 1; reasons.append(f"Sector {sector_etf} {spct:+.1f}% — aligned")
            elif spct >= 0.6:
                score -= 2; reasons.append(f"Sector {sector_etf} {spct:+.1f}% — fighting sector")

    return score, reasons
