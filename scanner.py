import logging
import pandas as pd

from alpaca_data import get_bars_multi
from config import TOP_MOVERS_COUNT, MIN_PRICE, MIN_VOLUME, MAX_TICKERS_TO_SCAN

logger = logging.getLogger(__name__)

# S&P 500 + Nasdaq-100 combined universe — hardcoded to avoid external fetch failures on Railway
_STOCK_UNIVERSE = [
    # ── Mega-cap / S&P 500 core ────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "LLY", "V", "XOM", "UNH", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "NFLX", "AMD", "KO", "PEP", "TMO", "ORCL",
    "ACN", "MCD", "CSCO", "LIN", "ABT", "TXN", "WMT", "DHR", "NEE", "NKE",
    "PM", "ADBE", "QCOM", "DIS", "AMGN", "UNP", "MS", "INTU", "IBM", "GE",
    "LOW", "UBER", "SPGI", "RTX", "GS", "BLK", "SYK", "PLD", "CAT", "AXP",
    "ELV", "T", "ISRG", "VRTX", "BKNG", "CI", "MDLZ", "ADI", "GILD", "ADP",
    "SBUX", "TJX", "CB", "MMC", "SO", "C", "SCHW", "ZTS", "MO", "CME",
    "ETN", "PGR", "DE", "BDX", "REGN", "BMY", "BSX", "WM", "AON", "SLB",
    "ITW", "NOC", "APD", "HCA", "EOG", "USB", "PYPL", "PNC", "NSC", "FDX",
    "CL", "TGT", "EMR", "COF", "CARR", "OKE", "PSA", "ECL", "MET", "KLAC",
    "LRCX", "AMAT", "MCHP", "SNPS", "CDNS", "NXPI", "GD", "HUM", "MCO", "ICE",
    "FCX", "COP", "DVN", "MPC", "PSX", "VLO", "KMI", "WMB", "LNG", "HAL",
    "F", "GM", "RIVN", "LCID", "PLTR", "COIN", "HOOD", "SOFI", "NU", "MSTR",
    "VST",   # Vistra Energy
    "CRDO",  # Credo Technology
    # ── Nasdaq-100 additions ───────────────────────────────────────────────────
    "ASML",  # ASML Holding
    "MU",    # Micron Technology
    "PANW",  # Palo Alto Networks
    "ANET",  # Arista Networks
    "MRVL",  # Marvell Technology
    "CRWD",  # CrowdStrike Holdings
    "DXCM",  # DexCom
    "TEAM",  # Atlassian
    "TTD",   # The Trade Desk
    "IDXX",  # IDEXX Laboratories
    "ON",    # ON Semiconductor
    "VRSK",  # Verisk Analytics
    "CPRT",  # Copart
    "WDAY",  # Workday
    "PAYX",  # Paychex
    "MNST",  # Monster Beverage
    "BIIB",  # Biogen
    "ZS",    # Zscaler
    "FTNT",  # Fortinet
    "PCAR",  # PACCAR
    "ROP",   # Roper Technologies
    "ROST",  # Ross Stores
    "MELI",  # MercadoLibre
    "CTAS",  # Cintas
    "CEG",   # Constellation Energy
    "APP",   # AppLovin
    "DASH",  # DoorDash
    "ABNB",  # Airbnb
    "ARM",   # Arm Holdings
    "SMCI",  # Super Micro Computer
    "FANG",  # Diamondback Energy
    "GEV",   # GE Vernova
    "KDP",   # Keurig Dr Pepper
    "FAST",  # Fastenal
    "ODFL",  # Old Dominion Freight
    "GEHC",  # GE HealthCare
    "CDW",   # CDW Corporation
    "ANSS",  # ANSYS
    # ── Broad market ETFs ─────────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "ARKK",
    # ── Specialty / thematic ETFs ─────────────────────────────────────────────
    "DRAM",  # Memory semiconductor ETF
]


# User watchlist — individual stocks from themed ETFs; always scanned first
_USER_WATCHLIST = [
    # Quantum computing (QTUM holdings)
    "IONQ",  # IonQ — pure-play trapped-ion quantum
    "HON",   # Honeywell Quantum Solutions
    # Uranium (URA holdings)
    "CCJ",   # Cameco — largest liquid uranium producer
    "UEC",   # Uranium Energy Corp
    # AI infrastructure & power
    "CRDO",  # Credo Technology — high-speed data centre interconnect
    "ARM",   # Arm Holdings — chip IP, AI edge compute
    "GEV",   # GE Vernova — power/energy transition
    "VST",   # Vistra Energy — nuclear + gas power for AI data centres
    # Tech additions from XLK not yet in main universe
    "INTC",  # Intel
    "HPQ",   # HP Inc
    "NTAP",  # NetApp
    "CTSH",  # Cognizant
    "KEYS",  # Keysight Technologies
    "GLW",   # Corning
    # Utilities from XLU not yet in main universe
    "DUK",   # Duke Energy
    "D",     # Dominion Energy
    "EXC",   # Exelon
    "AEP",   # American Electric Power
    "XEL",   # Xcel Energy
    "SRE",   # Sempra
    "PEG",   # PSEG
    "WEC",   # WEC Energy Group
    "AWK",   # American Water Works
    "ETR",   # Entergy
]


_premarket_order: list[str] = []  # refreshed each morning by sort_by_premarket_activity()

def sort_by_premarket_activity() -> None:
    """
    Called at 9:20 AM. Fetches 2-day daily bars from Alpaca for the full universe
    and sorts tickers by abs(overnight_gap%) so the most active stocks fill the
    MAX_TICKERS_TO_SCAN window first. Falls back to static order on any failure.
    """
    global _premarket_order
    all_tickers = _USER_WATCHLIST + [t for t in _STOCK_UNIVERSE if t not in _USER_WATCHLIST]
    try:
        data = get_bars_multi(all_tickers, "1d", days=5)
        scores: dict[str, float] = {}
        for t, df in data.items():
            try:
                closes = df["Close"].dropna()
                if len(closes) >= 2:
                    scores[t] = abs(float(closes.iloc[-1]) - float(closes.iloc[-2])) / float(closes.iloc[-2])
            except Exception:
                pass
        wl_sorted   = sorted([t for t in _USER_WATCHLIST if t in scores],
                             key=lambda t: scores.get(t, 0), reverse=True)
        rest_sorted = sorted([t for t in all_tickers if t not in _USER_WATCHLIST and t in scores],
                             key=lambda t: scores.get(t, 0), reverse=True)
        _premarket_order = wl_sorted + rest_sorted
        logger.info(f"Pre-market sort: top movers = {_premarket_order[:10]}")
    except Exception as exc:
        logger.warning(f"Pre-market sort failed ({exc}), using static order")
        _premarket_order = []


def get_watchlist_tickers() -> list[str]:
    """
    Returns tickers sorted by pre-market activity (if sort_by_premarket_activity()
    has run today) so MAX_TICKERS_TO_SCAN window captures the day's active stocks.
    Falls back to static order.
    """
    if _premarket_order:
        return _premarket_order
    rest = [t for t in _STOCK_UNIVERSE if t not in _USER_WATCHLIST]
    return _USER_WATCHLIST + rest


# Keep old name so other modules do not break
def get_sp500_tickers() -> list[str]:
    return _STOCK_UNIVERSE


def get_top_movers(
    count: int = TOP_MOVERS_COUNT,
    min_price: float = MIN_PRICE,
    min_volume: int = MIN_VOLUME,
) -> list[dict]:
    tickers = get_watchlist_tickers()[:MAX_TICKERS_TO_SCAN]
    logger.info(f"Scanning {len(tickers)} tickers via Alpaca ...")

    data = get_bars_multi(tickers, "1d", days=5)
    if not data:
        logger.error("Alpaca bulk daily bars returned empty")
        return []

    movers: list[dict] = []
    for ticker, df in data.items():
        try:
            closes  = df["Close"].dropna()
            volumes = df["Volume"].dropna()
            if len(closes) < 2 or len(volumes) < 1:
                continue

            curr_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2])
            volume     = float(volumes.iloc[-1])

            if curr_close < min_price or volume < min_volume:
                continue

            pct_change     = (curr_close - prev_close) / prev_close
            momentum_score = abs(pct_change) * volume

            movers.append({
                "ticker":         ticker,
                "price":          curr_close,
                "pct_change":     pct_change,
                "volume":         int(volume),
                "momentum_score": momentum_score,
            })
        except Exception:
            continue

    movers.sort(key=lambda x: x["momentum_score"], reverse=True)
    top     = movers[:count]
    summary = [(m["ticker"], "{:+.1%}".format(m["pct_change"])) for m in top]
    logger.info(f"Top movers: {summary}")
    return top


def get_postmarket_setups(count: int = MAX_TICKERS_TO_SCAN) -> dict:
    """
    After-close scan (run at 4:45 PM ET).
    Categorises tickers into tomorrow's watchlist based on today's close:
      breakouts      — closed above yesterday's high (PDH confirmed)
      breakdowns     — closed below yesterday's low  (PDL confirmed)
      approaching_pdh — within 0.5 % of PDH from below (coiled spring)
      approaching_pdl — within 0.5 % of PDL from above (teetering)
      high_volume    — today's vol > 2× 20-day avg (institutional activity)
      gap_fills      — opened >1.5 % gap but closed flat (fade candidate)
    """
    tickers = get_watchlist_tickers()[:count]
    logger.info(f"Post-market setup scan: fetching 30d daily bars for {len(tickers)} tickers via Alpaca")

    data = get_bars_multi(tickers, "1d", days=30)
    if not data:
        logger.error("Post-market Alpaca fetch returned empty")
        return {}

    results: dict = {
        "breakouts":      [],
        "breakdowns":     [],
        "approaching_pdh": [],
        "approaching_pdl": [],
        "high_volume":    [],
        "gap_fills":      [],
    }

    for ticker, df in data.items():
        try:
            closes  = df["Close"].dropna()
            highs   = df["High"].dropna()
            lows    = df["Low"].dropna()
            opens   = df["Open"].dropna()
            volumes = df["Volume"].dropna()

            if len(closes) < 22 or len(volumes) < 21:
                continue

            today_close  = float(closes.iloc[-1])
            today_high   = float(highs.iloc[-1])
            today_low    = float(lows.iloc[-1])
            today_open   = float(opens.iloc[-1])
            today_vol    = float(volumes.iloc[-1])
            prev_close   = float(closes.iloc[-2])
            prev_high    = float(highs.iloc[-2])
            prev_low     = float(lows.iloc[-2])

            if today_close < MIN_PRICE:
                continue

            vol_avg20 = float(volumes.iloc[-21:-1].mean())
            vol_ratio = today_vol / vol_avg20 if vol_avg20 > 0 else 1.0

            t = ticker

            # ── Breakout: closed above yesterday's high ────────────────────────
            if today_close > prev_high * 1.001:
                results["breakouts"].append({
                    "ticker":    t,
                    "close":     round(today_close, 2),
                    "pdh":       round(prev_high, 2),
                    "pct_above": round((today_close - prev_high) / prev_high * 100, 2),
                    "vol_ratio": round(vol_ratio, 1),
                })

            # ── Breakdown: closed below yesterday's low ────────────────────────
            elif today_close < prev_low * 0.999:
                results["breakdowns"].append({
                    "ticker":    t,
                    "close":     round(today_close, 2),
                    "pdl":       round(prev_low, 2),
                    "pct_below": round((prev_low - today_close) / prev_low * 100, 2),
                    "vol_ratio": round(vol_ratio, 1),
                })

            # ── Approaching PDH: within 0.5 % below, not yet broken ────────────
            elif prev_high * 0.995 <= today_close <= prev_high:
                results["approaching_pdh"].append({
                    "ticker":   t,
                    "close":    round(today_close, 2),
                    "pdh":      round(prev_high, 2),
                    "pct_away": round((prev_high - today_close) / prev_high * 100, 2),
                })

            # ── Approaching PDL: within 0.5 % above, not yet broken ────────────
            elif prev_low <= today_close <= prev_low * 1.005:
                results["approaching_pdl"].append({
                    "ticker":   t,
                    "close":    round(today_close, 2),
                    "pdl":      round(prev_low, 2),
                    "pct_away": round((today_close - prev_low) / prev_low * 100, 2),
                })

            # ── High volume: institutional-size activity ───────────────────────
            if vol_ratio >= 2.0:
                results["high_volume"].append({
                    "ticker":    t,
                    "close":     round(today_close, 2),
                    "vol_ratio": round(vol_ratio, 1),
                    "direction": "UP" if today_close >= prev_close else "DOWN",
                    "pct_chg":   round((today_close - prev_close) / prev_close * 100, 2),
                })

            # ── Gap fill: gapped >1.5 % but closed within 0.5 % of prev close ─
            gap_pct   = (today_open - prev_close) / prev_close * 100
            close_chg = (today_close - prev_close) / prev_close * 100
            if abs(gap_pct) >= 1.5 and abs(close_chg) <= 0.5:
                results["gap_fills"].append({
                    "ticker":   t,
                    "close":    round(today_close, 2),
                    "gap_pct":  round(gap_pct, 2),
                    "gap_dir":  "UP" if gap_pct > 0 else "DOWN",
                })

        except Exception:
            continue

    # Sort each list for most useful ordering
    results["breakouts"].sort(     key=lambda x: x["vol_ratio"],  reverse=True)
    results["breakdowns"].sort(    key=lambda x: x["vol_ratio"],  reverse=True)
    results["approaching_pdh"].sort(key=lambda x: x["pct_away"])
    results["approaching_pdl"].sort(key=lambda x: x["pct_away"])
    results["high_volume"].sort(   key=lambda x: x["vol_ratio"],  reverse=True)
    results["gap_fills"].sort(     key=lambda x: abs(x["gap_pct"]), reverse=True)

    total = sum(len(v) for v in results.values())
    logger.info(
        f"Post-market setups: {len(results['breakouts'])} breakouts, "
        f"{len(results['breakdowns'])} breakdowns, "
        f"{len(results['approaching_pdh'])} near PDH, "
        f"{len(results['approaching_pdl'])} near PDL, "
        f"{len(results['high_volume'])} high-vol, "
        f"{len(results['gap_fills'])} gap-fills"
    )
    return results
