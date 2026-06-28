"""
Congress & Insider Trade Tracker

Sources:
  House  — housestockwatcher.com/api/transactions
  Senate — senatestockwatcher.com/api/transactions

Fires three alerts per new BUY or SELL disclosure:
  1. Disclosure alert  — who bought/sold what, amount, when disclosed
  2. Intraday signal   — BUY=long or SELL=short, 1.5% stop, 3% target
  3. 1-month view      — swing setup with 2xATR stop, +/-10% target
"""
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pytz
import requests
import urllib3
import yfinance as yf

logger = logging.getLogger(__name__)
_ET = pytz.timezone("America/New_York")

_HOUSE_API  = "https://housestockwatcher.com/api/transactions"
_SENATE_API = "https://senatestockwatcher.com/api/transactions"


def _fetch_url(url: str, timeout: int = 15) -> requests.Response | None:
    """
    Fetch URL with automatic DNS fallback via Google DoH.
    Railway's DNS sometimes can't resolve housestockwatcher.com /
    senatestockwatcher.com — DoH bypasses the broken resolver.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}
    try:
        return requests.get(url, timeout=timeout, headers=headers)
    except requests.exceptions.ConnectionError as exc:
        if "Failed to resolve" not in str(exc) and "NameResolution" not in str(exc):
            raise

    # Standard DNS failed — resolve via Google DoH then connect by IP
    hostname = urlparse(url).hostname
    try:
        doh = requests.get(
            "https://dns.google/resolve",
            params={"name": hostname, "type": "A"},
            timeout=5,
            headers=headers,
        )
        answers = doh.json().get("Answer", [])
        ip = next((a["data"] for a in answers if a.get("type") == 1), None)
        if not ip:
            logger.warning(f"DoH returned no A record for {hostname}")
            return None
        ip_url = url.replace(f"://{hostname}", f"://{ip}", 1)
        # verify=False because cert is bound to hostname not IP;
        # acceptable here — we're reading public government disclosure data
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(
            ip_url, timeout=timeout,
            headers={**headers, "Host": hostname},
            verify=False,
        )
        logger.info(f"DoH fallback success for {hostname} → {ip}  HTTP {resp.status_code}")
        return resp
    except Exception as exc2:
        logger.warning(f"DoH fallback failed for {hostname}: {exc2}")
        return None

# Members whose trades carry extra weight (Trump-aligned or historically accurate)
_TRUMP_ALIGNED = {
    "Tommy Tuberville", "Marjorie Taylor Greene", "Jim Jordan",
    "Matt Gaetz", "Dan Crenshaw", "Brian Babin", "Andy Biggs",
    "Paul Gosar", "Lauren Boebert", "Mike Johnson", "Chip Roy",
    "Scott Perry", "Mark Green", "Jeff Van Drew", "Troy Nehls",
}
_NOTABLE_TRADERS = _TRUMP_ALIGNED | {
    "Nancy Pelosi", "Austin Scott", "Daniel Meuser",
    "Ro Khanna", "Josh Gottheimer", "Greg Gianforte",
}

# Map disclosure amount strings to a priority tier (higher = more significant)
_AMOUNT_TIER = {
    "over $1,000,000":        5,
    "$500,001 - $1,000,000":  4,
    "$250,001 - $500,000":    3,
    "$100,001 - $250,000":    2,
    "$50,001 - $100,000":     1,
}


def _amount_priority(amount_str: str) -> int:
    low = amount_str.lower().strip()
    for k, v in _AMOUNT_TIER.items():
        if k in low:
            return v
    return 0


def get_congress_trades(since_hours: float = 48.0) -> list[dict]:
    """
    Fetch recent Congress stock disclosures (purchases AND sales).
    Returns a list of trade dicts sorted by priority.
    Skips non-actionable types: exchange, receive, gift.
    """
    cutoff = datetime.now(pytz.utc) - timedelta(hours=since_hours)
    trades: list[dict] = []

    sources = [
        (_HOUSE_API,  "House",  "representative"),
        (_SENATE_API, "Senate", "senator"),
    ]

    for url, chamber, name_field in sources:
        try:
            resp = _fetch_url(url, timeout=15)
            if resp is None:
                continue
            logger.info(f"Congress {chamber} API: HTTP {resp.status_code}")
            if resp.status_code != 200:
                continue

            items = resp.json()
            if not isinstance(items, list):
                items = items.get("transactions", []) if isinstance(items, dict) else []

            for item in items[:200]:
                raw_type   = str(item.get("type", "")).lower().strip()
                # Skip non-actionable disclosures
                if any(skip in raw_type for skip in ("exchange", "receive", "gift", "transfer")):
                    continue
                if not any(act in raw_type for act in ("purchase", "sale", "sell")):
                    continue
                # Normalise direction
                direction  = "BUY" if "purchase" in raw_type else "SELL"
                trade_type = raw_type

                # Parse disclosure date and apply cutoff
                disc_str = (
                    item.get("disclosure_date")
                    or item.get("disclosureDate")
                    or ""
                )
                if not disc_str:
                    continue
                try:
                    disc_dt = datetime.fromisoformat(
                        disc_str.replace("Z", "+00:00").replace(" ", "T")[:10]
                    )
                    disc_dt = pytz.utc.localize(disc_dt)
                    if disc_dt < cutoff:
                        continue
                except Exception:
                    continue

                ticker = str(item.get("ticker", "")).strip().upper()
                if not ticker or ticker in ("N/A", "--", ""):
                    continue

                member = (
                    item.get(name_field)
                    or item.get("representative")
                    or item.get("senator")
                    or item.get("name")
                    or "Unknown"
                ).strip()

                amount   = item.get("amount", "Unknown")
                tx_date  = (
                    item.get("transaction_date")
                    or item.get("transactionDate")
                    or disc_str[:10]
                )
                asset    = (
                    item.get("asset_description")
                    or item.get("assetDescription")
                    or ticker
                )
                party    = item.get("party", "")

                is_trump = any(t.lower() in member.lower() for t in _TRUMP_ALIGNED)
                notable  = any(n.lower() in member.lower() for n in _NOTABLE_TRADERS)

                trades.append({
                    "ticker":        ticker,
                    "member":        member,
                    "chamber":       chamber,
                    "party":         party,
                    "trade_type":    trade_type,
                    "direction":     direction,
                    "amount":        amount,
                    "amount_tier":   _amount_priority(amount),
                    "transaction_date": str(tx_date)[:10],
                    "disclosure_date":  disc_str[:10],
                    "asset":         asset,
                    "trump_aligned": is_trump,
                    "notable":       notable,
                })

        except Exception as exc:
            logger.warning(f"Congress tracker {chamber} failed: {exc}")

    # Sort: Trump-aligned first, then by amount tier, then by disclosure date
    trades.sort(key=lambda t: (not t["trump_aligned"], not t["notable"], -t["amount_tier"]))
    buys  = sum(1 for t in trades if t["direction"] == "BUY")
    sells = sum(1 for t in trades if t["direction"] == "SELL")
    logger.info(f"Congress tracker: {buys} buy(s), {sells} sell(s) in last {since_hours}h")
    return trades


def format_congress_alert(trade: dict) -> str:
    """Format the disclosure notification WhatsApp message."""
    flag      = "🇺🇸" if trade["trump_aligned"] else "🏛️"
    badge     = " [TRUMP ALLY]" if trade["trump_aligned"] else (
                " [NOTABLE]"    if trade["notable"]       else "")
    party_tag = f" ({trade['party'][0]})" if trade["party"] else ""
    direction = trade.get("direction", "BUY")
    action_emoji = "🟢 BOUGHT" if direction == "BUY" else "🔴 SOLD"
    raw_type  = trade.get("trade_type", "")
    type_note = " (partial)" if "partial" in raw_type else (" (full)" if "full" in raw_type else "")

    return "\n".join([
        f"{flag} CONGRESS TRADE DISCLOSED{badge}",
        "",
        f"  Who:    {trade['member']}{party_tag} — {trade['chamber']}",
        f"  Stock:  {trade['ticker']}  ({trade['asset']})",
        f"  Action: {action_emoji}{type_note}",
        f"  Amount: {trade['amount']}",
        f"  Traded: {trade['transaction_date']}",
        f"  Filed:  {trade['disclosure_date']}",
        "",
        "⚡ Intraday + 1-month signals follow below.",
    ])


def format_congress_intraday(trade: dict, price: float) -> str | None:
    """Quick intraday signal for the disclosed ticker."""
    now       = datetime.now(_ET).strftime("%I:%M %p ET")
    flag      = "🇺🇸" if trade["trump_aligned"] else "🏛️"
    direction = trade.get("direction", "BUY")
    member    = trade["member"]
    amount    = trade["amount"]

    if direction == "BUY":
        stop      = round(price * 0.985, 2)   # 1.5% below entry
        target    = round(price * 1.03,  2)   # 3% above entry → 2:1 RR
        dir_label = "BUY 🟢"
        stop_note = "-1.5%"
        tgt_note  = "+3.0%"
        catalyst  = f"{member} PURCHASED {amount}"
        tip       = "Intraday follow-through play — buy the insider conviction."
    else:
        stop      = round(price * 1.015, 2)   # 1.5% above entry (short stop)
        target    = round(price * 0.97,  2)   # 3% below entry → 2:1 RR
        dir_label = "SHORT 🔴"
        stop_note = "+1.5%"
        tgt_note  = "-3.0%"
        catalyst  = f"{member} SOLD {amount}"
        tip       = "Intraday fade — insider distribution signal. Cover before close."

    return "\n".join([
        f"{flag} CONGRESS INTRADAY — {trade['ticker']} {dir_label}",
        "",
        f"Entry:  ${price:.2f}  (market / limit)",
        f"Stop:   ${stop:.2f}   ({stop_note})",
        f"Target: ${target:.2f}  ({tgt_note})  →  1:2 R:R",
        "",
        f"Catalyst: {catalyst}",
        tip,
        f"Confidence: CONGRESS DISCLOSURE  |  {now}",
    ])


def format_congress_swing(trade: dict, price: float, atr: float) -> str:
    """1-month swing signal based on the Congress disclosure."""
    now       = datetime.now(_ET).strftime("%I:%M %p ET")
    flag      = "🇺🇸" if trade["trump_aligned"] else "🏛️"
    direction = trade.get("direction", "BUY")
    member    = trade["member"]
    chamber   = trade["chamber"]
    amount    = trade["amount"]
    tx_date   = trade["transaction_date"]

    if direction == "BUY":
        stop      = round(price - 2.0 * atr, 2)
        target    = round(price * 1.10, 2)
        pct_sl    = round((price - stop) / price * 100, 1)
        rr        = round((target - price) / max(price - stop, 0.01), 1)
        dir_label = "BUY 🟢"
        tgt_note  = "+10%"
        action    = "purchased"
        strategy  = "Congress insider-follow. Hold 3-4 weeks."
        exit_rule = "Exit: hit +10% OR break of 2× ATR stop"
    else:
        stop      = round(price + 2.0 * atr, 2)
        target    = round(price * 0.90, 2)
        pct_sl    = round((stop - price) / price * 100, 1)
        rr        = round((price - target) / max(stop - price, 0.01), 1)
        dir_label = "SHORT 🔴"
        tgt_note  = "-10%"
        action    = "sold"
        strategy  = "Congress insider-distribution fade. Hold 3-4 weeks."
        exit_rule = "Exit: hit -10% target OR break above 2× ATR stop"

    return "\n".join([
        f"{flag} CONGRESS 1-MONTH VIEW — {trade['ticker']} {dir_label}",
        "",
        f"  Entry:  ${price:.2f}",
        f"  Stop:   ${stop:.2f}   ({'+' if direction == 'SELL' else '-'}{pct_sl}%  |  2× ATR)",
        f"  Target: ${target:.2f}  ({tgt_note}  |  ~4 weeks)",
        f"  R:R     1:{rr}",
        "",
        f"Catalyst: {member} ({chamber}) {action}",
        f"  {amount} on {tx_date}",
        "",
        f"Strategy: {strategy}",
        "  Suggested sizing: 0.5% account risk ($500)",
        f"  {exit_rule}",
        "",
        f"Confidence: CONGRESS DISCLOSURE  |  {now}",
    ])


def get_price_and_atr(ticker: str) -> tuple[float, float] | None:
    """Fetch current price and daily ATR for a ticker. Returns None on failure."""
    try:
        df = yf.download(ticker, period="20d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 5:
            return None
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        price = float(df["Close"].iloc[-1])
        tr = (
            (df["High"] - df["Low"]).abs()
            .combine((df["High"] - df["Close"].shift()).abs(), max)
            .combine((df["Low"]  - df["Close"].shift()).abs(), max)
        )
        atr = float(tr.iloc[-14:].mean())
        return price, atr
    except Exception as exc:
        logger.warning(f"price/ATR fetch failed ({ticker}): {exc}")
        return None
