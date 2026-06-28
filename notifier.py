"""
notifier.py — WhatsApp alerts for the Alpaca paper bot.

Sends two types of messages:
  1. Signal alert  : same format as whatsapp_signal_bot but with "AUTO-TRADE" header
  2. Fill/EOD alert: trade confirmations, daily P&L from Alpaca
"""
import logging
from datetime import datetime

import pytz
from twilio.rest import Client

from config import (
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM, WHATSAPP_TO, ACCOUNT_SIZE, TIMEZONE,
)

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)


def _now_et() -> str:
    return datetime.now(_ET).strftime("%I:%M %p ET")


def _send(body: str) -> bool:
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=WHATSAPP_TO,
            body=body,
        )
        logger.info(f"WhatsApp sent: {msg.sid}")
        return True
    except Exception as e:
        logger.error(f"WhatsApp send failed: {e}")
        return False


def send_alert(text: str) -> bool:
    return _send(text)


def send_signal_alert(
    ticker: str,
    direction: str,
    grade: str,
    entry: float,
    stop: float,
    r1_price: float,
    r2_price: float,
    units: int,
    risk_amount: float,
    target_pnl: float,
    reasons: list[str],
    confidence: int,
    signal_type: str = "ORB",
    spy_pct: float | None = None,
    regime: str = "",
) -> bool:
    """Signal + auto-order notification sent before the order is placed."""
    now   = _now_et()
    emoji = "🟢" if direction == "BUY" else "🔴"
    grade_map = {
        "A": "🥇 Grade A (1.0% risk)",
        "B": "🥈 Grade B (0.75% risk)",
        "C": "🥉 Grade C (0.5% risk)",
    }
    pct_risk   = abs(entry - stop) / entry * 100
    pct_reward = abs(r2_price - entry) / entry * 100

    lines = [
        f"🤖 AUTO-TRADE PLACING: {ticker} {direction} {emoji}",
        f"[{grade_map.get(grade, grade)}  |  {signal_type}]",
        "",
        f"💰 Entry:    ${entry:.2f}   (market order)",
        f"🛑 Stop:     ${stop:.2f}   (−{pct_risk:.1f}%)",
        f"📤 +1R:      ${r1_price:.2f}  → sell 50%",
        f"🎯 +2R:      ${r2_price:.2f}  → close all  (+{pct_reward:.1f}%)",
        "",
        f"📦 {units} shares  |  Risk: −${risk_amount:,.0f}  |  Target: +${target_pnl:,.0f}",
    ]

    if spy_pct is not None and regime:
        lines += ["", f"📈 SPY {spy_pct:+.1f}%  |  Regime: {regime}"]

    lines += ["", "Why:"]
    for r in reasons[:5]:
        lines.append(f"  ✅ {r}")

    lines += ["", f"Confidence: {confidence}%  |  {now}"]
    return _send("\n".join(lines))


def send_eod_summary(account: dict, trades_today: list[dict]) -> bool:
    """EOD: Alpaca account P&L + trade log for the day."""
    now      = _now_et()
    day_pnl  = account.get("day_pnl", 0)
    equity   = account.get("equity", ACCOUNT_SIZE)
    pct      = account.get("day_pnl_pct", 0)
    emoji    = "✅" if day_pnl >= 0 else "❌"

    lines = [
        f"📊 ALPACA PAPER — EOD SUMMARY",
        "",
        f"{emoji} Day P&L:  ${day_pnl:+,.0f}  ({pct:+.2f}%)",
        f"   Equity: ${equity:,.0f}",
        "",
    ]

    if trades_today:
        lines.append(f"Trades today ({len(trades_today)}):")
        for t in trades_today:
            outcome_emoji = "✅" if t.get("pnl", 0) >= 0 else "❌"
            lines.append(
                f"  {outcome_emoji} {t['ticker']} {t['direction']}"
                f"  ${t.get('fill_px', 0):.2f}  P&L ${t.get('pnl', 0):+,.0f}"
            )
    else:
        lines.append("No trades executed today.")

    lines += ["", f"⏰ {now}"]
    return _send("\n".join(lines))


def send_startup_message() -> bool:
    now  = datetime.now(_ET).strftime("%Y-%m-%d %I:%M %p ET")
    mode = "PAPER" if True else "LIVE"  # always paper for now
    return _send(
        f"🤖 Alpaca Paper Bot started  [{mode}]\n"
        f"⏰ {now}\n"
        f"\n"
        f"Strategy: ORB + Z-Score (same signals as WhatsApp bot)\n"
        f"Execution: Auto bracket orders via Alpaca paper API\n"
        f"\n"
        f"Scale-out: +1R sell 50%  •  +2R close all\n"
        f"Daily loss limit: −${1000:,}\n"
        f"\n"
        f"Signals fire 9:30 AM – 4:00 PM ET\n"
        f"Fill alerts sent on every order event."
    )
