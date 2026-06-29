"""
notifier.py — Telegram alerts for the Alpaca paper bot.

Sends via Telegram Bot API (no session windows, no Meta dependency).
Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import logging
from datetime import datetime

import pytz
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ACCOUNT_SIZE, TIMEZONE

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

_TG_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _now_et() -> str:
    return datetime.now(_ET).strftime("%I:%M %p ET")


def _send(body: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False
    try:
        resp = requests.post(
            _TG_URL.format(token=TELEGRAM_BOT_TOKEN),
            json={"chat_id": TELEGRAM_CHAT_ID, "text": body, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"Telegram sent: {resp.json().get('result', {}).get('message_id')}")
            return True
        logger.error(f"Telegram send failed: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
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
    now        = _now_et()
    emoji      = "🟢" if direction == "BUY" else "🔴"
    grade_map  = {"A": "🥇 Grade A (1.0%)", "B": "🥈 Grade B (0.75%)", "C": "🥉 Grade C (0.5%)"}
    pct_risk   = abs(entry - stop) / entry * 100
    pct_reward = abs(r2_price - entry) / entry * 100

    lines = [
        f"🤖 <b>AUTO-TRADE: {ticker} {direction} {emoji}</b>",
        f"<i>{grade_map.get(grade, grade)}  |  {signal_type}</i>",
        "",
        f"💰 Entry:  <b>${entry:.2f}</b>  (market)",
        f"🛑 Stop:   ${stop:.2f}  (−{pct_risk:.1f}%)",
        f"📤 +1R:    ${r1_price:.2f}  → sell 50%",
        f"🎯 +2R:    ${r2_price:.2f}  → close all  (+{pct_reward:.1f}%)",
        "",
        f"📦 {units} shares  |  Risk: −${risk_amount:,.0f}  |  Target: +${target_pnl:,.0f}",
    ]
    if spy_pct is not None and regime:
        lines += ["", f"📈 SPY {spy_pct:+.1f}%  |  Regime: {regime}"]
    lines += ["", "<b>Why:</b>"]
    for r in reasons[:5]:
        lines.append(f"  ✅ {r}")
    lines += ["", f"Confidence: {confidence}%  |  {now}"]
    return _send("\n".join(lines))


def send_eod_summary(account: dict, trades_today: list[dict]) -> bool:
    now     = _now_et()
    day_pnl = account.get("day_pnl", 0)
    equity  = account.get("equity", ACCOUNT_SIZE)
    pct     = account.get("day_pnl_pct", 0)
    emoji   = "✅" if day_pnl >= 0 else "❌"

    lines = [
        f"📊 <b>ALPACA PAPER — EOD SUMMARY</b>",
        "",
        f"{emoji} Day P&L:  <b>${day_pnl:+,.0f}</b>  ({pct:+.2f}%)",
        f"   Equity: ${equity:,.0f}",
        "",
    ]
    if trades_today:
        lines.append(f"Trades today ({len(trades_today)}):")
        for i, t in enumerate(trades_today, 1):
            pnl     = t.get("pnl", 0)
            fill_px = t.get("fill_px") or t.get("entry", 0)
            grade   = t.get("grade", "")
            stype   = t.get("signal_type", "ORB")
            stop    = t.get("stop", 0)
            e       = "✅" if pnl >= 0 else "❌"
            pnl_str = f"${pnl:+,.0f}" if pnl != 0 else "open/pending"
            lines.append(
                f"  {i}. {e} {t['ticker']} {t['direction']} [{stype}/Gr.{grade}]"
                f"  @${fill_px:.2f}  stop=${stop:.2f}  P&L {pnl_str}"
            )
    else:
        lines.append("No trades executed today.")
    lines += ["", f"⏰ {now}"]
    return _send("\n".join(lines))


def send_startup_message() -> bool:
    now = datetime.now(_ET).strftime("%Y-%m-%d %I:%M %p ET")
    return _send(
        f"🤖 <b>Alpaca Paper Bot started  [PAPER]</b>\n"
        f"⏰ {now}\n\n"
        f"Strategy: ORB + Z-Score (v3 Intelligence)\n"
        f"Execution: Auto bracket orders via Alpaca paper API\n\n"
        f"Scale-out: +1R sell 50%  •  +2R close all\n"
        f"Daily loss limit: −$1,000\n\n"
        f"Signals fire 9:30 AM – 4:00 PM ET\n"
        f"Fill alerts sent on every order event."
    )
