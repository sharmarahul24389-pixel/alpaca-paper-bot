"""
monitor.py — Telegram /status command + morning briefing for Alpaca bot.

Commands (type in your Telegram chat with the bot):
  /status    — full snapshot: positions, P&L, open orders, brain, regime
  /positions — open positions only
  /orders    — open orders only
  /pnl       — today's P&L summary
"""
import logging
import threading
import time as _time

import requests
import pytz
from datetime import datetime

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE, ACCOUNT_SIZE

logger = logging.getLogger(__name__)
_ET   = pytz.timezone(TIMEZONE)
_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Last Telegram update_id we processed — persists across polls in-process
_last_update_id: int = 0


# ── Telegram helpers ──────────────────────────────────────────────────────────

def _send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"{_BASE}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Monitor send failed: {e}")


def _get_updates(offset: int) -> list:
    try:
        resp = requests.get(
            f"{_BASE}/getUpdates",
            params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
            timeout=40,
        )
        data = resp.json()
        return data.get("result", []) if data.get("ok") else []
    except Exception as e:
        logger.warning(f"getUpdates failed: {e}")
        return []


# ── Status builders ───────────────────────────────────────────────────────────

def build_status() -> str:
    lines = [f"<b>BOT STATUS</b>  {datetime.now(_ET).strftime('%H:%M ET %b %d')}\n"]

    # Account + positions
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        import os
        client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
        acct = client.get_account()
        equity = float(acct.equity)
        bp     = float(acct.buying_power)
        gain   = equity - ACCOUNT_SIZE
        lines.append(f"<b>Account</b>")
        lines.append(f"  Equity:  ${equity:,.0f}  ({gain:+,.0f} vs ${ACCOUNT_SIZE:,.0f} start)")
        lines.append(f"  BP:      ${bp:,.0f}")

        positions = client.get_all_positions()
        if positions:
            lines.append(f"\n<b>Open Positions ({len(positions)})</b>")
            for p in positions:
                pnl = float(p.unrealized_pl or 0)
                pnl_pct = float(p.unrealized_plpc or 0) * 100
                lines.append(f"  {p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f}  PnL: ${pnl:+,.0f} ({pnl_pct:+.1f}%)")
        else:
            lines.append("\n<b>Positions:</b> None")

        open_orders = client.get_orders()
        if open_orders:
            lines.append(f"\n<b>Open Orders ({len(open_orders)})</b>")
            for o in open_orders:
                lines.append(f"  {o.symbol} {o.side.value} {o.qty} {o.type.value} [{o.status.value}]")
        else:
            lines.append("\n<b>Orders:</b> None pending")

    except Exception as e:
        lines.append(f"\n[Alpaca error: {e}]")

    # Brain + regime
    try:
        import brain as _brain
        params = _brain.get_params()
        regime = _brain.get_regime() if hasattr(_brain, "get_regime") else "—"
        min_conf = params.get("min_confidence_orb", "—")
        size_mult = params.get("position_size_mult", 1.0)
        lines.append(f"\n<b>Brain</b>")
        lines.append(f"  Regime:     {regime}")
        lines.append(f"  Min conf:   {min_conf}%")
        lines.append(f"  Size mult:  {size_mult}x")
        skip = params.get("skip_types", [])
        if skip:
            lines.append(f"  Paused:     {', '.join(skip)}")
    except Exception as e:
        lines.append(f"\n[Brain error: {e}]")

    # Today's P&L from fill monitor
    try:
        from fill_monitor import get_daily_pnl
        pnl = get_daily_pnl()
        lines.append(f"\n<b>Today P&L:</b> ${pnl:+,.2f}")
    except Exception:
        pass

    # Today's signals
    try:
        from main import _signals_today
        if _signals_today:
            lines.append(f"\n<b>Signals Today ({len(_signals_today)})</b>")
            for s in _signals_today:
                lines.append(f"  {s['ticker']} {s['direction']} {s['grade']}  entry=${s['entry']:.2f}")
        else:
            lines.append("\n<b>Signals Today:</b> None yet")
    except Exception:
        pass

    lines.append(f"\n<i>Scan window: 9:30–12:30 PM ET | EOD: 3:55 PM</i>")
    return "\n".join(lines)


def build_morning_briefing() -> str:
    lines = [f"<b>MORNING BRIEFING</b>  {datetime.now(_ET).strftime('%A %b %d')}\n"]

    try:
        from alpaca.trading.client import TradingClient
        import os
        client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
        acct = client.get_account()
        equity = float(acct.equity)
        gain   = equity - ACCOUNT_SIZE
        lines.append(f"Account: ${equity:,.0f}  ({gain:+,.0f} total)")

        positions = client.get_all_positions()
        if positions:
            lines.append(f"\nCarried positions — SHOULD BE NONE:")
            for p in positions:
                pnl = float(p.unrealized_pl or 0)
                lines.append(f"  {p.symbol}: {p.qty} shares  PnL: ${pnl:+,.0f}")
        else:
            lines.append("No carried positions — clean start")
    except Exception as e:
        lines.append(f"[Alpaca error: {e}]")

    try:
        import brain as _brain
        from trend_filter import get_market_regime
        regime = get_market_regime()
        params = _brain.get_params()
        lines.append(f"\nRegime: {regime}")
        lines.append(f"Min confidence: {params.get('min_confidence_orb', '?')}%")
        lines.append(f"Size mult: {params.get('position_size_mult', 1.0)}x")
    except Exception:
        pass

    lines.append(f"\nFirst scan: 9:30 AM | Last: 12:30 PM | EOD: 3:55 PM")
    lines.append(f"Type /status anytime for live snapshot")
    return "\n".join(lines)


# ── Command dispatcher ────────────────────────────────────────────────────────

def _handle_command(text: str) -> None:
    cmd = text.strip().lower().split()[0]
    if cmd == "/status":
        _send(build_status())
    elif cmd == "/positions":
        try:
            from alpaca.trading.client import TradingClient
            import os
            client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
            positions = client.get_all_positions()
            if not positions:
                _send("No open positions.")
                return
            lines = ["<b>Open Positions</b>"]
            for p in positions:
                pnl = float(p.unrealized_pl or 0)
                lines.append(f"  {p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f}  ${pnl:+,.0f}")
            _send("\n".join(lines))
        except Exception as e:
            _send(f"Error: {e}")
    elif cmd == "/orders":
        try:
            from alpaca.trading.client import TradingClient
            import os
            client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
            orders = client.get_orders()
            if not orders:
                _send("No open orders.")
                return
            lines = [f"<b>Open Orders ({len(orders)})</b>"]
            for o in orders:
                lines.append(f"  {o.symbol} {o.side.value} {o.qty} {o.type.value}")
            _send("\n".join(lines))
        except Exception as e:
            _send(f"Error: {e}")
    elif cmd == "/pnl":
        try:
            from fill_monitor import get_daily_pnl
            _send(f"Today P&L: ${get_daily_pnl():+,.2f}")
        except Exception as e:
            _send(f"Error: {e}")
    elif cmd in ("/help", "/start"):
        _send(
            "<b>Alpaca Bot Commands</b>\n"
            "/status    — full snapshot\n"
            "/positions — open positions\n"
            "/orders    — pending orders\n"
            "/pnl       — today's P&amp;L"
        )


# ── Background polling thread ─────────────────────────────────────────────────

def _poll_loop() -> None:
    global _last_update_id
    logger.info("Telegram command poller started")
    while True:
        try:
            updates = _get_updates(_last_update_id + 1)
            for upd in updates:
                _last_update_id = upd["update_id"]
                msg = upd.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                # Only respond to the configured chat
                if text.startswith("/") and chat_id == str(TELEGRAM_CHAT_ID):
                    logger.info(f"Command received: {text}")
                    try:
                        _handle_command(text)
                    except Exception as e:
                        logger.error(f"Command handler error: {e}")
        except Exception as e:
            logger.warning(f"Poll loop error: {e}")
            _time.sleep(5)


def start_command_poller() -> None:
    """Start background Telegram command listener. Call once at bot startup."""
    t = threading.Thread(target=_poll_loop, daemon=True, name="tg-command-poller")
    t.start()


def send_morning_briefing() -> None:
    """Send morning briefing — wire to 9:20 AM cron."""
    try:
        _send(build_morning_briefing())
    except Exception as e:
        logger.error(f"Morning briefing failed: {e}")
