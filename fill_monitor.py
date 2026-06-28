"""
fill_monitor.py — Polls Alpaca every 5 min for order fills and position changes.

Sends WhatsApp alerts when:
  - A BUY/SELL order is filled (entry confirmed)
  - A take-profit fires (scale-out at +1R or close at +2R)
  - A stop-loss fires (loss)
  - A daily P&L threshold is breached
"""
import logging
import time
from datetime import datetime

import pytz

from alpaca_trader import get_recent_orders, get_account, close_all_positions, get_open_positions
from config import AUTO_MAX_DAILY_LOSS, TIMEZONE

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

# Track which order IDs we've already alerted on
_alerted_orders: set[str] = set()


def _now_et() -> str:
    return datetime.now(_ET).strftime("%I:%M %p ET")


def check_fills(send_fn) -> None:
    """
    Check Alpaca for newly filled/closed orders.
    send_fn: callable that takes a str and sends a WhatsApp message.
    """
    global _alerted_orders

    orders = get_recent_orders(status="all", limit=100)
    account = get_account()

    # ── Hard loss limit check ─────────────────────────────────────────────────
    if account:
        day_pnl = account.get("day_pnl", 0)
        if day_pnl <= -AUTO_MAX_DAILY_LOSS:
            logger.warning(f"Hard loss limit hit: day P&L ${day_pnl:,.0f}")
            send_fn(
                f"🚨 HARD LOSS LIMIT HIT\n"
                f"\n"
                f"Day P&L: ${day_pnl:,.0f}  (limit: -${AUTO_MAX_DAILY_LOSS:,.0f})\n"
                f"Closing ALL positions and cancelling ALL orders now.\n"
                f"\n"
                f"⏰ {_now_et()}"
            )
            close_all_positions()
            return

    for order in orders:
        oid = str(order.id)
        if oid in _alerted_orders:
            continue

        status = str(order.status)

        if status == "filled":
            _alerted_orders.add(oid)
            _handle_fill(order, send_fn)

        elif status in ("cancelled", "expired", "rejected"):
            _alerted_orders.add(oid)
            _handle_cancel(order, send_fn)


def _handle_fill(order, send_fn) -> None:
    """Format and send a fill alert."""
    ticker    = order.symbol
    side      = str(order.side).upper()
    qty       = int(float(order.filled_qty))
    fill_px   = float(order.filled_avg_price or 0)
    now       = _now_et()

    # Determine leg type from client_order_id tag
    cid = str(order.client_order_id or "")
    if "scale-out" in cid:
        leg_label = "📤 +1R SCALE-OUT FILLED"
        emoji = "💰"
    elif "close-all" in cid:
        leg_label = "🏁 +2R CLOSE-ALL FILLED"
        emoji = "✅"
    elif "stop-loss" in cid.lower() or (order.order_class and "bracket" in str(order.order_class).lower()):
        # Could be stop triggered — check side vs original
        leg_label = "📥 ENTRY FILLED"
        emoji = "📋"
    else:
        leg_label = "📋 ORDER FILLED"
        emoji = "📋"

    # Try to get P&L from legs if it's a closing order
    pnl_str = ""
    if hasattr(order, "legs") and order.legs:
        for leg in order.legs:
            if getattr(leg, "filled_avg_price", None):
                pass  # Alpaca legs don't expose P&L directly

    msg = (
        f"{emoji} {leg_label}\n"
        f"\n"
        f"  {ticker}  {side}  {qty} shares\n"
        f"  Fill price: ${fill_px:.2f}\n"
        f"  Order ID:   {str(order.id)[:8]}...\n"
        f"\n"
        f"⏰ {now}"
    )

    logger.info(f"Fill alert: {ticker} {side} {qty} @ {fill_px}")
    send_fn(msg)


def _handle_cancel(order, send_fn) -> None:
    """Alert on cancelled/rejected orders."""
    ticker = order.symbol
    side   = str(order.side).upper()
    qty    = int(float(order.qty or 0))
    reason = str(order.status)
    now    = _now_et()

    # Don't alert on routine EOD cancellations (too noisy)
    cid = str(order.client_order_id or "")
    if "close-all" in cid or "scale-out" in cid:
        return   # part of a bracket that already resolved

    msg = (
        f"⚠️ ORDER {reason.upper()}\n"
        f"\n"
        f"  {ticker}  {side}  {qty} shares\n"
        f"  Order ID: {str(order.id)[:8]}...\n"
        f"\n"
        f"⏰ {now}"
    )

    logger.info(f"Cancel alert: {ticker} {side} — {reason}")
    send_fn(msg)


def get_positions_summary() -> str:
    """Format current open positions for EOD/status messages."""
    positions = get_open_positions()
    if not positions:
        return "No open positions."

    lines = [f"Open positions ({len(positions)}):"]
    for p in positions:
        side    = "LONG" if float(p.qty) > 0 else "SHORT"
        qty     = abs(int(float(p.qty)))
        cost    = float(p.avg_entry_price)
        current = float(p.current_price)
        upnl    = float(p.unrealized_pl)
        pct     = float(p.unrealized_plpc) * 100
        lines.append(
            f"  {p.symbol:<6} {side} {qty} @ ${cost:.2f}"
            f"  now ${current:.2f}  P&L ${upnl:+,.0f} ({pct:+.1f}%)"
        )
    return "\n".join(lines)
