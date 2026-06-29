"""
fill_monitor.py — Polls Alpaca every 5 min for order fills and position changes.
Also tracks realized daily P&L and records completed trades to the brain.
"""
import logging
from datetime import datetime

import pytz

from alpaca_trader import get_recent_orders, get_account, close_all_positions, get_open_positions
from config import AUTO_MAX_DAILY_LOSS, DAILY_PROFIT_TARGET, TIMEZONE
import brain as _brain

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

_alerted_orders: set[str] = set()
_halt_notified:  bool     = False   # prevents duplicate hard-loss alerts

# Realized P&L accumulated today from closed orders (entry fills excluded)
_realized_pnl: float = 0.0
# Map order_id → (ticker, signal_type, grade, direction, entry_price, qty)
_pending_entries: dict[str, dict] = {}


def register_entry(order_id: str, ticker: str, signal_type: str,
                   grade: str, direction: str, entry: float, qty: int):
    """Call from main._execute_signal() so we can compute P&L on close."""
    _pending_entries[order_id] = dict(
        ticker=ticker, signal_type=signal_type, grade=grade,
        direction=direction, entry=entry, qty=qty,
    )


def get_daily_pnl() -> float:
    return _realized_pnl


def reset_daily_pnl():
    global _realized_pnl, _halt_notified
    _realized_pnl   = 0.0
    _halt_notified  = False
    _alerted_orders.clear()


def _now_et() -> str:
    return datetime.now(_ET).strftime("%I:%M %p ET")


def check_fills(send_fn) -> None:
    global _realized_pnl

    orders  = get_recent_orders(status="all", limit=150)
    account = get_account()

    # ── Daily loss limit check ────────────────────────────────────────────────
    global _halt_notified
    if account:
        day_pnl = account.get("day_pnl", 0)
        if day_pnl <= -AUTO_MAX_DAILY_LOSS:
            if not _halt_notified:
                _halt_notified = True
                open_pos = get_open_positions()
                if open_pos:
                    logger.warning(f"Hard loss limit: day P&L ${day_pnl:,.0f}")
                    send_fn(
                        f"HARD LOSS LIMIT HIT\n\n"
                        f"Day P&L: ${day_pnl:,.0f}  (limit: -${AUTO_MAX_DAILY_LOSS:,.0f})\n"
                        f"Closing ALL positions now.\n\n"
                        f"{_now_et()}"
                    )
                    close_all_positions()
                else:
                    logger.info(f"Hard loss limit: day P&L ${day_pnl:,.0f} — already closed, silencing")
            return

    for order in orders:
        oid    = str(order.id)
        status = str(order.status)

        if oid in _alerted_orders:
            continue

        if status == "filled":
            _alerted_orders.add(oid)
            pnl = _handle_fill(order, send_fn)
            if pnl is not None:
                _realized_pnl += pnl
                _brain.update_daily_pnl(_realized_pnl)

                # Record to brain if we have entry context
                cid = str(order.client_order_id or "")
                for entry_oid, ctx in list(_pending_entries.items()):
                    if ctx["ticker"] == order.symbol and pnl != 0:
                        result = "WIN" if pnl > 0 else ("LOSS" if pnl < -10 else "SCRATCH")
                        _brain.record_trade(
                            ticker=ctx["ticker"],
                            signal_type=ctx["signal_type"],
                            grade=ctx["grade"],
                            direction=ctx["direction"],
                            pnl=pnl,
                            result=result,
                        )
                        break

        elif status in ("cancelled", "expired", "rejected"):
            _alerted_orders.add(oid)
            _handle_cancel(order, send_fn)


def _handle_fill(order, send_fn) -> float | None:
    """Send WhatsApp fill alert; return realized P&L for closing legs (None for entries)."""
    ticker  = order.symbol
    side    = str(order.side).upper()
    qty     = int(float(order.filled_qty))
    fill_px = float(order.filled_avg_price or 0)
    now     = _now_et()
    cid     = str(order.client_order_id or "")
    oid     = str(order.id)

    realized_pnl = None

    if "scale-out" in cid or "close-all" in cid:
        # Closing leg — compute P&L against known entry
        label = "+1R SCALE-OUT FILLED" if "scale-out" in cid else "+2R CLOSE-ALL FILLED"
        emoji = "💰" if "scale-out" in cid else "✅"

        # Look up entry price
        for ctx in _pending_entries.values():
            if ctx["ticker"] == ticker:
                if ctx["direction"] == "BUY":
                    realized_pnl = (fill_px - ctx["entry"]) * qty
                else:
                    realized_pnl = (ctx["entry"] - fill_px) * qty
                break

        pnl_str = f"\n  P&L: ${realized_pnl:+,.0f}" if realized_pnl is not None else ""
        daily_str = f"\n  Day total: ${_realized_pnl + (realized_pnl or 0):+,.0f}"
        target_note = ""
        new_total = _realized_pnl + (realized_pnl or 0)
        if new_total >= DAILY_PROFIT_TARGET:
            target_note = "\n\n  TARGET HIT — Grade A signals only from now."

        msg = (
            f"{emoji} {label}\n\n"
            f"  {ticker}  {side}  {qty} shares @ ${fill_px:.2f}"
            f"{pnl_str}{daily_str}{target_note}\n\n"
            f"{now}"
        )
    else:
        # Entry fill
        label = "ENTRY FILLED"
        msg = (
            f"ENTRY FILLED\n\n"
            f"  {ticker}  {side}  {qty} shares @ ${fill_px:.2f}\n"
            f"  Order: {oid[:8]}...\n\n"
            f"{now}"
        )

    logger.info(f"Fill: {ticker} {side} {qty} @ {fill_px}  pnl={realized_pnl}")
    send_fn(msg)
    return realized_pnl


def _handle_cancel(order, send_fn) -> None:
    ticker = order.symbol
    side   = str(order.side).upper()
    qty    = int(float(order.qty or 0))
    reason = str(order.status)
    cid    = str(order.client_order_id or "")

    if "close-all" in cid or "scale-out" in cid:
        return  # routine bracket cleanup

    send_fn(
        f"ORDER {reason.upper()}\n\n"
        f"  {ticker}  {side}  {qty} shares\n\n"
        f"{_now_et()}"
    )
    logger.info(f"Cancel: {ticker} {side} — {reason}")


def get_positions_summary() -> str:
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
