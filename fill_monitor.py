"""
fill_monitor.py — Polls Alpaca every 5 min for order fills and position changes.
Also tracks realized daily P&L and records completed trades to the brain.
"""
import logging
from datetime import datetime

import pytz

from alpaca_trader import (
    get_recent_orders, get_account, close_all_positions,
    get_open_positions, move_stop_to_breakeven,
)
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
# Prevent brain from double-recording a trade when both +1R and +2R legs close
_brain_recorded: set[str] = set()   # "TICKER_DATE" keys


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
    _pending_entries.clear()   # stale yesterday entries → wrong P&L if ticker re-trades
    _brain_recorded.clear()    # fresh dedup set for new day


def _now_et() -> str:
    return datetime.now(_ET).strftime("%I:%M %p ET")


def check_fills(send_fn, signals_list=None) -> None:
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
            pnl, actual_fill_px = _handle_fill(order, send_fn)

            # Write actual fill price / realized P&L back to in-memory signals list
            if signals_list is not None:
                for sig in signals_list:
                    if sig["ticker"] == order.symbol:
                        if actual_fill_px > 0 and pnl is None:   # entry fill → update fill_px
                            sig["fill_px"] = actual_fill_px
                        if pnl is not None:                        # closing fill → accumulate pnl
                            sig["pnl"] = sig.get("pnl", 0) + pnl
                        break

            if pnl is not None:
                _realized_pnl += pnl
                _brain.update_daily_pnl(_realized_pnl)

                # Record to brain — once per ticker per day to prevent 2-leg winners
                # from registering as two separate WIN records (inflates win rate).
                from datetime import date as _date
                for entry_oid, ctx in list(_pending_entries.items()):
                    if ctx["ticker"] == order.symbol and pnl != 0:
                        brain_key = f"{ctx['ticker']}_{_date.today().isoformat()}"
                        if brain_key not in _brain_recorded:
                            result = "WIN" if pnl > 0 else ("LOSS" if pnl < -10 else "SCRATCH")
                            _brain.record_trade(
                                ticker=ctx["ticker"],
                                signal_type=ctx["signal_type"],
                                grade=ctx["grade"],
                                direction=ctx["direction"],
                                pnl=pnl,
                                result=result,
                            )
                            _brain_recorded.add(brain_key)
                        break

        elif status in ("cancelled", "expired", "rejected"):
            _alerted_orders.add(oid)
            _handle_cancel(order, send_fn)


def _handle_fill(order, send_fn) -> tuple[float | None, float]:
    """Send fill alert; return (realized_pnl, fill_px). pnl is None for entry fills."""
    ticker  = order.symbol
    side    = str(order.side).upper()
    qty     = int(float(order.filled_qty))
    fill_px = float(order.filled_avg_price or 0)
    now     = _now_et()
    cid     = str(order.client_order_id or "")
    oid     = str(order.id)

    # Our entry orders carry custom cids: "{ticker}_{dir}_{type}_scale-out" or "_close-all"
    # Bracket child orders (stop-loss and take-profit) get auto-generated UUID cids.
    if "scale-out" in cid or "close-all" in cid:
        # This is an ENTRY fill — position is now open.
        # Update stored entry price from signal price → actual fill price so P&L is accurate.
        if oid in _pending_entries:
            _pending_entries[oid]["entry"] = fill_px
        leg = "1st half" if "scale-out" in cid else "2nd half"
        msg = (
            f"ENTRY FILLED ({leg})\n\n"
            f"  {ticker}  {side}  {qty} shares @ ${fill_px:.2f}\n"
            f"  Order: {oid[:8]}...\n\n"
            f"{now}"
        )
        logger.info(f"Entry fill ({leg}): {ticker} {side} {qty} @ {fill_px}")
        send_fn(msg)
        return None, fill_px

    # UUID cid → bracket child (stop-loss or take-profit exit)
    entry_ctx = None
    for ctx in _pending_entries.values():
        if ctx["ticker"] == ticker:
            entry_ctx = ctx
            break

    if entry_ctx is None:
        # No context (e.g. manual order) — generic alert
        send_fn(f"FILL\n\n  {ticker}  {side}  {qty} @ ${fill_px:.2f}\n\n{now}")
        return None, fill_px

    direction = entry_ctx["direction"]
    entry     = entry_ctx["entry"]

    if direction == "BUY":
        realized_pnl = (fill_px - entry) * qty
        is_profit    = fill_px > entry * 1.001   # allow tiny slippage below entry
    else:
        realized_pnl = (entry - fill_px) * qty
        is_profit    = fill_px < entry * 0.999

    new_total = _realized_pnl + realized_pnl

    if is_profit:
        # Take-profit hit — move remaining half's stop to breakeven so it rides free
        moved    = move_stop_to_breakeven(ticker, direction, entry)
        be_note  = f"\n  Stop → breakeven ${entry:.2f} (remaining half rides free)" if moved else ""
        tgt_note = "\n\n  DAILY TARGET HIT — Grade A only from now." if new_total >= DAILY_PROFIT_TARGET else ""

        msg = (
            f"PROFIT EXIT\n\n"
            f"  {ticker}  {side}  {qty} shares @ ${fill_px:.2f}\n"
            f"  P&L: ${realized_pnl:+,.0f}  |  Day total: ${new_total:+,.0f}"
            f"{be_note}{tgt_note}\n\n"
            f"{now}"
        )
        logger.info(f"Profit exit: {ticker} {side} {qty} @ {fill_px}  pnl={realized_pnl:+.0f}  be_moved={moved}")
    else:
        # Stop-loss hit
        msg = (
            f"STOP HIT\n\n"
            f"  {ticker}  {side}  {qty} shares @ ${fill_px:.2f}\n"
            f"  P&L: ${realized_pnl:+,.0f}  |  Day total: ${new_total:+,.0f}\n\n"
            f"{now}"
        )
        logger.info(f"Stop hit: {ticker} {side} {qty} @ {fill_px}  pnl={realized_pnl:+.0f}")

    send_fn(msg)
    return realized_pnl, fill_px


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
