"""
alpaca_trader.py — Paper trading execution via Alpaca API.

Split-bracket strategy:
  Order 1 (50% qty): stop at -1R,  take-profit at +1R  → scale-out
  Order 2 (50% qty): stop at -1R,  take-profit at +2R  → close all

Both orders are native Alpaca bracket orders so stops/targets live at the
broker — the bot does not need to manage them.
"""
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

logger = logging.getLogger(__name__)


def _client() -> TradingClient:
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def get_account() -> dict:
    """Return equity, cash, buying_power, today's P&L."""
    try:
        a = _client().get_account()
        return {
            "equity":        float(a.equity),
            "cash":          float(a.cash),
            "buying_power":  float(a.buying_power),
            "day_pnl":       float(a.equity) - float(a.last_equity),
            "day_pnl_pct":   (float(a.equity) - float(a.last_equity)) / float(a.last_equity) * 100,
        }
    except Exception as e:
        logger.error(f"Account fetch failed: {e}")
        return {}


def place_bracket_orders(
    ticker: str,
    direction: str,
    units: int,
    stop: float,
    r1_price: float,
    r2_price: float,
    tag: str = "",
) -> list:
    """
    Place two bracket orders to implement 2-stage scale-out.
      Half position exits at +1R (r1_price).
      Remaining half exits at +2R (r2_price).
    Stop-loss is the same for both.

    Returns list of submitted orders (may be empty on failure).
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca keys not configured — order skipped")
        return []

    client = _client()
    side   = OrderSide.BUY if direction == "BUY" else OrderSide.SELL
    half   = max(1, units // 2)
    rest   = units - half

    submitted = []
    pairs = [(half, r1_price, "scale-out"), (rest, r2_price, "close-all")]

    for qty, tp_price, leg in pairs:
        if qty < 1:
            continue
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop, 2)),
                client_order_id=f"{tag}_{leg}"[:48] if tag else None,
            )
            order = client.submit_order(req)
            submitted.append(order)
            logger.info(
                f"Order placed [{leg}]: {direction} {qty} {ticker} "
                f"stop=${stop:.2f} tp=${tp_price:.2f}  id={order.id}"
            )
        except Exception as e:
            logger.error(f"Order failed [{leg}] for {ticker}: {e}")

    return submitted


def get_recent_orders(status: str = "all", limit: int = 50) -> list:
    """Fetch recent orders — used by fill_monitor."""
    try:
        client = _client()
        req = GetOrdersRequest(
            status=QueryOrderStatus(status),
            limit=limit,
        )
        return client.get_orders(filter=req)
    except Exception as e:
        logger.warning(f"Get orders failed: {e}")
        return []


def get_open_positions() -> list:
    """Return all currently open positions."""
    try:
        return _client().get_all_positions()
    except Exception as e:
        logger.warning(f"Get positions failed: {e}")
        return []


def cancel_all_orders() -> None:
    """Cancel all open (unfilled) orders — called at EOD."""
    try:
        _client().cancel_orders()
        logger.info("All open orders cancelled at EOD")
    except Exception as e:
        logger.warning(f"Cancel all orders failed: {e}")


def close_all_positions() -> None:
    """Emergency: flatten everything. Called only on hard loss limit breach."""
    try:
        _client().close_all_positions(cancel_orders=True)
        logger.warning("ALL POSITIONS CLOSED — hard loss limit triggered")
    except Exception as e:
        logger.error(f"Close all positions failed: {e}")
