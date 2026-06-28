"""
main.py — Alpaca Paper Trading Bot

Reuses ALL signal generation logic from whatsapp_signal_bot (via sys.path).
Only NEW pieces: Alpaca order placement + fill monitoring.

Flow per signal:
  1. Generate signal  (same ORB/Quant/Swing code as existing bot)
  2. Send WhatsApp    (so you know what's happening)
  3. Place 2x bracket orders on Alpaca paper account
  4. Every 5 min: poll fills, send WhatsApp on exits/stops
  5. EOD: cancel unfilled orders, send P&L summary
"""
import gc
import logging
import os
import time
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from analyzer import analyze, get_spy_day_pct, get_days_to_earnings
from config import (                                  # pulls from THIS config.py
    TIMEZONE, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, TOP_MOVERS_COUNT,
    MIN_PRICE, MIN_VOLUME, MIN_CONFIDENCE, INTERVAL_MINUTES,
    MAX_SWING_SIGNALS, AUTO_MIN_GRADE, AUTO_MIN_CONFIDENCE,
    AUTO_MAX_SIGNALS, FILL_CHECK_INTERVAL,
)
from levels import get_pivot_levels
from options_flow import get_options_sentiment
from position_sizer import calculate_position
from quant_signals import batch_z_scores, generate_quant_signal, rank_momentum
from scanner import get_top_movers, _USER_WATCHLIST
from signal_generator import generate_signal
from swing_analyzer import analyze_swing, generate_swing_signal
from trend_filter import get_market_regime, get_daily_trend, get_relative_strength

from alpaca_trader import (
    place_bracket_orders, get_account,
    cancel_all_orders, close_all_positions,
)
from fill_monitor import check_fills, get_positions_summary
from notifier import (
    send_alert, send_signal_alert, send_eod_summary, send_startup_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

_USER_WATCHLIST_SET = set(_USER_WATCHLIST)

# ── State ─────────────────────────────────────────────────────────────────────
_signals_today:    list[dict] = []   # [{ticker, direction, grade, fill_px, pnl}]
_tickers_signaled: set[str]   = set()  # "TICKER_DIRECTION" dedup
_signaled_date:    str        = ""

# Momentum cache
_momentum_longs:      list[str] = []
_momentum_shorts:     list[str] = []
_momentum_rank_date:  str       = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MINUTE,  second=0, microsecond=0)
    close_t = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return open_t <= now <= close_t


def _reset_daily_state() -> None:
    global _signals_today, _tickers_signaled, _signaled_date
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    if today != _signaled_date:
        _signals_today    = []
        _tickers_signaled = set()
        _signaled_date    = today


def _already_signaled(ticker: str, direction: str) -> bool:
    _reset_daily_state()
    return f"{ticker}_{direction}" in _tickers_signaled


def _mark_signaled(ticker: str, direction: str) -> None:
    _reset_daily_state()
    _tickers_signaled.add(f"{ticker}_{direction}")


def _quota_ok(grade: str) -> bool:
    """Return True if we're allowed to send another auto-trade today."""
    _reset_daily_state()
    if len(_signals_today) >= AUTO_MAX_SIGNALS:
        return False
    if AUTO_MIN_GRADE == "A" and grade not in ("A",):
        return False
    if AUTO_MIN_GRADE == "B" and grade not in ("A", "B"):
        return False
    return True


def _get_momentum_ranked() -> tuple[list[str], list[str]]:
    global _momentum_longs, _momentum_shorts, _momentum_rank_date
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    if _momentum_rank_date == today and _momentum_longs:
        return _momentum_longs, _momentum_shorts
    from scanner import _STOCK_UNIVERSE
    universe = list(dict.fromkeys(list(_USER_WATCHLIST) + _STOCK_UNIVERSE[:60]))
    longs, shorts = rank_momentum(universe)
    _momentum_longs, _momentum_shorts, _momentum_rank_date = longs, shorts, today
    return longs, shorts


def _execute_signal(
    ticker: str, direction: str, grade: str, confidence: int,
    entry: float, stop: float, target: float,
    reasons: list[str], signal_type: str = "ORB",
    spy_pct: float | None = None, regime: str = "",
) -> None:
    """Validate, notify, and place Alpaca bracket orders for a single signal."""
    if not _quota_ok(grade):
        logger.info(f"Signal skipped (quota/grade): {ticker} {direction} grade={grade}")
        return

    if confidence < AUTO_MIN_CONFIDENCE:
        logger.info(f"Signal skipped (confidence {confidence}% < {AUTO_MIN_CONFIDENCE}%): {ticker}")
        return

    if _already_signaled(ticker, direction):
        logger.info(f"Signal skipped (already sent): {ticker} {direction}")
        return

    # Account daily loss guard
    acct = get_account()
    if acct and acct.get("day_pnl", 0) <= -1000:
        logger.warning(f"Daily loss limit reached — skipping {ticker}")
        return

    pos = calculate_position(entry, stop, grade)
    if not pos:
        logger.warning(f"Position sizing failed for {ticker}")
        return

    units     = pos["units"]
    risk_amt  = pos["risk_amount"]
    rr        = pos["rr"]
    r_dist    = abs(entry - stop)
    r1_price  = round(entry + r_dist if direction == "BUY" else entry - r_dist, 2)
    r2_price  = round(entry + 2 * r_dist if direction == "BUY" else entry - 2 * r_dist, 2)
    target_pnl = pos["target_pnl"]

    # 1. WhatsApp notification (before order so user can intervene)
    send_signal_alert(
        ticker=ticker, direction=direction, grade=grade,
        entry=entry, stop=stop, r1_price=r1_price, r2_price=r2_price,
        units=units, risk_amount=risk_amt, target_pnl=target_pnl,
        reasons=reasons, confidence=confidence, signal_type=signal_type,
        spy_pct=spy_pct, regime=regime,
    )

    # 2. Place orders on Alpaca
    tag = f"{ticker}_{direction}_{signal_type}"
    orders = place_bracket_orders(
        ticker=ticker, direction=direction, units=units,
        stop=stop, r1_price=r1_price, r2_price=r2_price,
        tag=tag,
    )

    if orders:
        _mark_signaled(ticker, direction)
        _signals_today.append({
            "ticker": ticker, "direction": direction, "grade": grade,
            "entry": entry, "stop": stop, "r1": r1_price, "r2": r2_price,
            "units": units, "signal_type": signal_type,
        })
        logger.info(f"Auto-trade placed: {ticker} {direction}  {len(orders)} orders")
        time.sleep(1)
    else:
        logger.error(f"Auto-trade FAILED for {ticker} {direction} — Alpaca returned no orders")
        send_alert(f"⚠️ ORDER FAILED\n{ticker} {direction} — Alpaca rejected order\n Check Railway logs.")


# ── Scan functions ────────────────────────────────────────────────────────────

def run_orb_scan() -> None:
    """ORB intraday scan — same tickers and logic as whatsapp_signal_bot."""
    if not is_market_open():
        return

    logger.info("=== [Alpaca Bot] ORB scan starting ===")
    market_regime = get_market_regime()
    spy_day_pct   = get_spy_day_pct()

    movers = get_top_movers(count=TOP_MOVERS_COUNT, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    if not movers:
        return

    for mover in movers:
        ticker = mover["ticker"]
        analysis = analyze(ticker)
        if not analysis:
            continue

        day_pct = analysis.get("day_pct", 0)
        if abs(day_pct) < 0.5 and ticker != "QQQ":
            continue

        analysis["market_regime"]     = market_regime
        analysis["spy_day_pct"]       = spy_day_pct
        analysis["daily_trend"]       = get_daily_trend(ticker)
        analysis["relative_strength"] = get_relative_strength(ticker)
        analysis["pivot_levels"]      = get_pivot_levels(ticker)
        analysis["options_sentiment"] = get_options_sentiment(ticker)

        signal = generate_signal(analysis)

        if signal.direction != "WAIT" and signal.confidence >= MIN_CONFIDENCE:
            _execute_signal(
                ticker=ticker, direction=signal.direction,
                grade=signal.grade, confidence=signal.confidence,
                entry=signal.entry, stop=signal.stop_loss, target=signal.target,
                reasons=signal.reasons, signal_type="ORB",
                spy_pct=spy_day_pct, regime=market_regime,
            )

    logger.info("=== [Alpaca Bot] ORB scan done ===")
    gc.collect()


def run_quant_scan() -> None:
    """Z-score mean reversion scan — 10:30 AM to 2:00 PM."""
    if not is_market_open():
        return
    now_et = datetime.now(_ET)
    mins   = now_et.hour * 60 + now_et.minute
    if not (630 <= mins < 840):
        return

    logger.info("=== [Alpaca Bot] Quant scan starting ===")
    longs, shorts = _get_momentum_ranked()
    candidates    = list(dict.fromkeys(longs + shorts))
    z_map         = batch_z_scores(candidates)
    if not z_map:
        return

    ranked = sorted(z_map.keys(), key=lambda t: abs(z_map[t]["z_score"]), reverse=True)

    for ticker in ranked:
        rank = longs.index(ticker) + 1 if ticker in longs else (shorts.index(ticker) + 1 if ticker in shorts else None)
        sig  = generate_quant_signal(ticker, z_data=z_map[ticker], rank=rank)
        if sig is None or sig.direction == "WAIT":
            continue
        if sig.confidence < MIN_CONFIDENCE:
            continue

        _execute_signal(
            ticker=ticker, direction=sig.direction,
            grade=sig.grade, confidence=sig.confidence,
            entry=sig.entry, stop=sig.stop_loss, target=sig.target,
            reasons=sig.reasons, signal_type="QUANT",
        )

    logger.info("=== [Alpaca Bot] Quant scan done ===")
    gc.collect()


def run_swing_scan() -> None:
    """Swing scan at 10:00 AM — hold 2-5 days."""
    if not is_market_open():
        return
    logger.info("=== [Alpaca Bot] Swing scan starting ===")

    movers = get_top_movers(count=10, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    if not movers:
        return

    sent = 0
    for mover in movers[:10]:
        if sent >= MAX_SWING_SIGNALS:
            break
        ticker = mover["ticker"]
        dte = get_days_to_earnings(ticker)
        if dte is not None and 0 <= dte <= 10:
            continue
        analysis = analyze_swing(ticker)
        if not analysis:
            continue
        signal = generate_swing_signal(analysis)
        if signal.direction == "WAIT" or signal.confidence < MIN_CONFIDENCE:
            continue

        _execute_signal(
            ticker=ticker, direction=signal.direction,
            grade=getattr(signal, "grade", "C"), confidence=signal.confidence,
            entry=signal.entry, stop=signal.stop_loss, target=signal.target,
            reasons=signal.reasons, signal_type="SWING",
        )
        sent += 1

    logger.info(f"=== [Alpaca Bot] Swing scan done. {sent} signals ===")


def run_fill_check() -> None:
    """Poll Alpaca for fills and position changes — runs every 5 min."""
    if not is_market_open():
        return
    try:
        check_fills(send_fn=send_alert)
    except Exception as e:
        logger.warning(f"Fill check error: {e}")


def run_eod() -> None:
    """EOD: cancel open orders, fetch final P&L, send summary."""
    logger.info("[Alpaca Bot] Running EOD cleanup")
    cancel_all_orders()
    time.sleep(2)

    acct = get_account()
    send_eod_summary(account=acct, trades_today=_signals_today)

    # Log positions still open (swing trades may remain)
    pos_summary = get_positions_summary()
    if "No open" not in pos_summary:
        send_alert(f"📌 Open positions after close:\n{pos_summary}")

    logger.info(f"[Alpaca Bot] EOD done. Day P&L: ${acct.get('day_pnl', 0):+,.0f}")


def run_position_status() -> None:
    """Mid-day position check at 1:00 PM — optional status ping."""
    if not is_market_open():
        return
    acct = get_account()
    pos  = get_positions_summary()
    day_pnl = acct.get("day_pnl", 0)
    pct     = acct.get("day_pnl_pct", 0)
    send_alert(
        f"📊 MID-DAY STATUS  (1:00 PM ET)\n"
        f"\n"
        f"Day P&L: ${day_pnl:+,.0f} ({pct:+.2f}%)\n"
        f"Equity:  ${acct.get('equity', 0):,.0f}\n"
        f"\n"
        f"{pos}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Alpaca Paper Bot starting...")

    send_startup_message()

    scheduler = BlockingScheduler(timezone=TIMEZONE)

    # ORB prime window: every 5 min 9:30-10:30 AM
    scheduler.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                      hour=9,  minute="30,35,40,45,50,55", id="orb_9",  misfire_grace_time=30)
    scheduler.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                      hour=10, minute="0,5,10,15,20,25,30",  id="orb_10", misfire_grace_time=30)

    # Regular intraday scan every 30 min 11 AM – 3:30 PM
    scheduler.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                      hour=f"11-{MARKET_CLOSE_HOUR}", minute=f"*/{INTERVAL_MINUTES}",
                      id="orb_intraday", misfire_grace_time=60)

    # Quant: 10:30 AM and every 30 min until 2 PM
    scheduler.add_job(run_quant_scan, "cron", day_of_week="mon-fri",
                      hour=10, minute=30, id="quant_1030")
    scheduler.add_job(run_quant_scan, "cron", day_of_week="mon-fri",
                      hour="11-13", minute="0,30", id="quant_midday")

    # Swing: once at 10:00 AM
    scheduler.add_job(run_swing_scan, "cron", day_of_week="mon-fri",
                      hour=10, minute=0, id="swing")

    # Fill monitor: every 5 min during market hours
    scheduler.add_job(run_fill_check, "cron", day_of_week="mon-fri",
                      hour=f"9-{MARKET_CLOSE_HOUR}", minute=f"*/{FILL_CHECK_INTERVAL}",
                      id="fill_check", misfire_grace_time=60)

    # Mid-day status ping at 1 PM
    scheduler.add_job(run_position_status, "cron", day_of_week="mon-fri",
                      hour=13, minute=0, id="midday_status")

    # EOD cleanup at 4:05 PM
    scheduler.add_job(run_eod, "cron", day_of_week="mon-fri",
                      hour=MARKET_CLOSE_HOUR, minute=5, id="eod")

    logger.info(
        "Alpaca Paper Bot scheduler ready:\n"
        "  📊 ORB scan      9:30-10:30 AM ET (every 5 min)\n"
        "  📊 ORB scan      11 AM-3:30 PM ET (every 30 min)\n"
        "  📐 Quant scan    10:30 AM-2:00 PM ET (every 30 min)\n"
        "  📈 Swing scan    10:00 AM ET\n"
        "  🔍 Fill monitor  every 5 min (market hours)\n"
        "  📊 Mid-day ping  1:00 PM ET\n"
        "  🏁 EOD cleanup   4:05 PM ET"
    )

    # Immediate scan if market is open on startup
    if is_market_open():
        logger.info("Market open on startup — running initial scan")
        run_orb_scan()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Alpaca Bot stopped.")


if __name__ == "__main__":
    main()
