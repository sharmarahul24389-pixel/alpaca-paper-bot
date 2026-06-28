"""
main.py — Alpaca Paper Trading Bot v2 (Adaptive Brain)

Flow per signal:
  1. Brain checks: regime, confidence threshold, skip list, position size mult
  2. Daily mode: if day P&L >= $500 target → Grade A only
  3. Loss guard: if day P&L <= -$1000 → stop trading
  4. Place 2x bracket orders on Alpaca paper account
  5. Fill monitor: poll fills, record trade outcomes to brain, send WhatsApp
  6. Brain learns: adjusts thresholds based on rolling win rates
"""
import gc
import logging
import os
import time
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from analyzer import analyze, get_spy_day_pct, get_days_to_earnings
from config import (
    TIMEZONE, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, TOP_MOVERS_COUNT,
    MIN_PRICE, MIN_VOLUME, MIN_CONFIDENCE, INTERVAL_MINUTES,
    MAX_SWING_SIGNALS, AUTO_MIN_GRADE, AUTO_MIN_CONFIDENCE,
    AUTO_MAX_SIGNALS, FILL_CHECK_INTERVAL,
    DAILY_PROFIT_TARGET, GRADE_A_ONLY_LABEL, AUTO_MAX_DAILY_LOSS,
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
from fill_monitor import (
    check_fills, get_positions_summary, get_daily_pnl,
    reset_daily_pnl, register_entry,
)
from notifier import send_alert, send_signal_alert, send_eod_summary, send_startup_message
import brain as _brain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

# ── Daily state ───────────────────────────────────────────────────────────────
_signals_today:    list[dict] = []
_tickers_signaled: set[str]   = set()
_signaled_date:    str        = ""
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
        reset_daily_pnl()


def _already_signaled(ticker: str, direction: str) -> bool:
    _reset_daily_state()
    return f"{ticker}_{direction}" in _tickers_signaled


def _mark_signaled(ticker: str, direction: str) -> None:
    _reset_daily_state()
    _tickers_signaled.add(f"{ticker}_{direction}")


def _mode() -> str:
    """Current trading mode based on day's realized P&L."""
    pnl = get_daily_pnl()
    if pnl >= DAILY_PROFIT_TARGET:
        return "A_ONLY"      # hit target — Grade A only
    if pnl <= -AUTO_MAX_DAILY_LOSS:
        return "HALTED"      # loss limit — no trading
    return "NORMAL"


def _can_trade(grade: str, confidence: int, signal_type: str) -> tuple[bool, str]:
    """Returns (ok, reason). Checks mode, brain params, grade, confidence."""
    _reset_daily_state()

    mode = _mode()
    if mode == "HALTED":
        return False, "daily loss limit reached"

    if mode == "A_ONLY" and grade != GRADE_A_ONLY_LABEL:
        return False, f"target hit — Grade A only (got {grade})"

    # Static grade filter
    if AUTO_MIN_GRADE == "A" and grade not in ("A",):
        return False, f"grade {grade} below min A"
    if AUTO_MIN_GRADE == "B" and grade not in ("A", "B"):
        return False, f"grade {grade} below min B"

    # Brain adaptive params
    params = _brain.get_params()
    skip   = params.get("skip_types", [])
    if signal_type.upper() in [s.upper() for s in skip]:
        return False, f"{signal_type} skipped by brain (low WR)"

    # Brain-adjusted confidence threshold
    conf_key = f"min_confidence_{signal_type.lower()}"
    brain_min_conf = params.get(conf_key, AUTO_MIN_CONFIDENCE)
    effective_min  = max(AUTO_MIN_CONFIDENCE, brain_min_conf)
    if confidence < effective_min:
        return False, f"confidence {confidence}% < {effective_min}% (brain-adjusted)"

    if len(_signals_today) >= AUTO_MAX_SIGNALS:
        return False, "max daily signals reached"

    return True, ""


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


# ── Core execute ──────────────────────────────────────────────────────────────

def _execute_signal(
    ticker: str, direction: str, grade: str, confidence: int,
    entry: float, stop: float, target: float,
    reasons: list[str], signal_type: str = "ORB",
    spy_pct: float | None = None, regime: str = "",
) -> None:
    if _already_signaled(ticker, direction):
        return

    ok, reason = _can_trade(grade, confidence, signal_type)
    if not ok:
        logger.info(f"Signal skipped [{reason}]: {ticker} {direction} {grade}")
        return

    # Position sizing with brain's size multiplier
    params = _brain.get_params()
    pos    = calculate_position(entry, stop, grade)
    if not pos:
        logger.warning(f"Position sizing failed: {ticker}")
        return

    mult   = params.get("position_size_mult", 1.0)
    units  = max(1, round(pos["units"] * mult))
    r_dist = abs(entry - stop)
    r1     = round(entry + r_dist     if direction=="BUY" else entry - r_dist,     2)
    r2     = round(entry + 2 * r_dist if direction=="BUY" else entry - 2 * r_dist, 2)

    pnl = get_daily_pnl()
    mode = _mode()

    # WhatsApp alert includes daily P&L context and brain mode
    mode_note = ""
    if mode == "A_ONLY":
        mode_note = "  [TARGET HIT - Grade A only mode]\n"

    send_signal_alert(
        ticker=ticker, direction=direction, grade=grade,
        entry=entry, stop=stop, r1_price=r1, r2_price=r2,
        units=units, risk_amount=pos["risk_amount"], target_pnl=pos["target_pnl"],
        reasons=reasons, confidence=confidence, signal_type=signal_type,
        spy_pct=spy_pct, regime=regime,
    )

    # Place on Alpaca
    tag    = f"{ticker}_{direction}_{signal_type}"
    orders = place_bracket_orders(
        ticker=ticker, direction=direction, units=units,
        stop=stop, r1_price=r1, r2_price=r2, tag=tag,
    )

    if orders:
        _mark_signaled(ticker, direction)
        entry_order_id = str(orders[0].id) if orders else ""
        register_entry(
            order_id=entry_order_id, ticker=ticker,
            signal_type=signal_type, grade=grade,
            direction=direction, entry=entry, qty=units,
        )
        _signals_today.append({
            "ticker": ticker, "direction": direction, "grade": grade,
            "entry": entry, "stop": stop, "r1": r1, "r2": r2,
            "units": units, "signal_type": signal_type,
        })
        logger.info(
            f"Auto-trade placed: {ticker} {direction} {grade} "
            f"x{units} (mult={mult:.2f})  day_pnl=${pnl:+.0f}"
        )
        time.sleep(1)
    else:
        logger.error(f"Auto-trade FAILED: {ticker} {direction}")
        send_alert(f"ORDER FAILED\n{ticker} {direction} — Alpaca rejected. Check logs.")


# ── Scan jobs ─────────────────────────────────────────────────────────────────

def run_orb_scan() -> None:
    if not is_market_open():
        return
    if _mode() == "HALTED":
        return

    logger.info("=== ORB scan ===")
    regime    = get_market_regime()
    spy_pct   = get_spy_day_pct()
    brain_reg = _brain.get_regime()

    movers = get_top_movers(count=TOP_MOVERS_COUNT, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    for mover in movers:
        ticker = mover["ticker"]
        analysis = analyze(ticker)
        if not analysis:
            continue
        if abs(analysis.get("day_pct", 0)) < 0.5 and ticker != "QQQ":
            continue

        analysis.update({
            "market_regime":     regime,
            "spy_day_pct":       spy_pct,
            "daily_trend":       get_daily_trend(ticker),
            "relative_strength": get_relative_strength(ticker),
            "pivot_levels":      get_pivot_levels(ticker),
            "options_sentiment": get_options_sentiment(ticker),
        })

        signal = generate_signal(analysis)
        if signal.direction != "WAIT" and signal.confidence >= MIN_CONFIDENCE:
            _execute_signal(
                ticker=ticker, direction=signal.direction,
                grade=signal.grade, confidence=signal.confidence,
                entry=signal.entry, stop=signal.stop_loss, target=signal.target,
                reasons=signal.reasons, signal_type="ORB",
                spy_pct=spy_pct, regime=f"{regime}/{brain_reg}",
            )

    logger.info("=== ORB scan done ===")
    gc.collect()


def run_quant_scan() -> None:
    if not is_market_open():
        return
    if _mode() == "HALTED":
        return

    now_et = datetime.now(_ET)
    mins   = now_et.hour * 60 + now_et.minute
    if not (630 <= mins < 840):
        return

    logger.info("=== Quant scan ===")
    longs, shorts = _get_momentum_ranked()
    candidates    = list(dict.fromkeys(longs + shorts))
    z_map         = batch_z_scores(candidates)
    if not z_map:
        return

    ranked = sorted(z_map.keys(), key=lambda t: abs(z_map[t]["z_score"]), reverse=True)
    for ticker in ranked:
        rank = longs.index(ticker) + 1 if ticker in longs else (shorts.index(ticker) + 1 if ticker in shorts else None)
        sig  = generate_quant_signal(ticker, z_data=z_map[ticker], rank=rank)
        if sig is None or sig.direction == "WAIT" or sig.confidence < MIN_CONFIDENCE:
            continue
        _execute_signal(
            ticker=ticker, direction=sig.direction,
            grade=sig.grade, confidence=sig.confidence,
            entry=sig.entry, stop=sig.stop_loss, target=sig.target,
            reasons=sig.reasons, signal_type="QUANT",
        )

    logger.info("=== Quant scan done ===")
    gc.collect()


def run_swing_scan() -> None:
    if not is_market_open():
        return
    if _mode() == "HALTED":
        return

    logger.info("=== Swing scan ===")
    movers = get_top_movers(count=10, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    sent   = 0
    for mover in movers:
        if sent >= MAX_SWING_SIGNALS:
            break
        ticker = mover["ticker"]
        if get_days_to_earnings(ticker) is not None:
            dte = get_days_to_earnings(ticker)
            if 0 <= dte <= 10:
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

    logger.info(f"=== Swing scan done ({sent} signals) ===")


def run_fill_check() -> None:
    if not is_market_open():
        return
    try:
        check_fills(send_fn=send_alert)
    except Exception as e:
        logger.warning(f"Fill check error: {e}")


def run_brain_update() -> None:
    """Refresh market regime and log brain status — runs at 9:25 AM daily."""
    _brain.get_regime(force_refresh=True)
    logger.info(_brain.summary())
    send_alert(
        f"Brain update\n\n"
        f"{_brain.summary()}\n\n"
        f"Day mode: {_mode()}  |  Day P&L: ${get_daily_pnl():+.0f}"
    )


def run_position_status() -> None:
    if not is_market_open():
        return
    acct    = get_account()
    pos     = get_positions_summary()
    day_pnl = acct.get("day_pnl", 0)
    pct     = acct.get("day_pnl_pct", 0)
    mode    = _mode()
    send_alert(
        f"MID-DAY STATUS  (1 PM ET)\n\n"
        f"Day P&L: ${day_pnl:+,.0f} ({pct:+.2f}%)\n"
        f"Mode: {mode}  |  Target: ${DAILY_PROFIT_TARGET:,.0f}\n"
        f"Equity: ${acct.get('equity', 0):,.0f}\n"
        f"Trades today: {len(_signals_today)}\n\n"
        f"{pos}\n\n"
        f"{_brain.summary()}"
    )


def run_eod() -> None:
    logger.info("EOD cleanup")
    cancel_all_orders()
    time.sleep(2)

    acct = get_account()
    send_eod_summary(account=acct, trades_today=_signals_today)

    pos = get_positions_summary()
    if "No open" not in pos:
        send_alert(f"Open positions after close:\n{pos}")

    _brain.update_daily_pnl(get_daily_pnl())
    logger.info(f"EOD done. Day P&L: ${acct.get('day_pnl', 0):+,.0f}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Alpaca Paper Bot starting...")
    send_startup_message()

    sched = BlockingScheduler(timezone=TIMEZONE)

    # ORB: 9:30-10:30 every 5 min
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=9,  minute="30,35,40,45,50,55", id="orb_9")
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute="0,5,10,15,20,25,30",  id="orb_10")
    # ORB: 11 AM – 3:30 PM every 30 min
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=f"11-{MARKET_CLOSE_HOUR}", minute=f"*/{INTERVAL_MINUTES}",
                  id="orb_intraday", misfire_grace_time=60)

    # Quant: 10:30 AM – 2:00 PM every 30 min
    sched.add_job(run_quant_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute=30, id="quant_1030")
    sched.add_job(run_quant_scan, "cron", day_of_week="mon-fri",
                  hour="11-13", minute="0,30", id="quant_mid")

    # Swing: 10:00 AM
    sched.add_job(run_swing_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute=0, id="swing")

    # Fill monitor: every 5 min
    sched.add_job(run_fill_check, "cron", day_of_week="mon-fri",
                  hour=f"9-{MARKET_CLOSE_HOUR}", minute=f"*/{FILL_CHECK_INTERVAL}",
                  id="fills", misfire_grace_time=60)

    # Brain refresh + regime update at market open
    sched.add_job(run_brain_update, "cron", day_of_week="mon-fri",
                  hour=9, minute=25, id="brain_update")

    # Mid-day status
    sched.add_job(run_position_status, "cron", day_of_week="mon-fri",
                  hour=13, minute=0, id="midday")

    # EOD
    sched.add_job(run_eod, "cron", day_of_week="mon-fri",
                  hour=MARKET_CLOSE_HOUR, minute=5, id="eod")

    logger.info(
        "Alpaca Paper Bot v2 (Adaptive Brain) scheduler ready:\n"
        "  ORB scan      9:30-10:30 AM ET (every 5 min)\n"
        "  ORB scan      11 AM-3:30 PM ET (every 30 min)\n"
        "  Quant scan    10:30 AM-2:00 PM ET (every 30 min)\n"
        "  Swing scan    10:00 AM ET\n"
        "  Fill monitor  every 5 min (market hours)\n"
        "  Brain update  9:25 AM ET daily\n"
        "  Mid-day ping  1:00 PM ET\n"
        "  EOD cleanup   4:05 PM ET\n"
        f"  Daily target  ${DAILY_PROFIT_TARGET:,.0f} then Grade-A only\n"
        f"  Loss halt     -${AUTO_MAX_DAILY_LOSS:,.0f}"
    )

    if is_market_open():
        logger.info("Market open on startup — running initial scan")
        run_orb_scan()

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Alpaca Bot stopped.")


if __name__ == "__main__":
    main()
