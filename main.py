"""
main.py — Alpaca Paper Trading Bot v3 (Full Intelligence)

Signal pipeline per trade:
  1. Trend alignment  (20-day MA — biggest single win-rate driver)
  2. Earnings block   (skip ORB/Quant within 3 days of earnings)
  3. Catalyst score   (Trump post, Congress trade, SEC 8-K, sector ETF)
  4. Brain filters    (adaptive confidence, skip underperforming types)
  5. Weekly loss mult (reduce size if week is in drawdown)
  6. Daily mode       (Grade A only after $500 target, halt at -$1000)
  7. Place 2x bracket orders on Alpaca paper account
  8. Fill monitor     (P&L tracking, trailing stop after +1R, brain recording)
  9. Time stop        (close flat positions after 90 min)
"""
import gc
import logging
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
    TIME_STOP_MINUTES, TREND_FILTER_ENABLED, CATALYST_HARD_SKIP_SCORE,
    CATALYST_GRADE_A_SCORE, EARNINGS_BLOCK_DAYS, MAX_SECTOR_SIGNALS,
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
    close_position, move_stop_to_breakeven,
)
from fill_monitor import (
    check_fills, get_positions_summary, get_daily_pnl,
    reset_daily_pnl, register_entry,
)
from notifier import send_alert, send_signal_alert, send_eod_summary, send_startup_message
import brain as _brain
from catalyst import (
    check_trend_alignment, get_catalyst_score,
    get_trump_catalyst, get_congress_buys,
    STOCK_TO_SECTOR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

# ── Daily state ───────────────────────────────────────────────────────────────
_signals_today:    list[dict]  = []
_tickers_signaled: set[str]    = set()
_sector_counts:    dict[str,int] = {}   # correlation filter: trades per sector today
_signaled_date:    str         = ""
_momentum_longs:   list[str]   = []
_momentum_shorts:  list[str]   = []
_momentum_rank_date: str       = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MINUTE,  second=0, microsecond=0)
    c = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return o <= now <= c


def _reset_daily_state() -> None:
    global _signals_today, _tickers_signaled, _sector_counts, _signaled_date
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    if today != _signaled_date:
        _signals_today    = []
        _tickers_signaled = set()
        _sector_counts    = {}
        _signaled_date    = today
        reset_daily_pnl()


def _already_signaled(ticker: str, direction: str) -> bool:
    _reset_daily_state()
    return f"{ticker}_{direction}" in _tickers_signaled


def _mark_signaled(ticker: str, direction: str) -> None:
    _reset_daily_state()
    _tickers_signaled.add(f"{ticker}_{direction}")


def _mode() -> str:
    pnl = get_daily_pnl()
    if pnl >= DAILY_PROFIT_TARGET:
        return "A_ONLY"
    if pnl <= -AUTO_MAX_DAILY_LOSS:
        return "HALTED"
    return "NORMAL"


def _can_trade(grade: str, confidence: int, signal_type: str) -> tuple[bool, str]:
    _reset_daily_state()

    # Weekly halt check
    w_mult = _brain.get_weekly_size_mult()
    if w_mult == 0.0:
        return False, "weekly loss halt — resumes Monday"

    mode = _mode()
    if mode == "HALTED":
        return False, "daily loss limit reached"
    if mode == "A_ONLY" and grade != GRADE_A_ONLY_LABEL:
        return False, f"$500 target hit — Grade A only (got {grade})"

    # Static grade filter
    if AUTO_MIN_GRADE == "A" and grade not in ("A",):
        return False, f"grade {grade} below configured min A"
    if AUTO_MIN_GRADE == "B" and grade not in ("A", "B"):
        return False, f"grade {grade} below configured min B"

    # Brain adaptive params
    params = _brain.get_params()
    if signal_type.upper() in [s.upper() for s in params.get("skip_types", [])]:
        return False, f"{signal_type} paused by brain (low WR)"

    conf_key     = f"min_confidence_{signal_type.lower()}"
    brain_min    = params.get(conf_key, AUTO_MIN_CONFIDENCE)
    eff_min      = max(AUTO_MIN_CONFIDENCE, brain_min)
    if confidence < eff_min:
        return False, f"confidence {confidence}% < {eff_min}% (brain-adjusted)"

    if len(_signals_today) >= AUTO_MAX_SIGNALS:
        return False, "max daily signals reached"

    return True, ""


def _earnings_too_close(ticker: str, block_days: int = EARNINGS_BLOCK_DAYS) -> bool:
    """Return True if earnings within block_days — skip ORB/Quant to avoid coin-flip."""
    try:
        dte = get_days_to_earnings(ticker)
        return dte is not None and 0 <= dte <= block_days
    except Exception:
        return False


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
    skip_earnings_check: bool = False,
) -> None:

    if _already_signaled(ticker, direction):
        return

    # ── Correlation filter: max MAX_SECTOR_SIGNALS per sector per day ─────────
    if MAX_SECTOR_SIGNALS > 0:
        sector = STOCK_TO_SECTOR.get(ticker, "UNKNOWN")
        if _sector_counts.get(sector, 0) >= MAX_SECTOR_SIGNALS:
            logger.info(f"Skip sector limit: {ticker} ({sector}) already has {MAX_SECTOR_SIGNALS} trades today")
            return

    ok, reason = _can_trade(grade, confidence, signal_type)
    if not ok:
        logger.info(f"Skip [{reason}]: {ticker} {direction} {grade}")
        return

    # ── Earnings block (ORB & QUANT only — swing already checks) ──────────────
    if not skip_earnings_check and signal_type in ("ORB", "QUANT"):
        if _earnings_too_close(ticker):
            logger.info(f"Skip earnings risk: {ticker} within {EARNINGS_BLOCK_DAYS}d of report")
            return

    # ── Trend alignment (20-day MA) ───────────────────────────────────────────
    if TREND_FILTER_ENABLED:
        trend_ok, trend_reason = check_trend_alignment(ticker, direction)
        if not trend_ok:
            # Allow override only for Grade A + strong catalyst
            logger.info(f"Trend miss: {ticker} {direction} — {trend_reason}")
            # Will be overridden below if catalyst strongly confirms

    # ── Catalyst scoring ──────────────────────────────────────────────────────
    cat_score, cat_reasons = get_catalyst_score(ticker, direction)
    all_reasons = list(reasons) + cat_reasons

    # Hard skip: catalyst strongly opposes AND (trend miss OR grade < A)
    if cat_score <= CATALYST_HARD_SKIP_SCORE:
        logger.info(f"Skip catalyst: {ticker} {direction} score={cat_score}  {cat_reasons}")
        return

    # If trend misaligned: need catalyst to confirm OR must be Grade A
    if TREND_FILTER_ENABLED and not trend_ok:
        if cat_score < 2 and grade != "A":
            logger.info(f"Skip trend+catalyst: {ticker} {direction} grade={grade} cat={cat_score}")
            return
        # Grade A with good catalyst can override trend miss
        all_reasons = [trend_reason] + all_reasons

    # If catalyst opposes weakly (-2 to -1): require Grade A
    if cat_score <= CATALYST_GRADE_A_SCORE and grade != "A":
        logger.info(f"Skip weak catalyst: {ticker} {direction} score={cat_score} grade={grade}")
        return

    # ── Position sizing (brain mult × weekly mult) ────────────────────────────
    params     = _brain.get_params()
    brain_mult = params.get("position_size_mult", 1.0)
    week_mult  = _brain.get_weekly_size_mult()
    # Catalyst boost: strong confirmation (+2) → slight size increase
    cat_mult   = 1.15 if cat_score >= 2 else 1.0
    total_mult = brain_mult * week_mult * cat_mult

    pos = calculate_position(entry, stop, grade)
    if not pos:
        logger.warning(f"Position sizing failed: {ticker}")
        return

    units  = max(1, round(pos["units"] * total_mult))
    r_dist = abs(entry - stop)
    r1     = round(entry + r_dist     if direction=="BUY" else entry - r_dist,     2)
    r2     = round(entry + 2 * r_dist if direction=="BUY" else entry - 2 * r_dist, 2)

    # ── WhatsApp alert ────────────────────────────────────────────────────────
    send_signal_alert(
        ticker=ticker, direction=direction, grade=grade,
        entry=entry, stop=stop, r1_price=r1, r2_price=r2,
        units=units, risk_amount=pos["risk_amount"], target_pnl=pos["target_pnl"],
        reasons=all_reasons, confidence=confidence, signal_type=signal_type,
        spy_pct=spy_pct, regime=regime,
    )

    # ── Place on Alpaca ───────────────────────────────────────────────────────
    tag    = f"{ticker}_{direction}_{signal_type}"
    orders = place_bracket_orders(
        ticker=ticker, direction=direction, units=units,
        stop=stop, r1_price=r1, r2_price=r2, tag=tag,
    )

    if orders:
        _mark_signaled(ticker, direction)
        sector = STOCK_TO_SECTOR.get(ticker, "UNKNOWN")
        _sector_counts[sector] = _sector_counts.get(sector, 0) + 1
        register_entry(
            order_id=str(orders[0].id), ticker=ticker,
            signal_type=signal_type, grade=grade,
            direction=direction, entry=entry, qty=units,
        )
        _signals_today.append({
            "ticker":     ticker,
            "direction":  direction,
            "grade":      grade,
            "entry":      entry,
            "stop":       stop,
            "r1":         r1,
            "r2":         r2,
            "units":      units,
            "signal_type":signal_type,
            "opened_at":  datetime.now(_ET),   # for time-based stop
        })
        logger.info(
            f"Trade placed: {ticker} {direction} {grade} x{units} "
            f"(brain={brain_mult:.2f} wk={week_mult:.2f} cat={cat_mult:.2f}) "
            f"cat_score={cat_score:+d}  day_pnl=${get_daily_pnl():+.0f}"
        )
        time.sleep(0.5)
    else:
        logger.error(f"Order FAILED: {ticker} {direction}")
        send_alert(f"ORDER FAILED\n{ticker} {direction} — Alpaca rejected. Check logs.")


# ── Scan jobs ─────────────────────────────────────────────────────────────────

def run_orb_scan() -> None:
    if not is_market_open() or _mode() == "HALTED":
        return
    logger.info("=== ORB scan ===")
    regime  = get_market_regime()
    spy_pct = get_spy_day_pct()

    movers = get_top_movers(count=TOP_MOVERS_COUNT, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    for mover in movers:
        ticker   = mover["ticker"]
        analysis = analyze(ticker)
        day_pct_val = analysis.get("day_pct", 0) if analysis else 0
        # Skip if not moving enough, or already too extended (ORB play is over)
        if not analysis or abs(day_pct_val) < 0.5 or abs(day_pct_val) > 4.0:
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
                spy_pct=spy_pct, regime=regime,
            )
    logger.info("=== ORB done ===")
    gc.collect()


def run_quant_scan() -> None:
    if not is_market_open() or _mode() == "HALTED":
        return
    now_et = datetime.now(_ET)
    mins   = now_et.hour * 60 + now_et.minute
    if not (630 <= mins < 840):
        return
    logger.info("=== Quant scan ===")
    longs, shorts = _get_momentum_ranked()
    z_map         = batch_z_scores(list(dict.fromkeys(longs + shorts)))
    if not z_map:
        return
    ranked = sorted(z_map, key=lambda t: abs(z_map[t]["z_score"]), reverse=True)
    for ticker in ranked:
        rank = longs.index(ticker)+1 if ticker in longs else (shorts.index(ticker)+1 if ticker in shorts else None)
        sig  = generate_quant_signal(ticker, z_data=z_map[ticker], rank=rank)
        if sig is None or sig.direction == "WAIT" or sig.confidence < MIN_CONFIDENCE:
            continue
        _execute_signal(
            ticker=ticker, direction=sig.direction,
            grade=sig.grade, confidence=sig.confidence,
            entry=sig.entry, stop=sig.stop_loss, target=sig.target,
            reasons=sig.reasons, signal_type="QUANT",
        )
    logger.info("=== Quant done ===")
    gc.collect()


def run_swing_scan() -> None:
    if not is_market_open() or _mode() == "HALTED":
        return
    logger.info("=== Swing scan ===")
    movers = get_top_movers(count=10, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    sent   = 0
    for mover in movers:
        if sent >= MAX_SWING_SIGNALS:
            break
        ticker = mover["ticker"]
        if _earnings_too_close(ticker, block_days=10):
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
            skip_earnings_check=True,   # we already checked above
        )
        sent += 1
    logger.info(f"=== Swing done ({sent}) ===")


def run_fill_check() -> None:
    if not is_market_open():
        return
    try:
        check_fills(send_fn=send_alert)
    except Exception as e:
        logger.warning(f"Fill check error: {e}")


def run_time_stop_check() -> None:
    """
    Close positions that are flat (within ±0.3R) after TIME_STOP_MINUTES.
    Frees capital for better setups instead of holding dead trades.
    """
    if not is_market_open() or TIME_STOP_MINUTES <= 0:
        return
    _reset_daily_state()
    from alpaca_trader import get_open_positions
    now    = datetime.now(_ET)
    open_p = {p.symbol: p for p in get_open_positions()}

    for sig in _signals_today:
        ticker    = sig["ticker"]
        opened_at = sig.get("opened_at")
        if not opened_at or ticker not in open_p:
            continue
        age_min = (now - opened_at).total_seconds() / 60
        if age_min < TIME_STOP_MINUTES:
            continue

        pos    = open_p[ticker]
        upnl   = float(pos.unrealized_pl)
        qty    = abs(int(float(pos.qty)))
        r_dist = abs(sig["entry"] - sig["stop"]) * qty
        flat   = r_dist > 0 and abs(upnl) < r_dist * 0.3

        if flat:
            logger.info(f"Time stop: {ticker} flat after {age_min:.0f} min (P&L ${upnl:+.0f}) — closing")
            if close_position(ticker):
                send_alert(
                    f"TIME STOP — {ticker}\n\n"
                    f"  Held {age_min:.0f} min with no move\n"
                    f"  Closed near breakeven: ${upnl:+.0f}\n"
                    f"  Capital freed for better setup"
                )


def run_brain_update() -> None:
    """9:25 AM: refresh regime, pull congress + Trump catalysts, log brain status."""
    _brain.get_regime(force_refresh=True)
    trump    = get_trump_catalyst()
    congress = get_congress_buys()

    trump_note    = trump["summary"] if trump["active"] else "No active Trump posts"
    congress_note = f"{len(congress)} active tickers" if congress else "None"

    status = _brain.summary()
    logger.info(status)
    send_alert(
        f"Morning Brain Update\n\n"
        f"{status}\n\n"
        f"Trump: {trump_note}\n"
        f"Congress trades: {congress_note}\n"
        f"Day mode: {_mode()}  |  Day P&L: ${get_daily_pnl():+.0f}\n"
        f"Week P&L: ${_brain.get_weekly_pnl():+.0f}"
    )


def run_position_status() -> None:
    if not is_market_open():
        return
    acct    = get_account()
    pos     = get_positions_summary()
    day_pnl = acct.get("day_pnl", 0)
    send_alert(
        f"MID-DAY STATUS  (1 PM ET)\n\n"
        f"Day P&L  : ${day_pnl:+,.0f}  |  Mode: {_mode()}\n"
        f"Week P&L : ${_brain.get_weekly_pnl():+,.0f}\n"
        f"Equity   : ${acct.get('equity',0):,.0f}\n"
        f"Trades   : {len(_signals_today)}\n\n"
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
    logger.info(f"EOD: day P&L ${acct.get('day_pnl',0):+,.0f}  week ${_brain.get_weekly_pnl():+,.0f}")


def run_weekly_summary() -> None:
    """Friday 4:15 PM ET — full week recap sent to Telegram."""
    logger.info("=== Weekly summary ===")
    state     = _brain.load()
    trades    = state.get("trades", [])
    daily_pnl = state.get("daily_pnl", {})

    # This week Mon–Fri
    from datetime import date, timedelta
    today  = date.today()
    monday = today - timedelta(days=today.weekday())
    week_dates = [(monday + timedelta(days=d)).isoformat() for d in range(5)]

    week_trades = [t for t in trades if t.get("date", "") >= monday.isoformat()]
    week_pnl    = sum(daily_pnl.get(d, 0) for d in week_dates)
    acct        = get_account()
    equity      = acct.get("equity", 100_000)

    wins     = [t for t in week_trades if t.get("result") == "WIN"]
    losses   = [t for t in week_trades if t.get("result") == "LOSS"]
    scratches = [t for t in week_trades if t.get("result") == "SCRATCH"]
    win_rate = len(wins) / len(week_trades) * 100 if week_trades else 0

    # Best and worst trade
    best  = max(week_trades, key=lambda t: t.get("pnl", 0), default=None)
    worst = min(week_trades, key=lambda t: t.get("pnl", 0), default=None)

    # Daily breakdown
    day_lines = []
    for d in week_dates:
        dp = daily_pnl.get(d, None)
        if dp is not None:
            emoji = "✅" if dp >= 0 else "❌"
            day_lines.append(f"  {emoji} {d}  ${dp:+,.0f}")

    regime = state.get("last_regime", "UNKNOWN")
    w_mult = _brain.get_weekly_size_mult()

    lines = [
        "📊 <b>WEEKLY SUMMARY</b>",
        f"Week of {monday.strftime('%b %d')} – {today.strftime('%b %d, %Y')}",
        "",
        f"💰 Week P&L  : <b>${week_pnl:+,.0f}</b>",
        f"   Equity    : ${equity:,.0f}",
        f"   Return    : {week_pnl/100_000*100:+.2f}%",
        "",
        f"📈 Trades    : {len(week_trades)}  |  W:{len(wins)} L:{len(losses)} S:{len(scratches)}",
        f"   Win Rate  : {win_rate:.1f}%",
    ]

    if best:
        lines.append(f"   Best      : {best['ticker']} {best['direction']} +${best['pnl']:,.0f}")
    if worst:
        lines.append(f"   Worst     : {worst['ticker']} {worst['direction']} ${worst['pnl']:,.0f}")

    if day_lines:
        lines += ["", "📅 Daily P&L:"] + day_lines

    lines += [
        "",
        f"🧠 Regime    : {regime}  |  Next week size: {w_mult:.2f}x",
        f"   {_brain.summary().splitlines()[1].strip()}",
        "",
        "Have a great weekend! 🎉",
    ]

    send_alert("\n".join(lines))
    logger.info(f"Weekly summary sent: {len(week_trades)} trades, P&L ${week_pnl:+,.0f}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Alpaca Paper Bot v3 (Full Intelligence) starting...")
    send_startup_message()

    sched = BlockingScheduler(timezone=TIMEZONE)

    # ORB: 9:30–10:30 every 5 min, then every 30 min until 3:30
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=9,  minute="30,35,40,45,50,55", id="orb_9")
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute="0,5,10,15,20,25,30",  id="orb_10")
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=f"11-{MARKET_CLOSE_HOUR}", minute=f"*/{INTERVAL_MINUTES}",
                  id="orb_intraday", misfire_grace_time=60)

    # Quant: 10:30 AM – 2 PM every 30 min
    sched.add_job(run_quant_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute=30, id="quant_1030")
    sched.add_job(run_quant_scan, "cron", day_of_week="mon-fri",
                  hour="11-13", minute="0,30", id="quant_mid")

    # Swing: 10:00 AM
    sched.add_job(run_swing_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute=0, id="swing")

    # Fill check: every 5 min
    sched.add_job(run_fill_check, "cron", day_of_week="mon-fri",
                  hour=f"9-{MARKET_CLOSE_HOUR}", minute=f"*/{FILL_CHECK_INTERVAL}",
                  id="fills", misfire_grace_time=60)

    # Time stop: every 30 min during market hours
    sched.add_job(run_time_stop_check, "cron", day_of_week="mon-fri",
                  hour=f"10-{MARKET_CLOSE_HOUR}", minute="0,30", id="time_stop")

    # Brain + catalyst refresh at 9:25 AM
    sched.add_job(run_brain_update, "cron", day_of_week="mon-fri",
                  hour=9, minute=25, id="brain_update")

    # Mid-day status
    sched.add_job(run_position_status, "cron", day_of_week="mon-fri",
                  hour=13, minute=0, id="midday")

    # EOD
    sched.add_job(run_eod, "cron", day_of_week="mon-fri",
                  hour=MARKET_CLOSE_HOUR, minute=5, id="eod")

    # Weekly summary — Friday 4:15 PM ET
    sched.add_job(run_weekly_summary, "cron", day_of_week="fri",
                  hour=MARKET_CLOSE_HOUR, minute=15, id="weekly")

    logger.info(
        "Bot v3 ready:\n"
        "  Filters : trend(20MA) + sector ETF + earnings block + Trump/Congress/SEC\n"
        f"  Target  : ${DAILY_PROFIT_TARGET:,.0f}/day → Grade A only after\n"
        f"  Stops   : -${AUTO_MAX_DAILY_LOSS:,.0f} daily | time stop {TIME_STOP_MINUTES}min\n"
        "  Brain   : adaptive confidence + position size + weekly halt\n"
        "  9:25 AM : regime + Trump + Congress briefing sent to WhatsApp"
    )

    if is_market_open():
        run_orb_scan()

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
