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
import hashlib
import logging
import time
import xml.etree.ElementTree as _ET_xml
from datetime import datetime

import requests

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from analyzer import analyze, get_spy_day_pct, get_qqq_day_pct, get_days_to_earnings
from config import (
    TIMEZONE, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, TOP_MOVERS_COUNT,
    MIN_PRICE, MIN_VOLUME, MIN_CONFIDENCE, INTERVAL_MINUTES,
    MAX_SWING_SIGNALS, AUTO_MIN_GRADE, AUTO_MIN_CONFIDENCE,
    AUTO_MAX_SIGNALS, FILL_CHECK_INTERVAL,
    DAILY_PROFIT_TARGET, PROFIT_PROTECT_DRAWDOWN,
    GRADE_A_ONLY_LABEL, AUTO_MAX_DAILY_LOSS,
    TIME_STOP_MINUTES, TREND_FILTER_ENABLED, CATALYST_HARD_SKIP_SCORE,
    CATALYST_GRADE_A_SCORE, EARNINGS_BLOCK_DAYS, MAX_SECTOR_SIGNALS,
    NEWS_RSS_FEEDS, TRUMP_KEYWORDS, MARKET_EVENT_KEYWORDS,
    ACCOUNT_SIZE,
)
from levels import get_pivot_levels
from options_flow import get_options_sentiment
from position_sizer import calculate_position
from scanner import get_top_movers, _USER_WATCHLIST, sort_by_premarket_activity
from signal_generator import generate_signal
from swing_analyzer import analyze_swing, generate_swing_signal
from trend_filter import get_market_regime, get_daily_trend, get_relative_strength

from alpaca_trader import (
    place_bracket_orders, get_account,
    cancel_all_orders, close_all_positions,
    close_position, move_stop_to_breakeven,
    is_market_open,
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
_target_ever_hit:  bool        = False  # True once day P&L >= DAILY_PROFIT_TARGET
_news_alerted_ids: set[str]    = set()  # md5 hashes of headlines/posts already alerted


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_daily_state() -> None:
    global _signals_today, _tickers_signaled, _sector_counts, _signaled_date, _target_ever_hit, _news_alerted_ids
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    if today != _signaled_date:
        _signals_today    = []
        _tickers_signaled = set()
        _sector_counts    = {}
        _signaled_date    = today
        _target_ever_hit  = False
        _news_alerted_ids = set()
        reset_daily_pnl()


def _already_signaled(ticker: str, direction: str) -> bool:
    _reset_daily_state()
    return f"{ticker}_{direction}" in _tickers_signaled


def _mark_signaled(ticker: str, direction: str) -> None:
    _reset_daily_state()
    _tickers_signaled.add(f"{ticker}_{direction}")


def _mode() -> str:
    global _target_ever_hit
    local_pnl  = get_daily_pnl()
    acct       = get_account()
    alpaca_pnl = acct.get("day_pnl", 0) if acct else 0
    worst_pnl  = min(local_pnl, alpaca_pnl)   # conservative for halt
    best_pnl   = max(local_pnl, alpaca_pnl)   # optimistic for profit target (survives restart)

    if worst_pnl <= -AUTO_MAX_DAILY_LOSS:
        return "HALTED"

    if best_pnl >= DAILY_PROFIT_TARGET:
        _target_ever_hit = True

    if _target_ever_hit and best_pnl <= DAILY_PROFIT_TARGET - PROFIT_PROTECT_DRAWDOWN:
        return "HALTED"

    if _target_ever_hit or best_pnl >= DAILY_PROFIT_TARGET:
        return "A_ONLY"

    return "NORMAL"


_BLOCKED_SIGNAL_TYPES: set[str] = {"QUANT"}  # hard-blocked; 6% WR, net money-loser

def _can_trade(grade: str, confidence: int, signal_type: str) -> tuple[bool, str]:
    _reset_daily_state()

    if signal_type.upper() in _BLOCKED_SIGNAL_TYPES:
        return False, f"{signal_type} permanently disabled"

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

    # Use live equity so profits compound into larger position sizes
    _acct = get_account()
    live_equity = _acct.get("equity", ACCOUNT_SIZE) if _acct else ACCOUNT_SIZE

    pos = calculate_position(entry, stop, grade, account_size=live_equity)
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
    # Include minute-level timestamp so Alpaca client_order_id is always unique
    # even if the same signal fires twice (bot restart, re-scan, stop hit & re-trigger)
    import time as _time
    tag    = f"{ticker}_{direction}_{signal_type}_{int(_time.time())}"
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
            "fill_px":    entry,               # signal price; updated to actual fill by fill_monitor
            "pnl":        0,
            "reasons":    list(all_reasons),   # why the bot took this trade (for reel commentary)
            "confidence": confidence,
            "cat_score":  cat_score,
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
    qqq_pct = get_qqq_day_pct()

    movers = get_top_movers(count=TOP_MOVERS_COUNT, min_price=MIN_PRICE, min_volume=MIN_VOLUME)
    for mover in movers:
        ticker   = mover["ticker"]
        analysis = analyze(ticker)
        day_pct_val = analysis.get("day_pct", 0) if analysis else 0
        # Skip if not moving enough, or already too extended (ORB play is over)
        if not analysis or abs(day_pct_val) < 0.3 or abs(day_pct_val) > 6.0:
            continue
        analysis.update({
            "market_regime":     regime,
            "spy_day_pct":       spy_pct,
            "qqq_day_pct":       qqq_pct,
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
        check_fills(send_fn=send_alert, signals_list=_signals_today)
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


def run_orb_diagnostic() -> None:
    """10:00 AM — verify ORB data is populating from Alpaca IEX. Sent to Telegram."""
    from analyzer import fetch_5min, _orb_from_df, _orb_15min
    regime  = get_market_regime()
    spy_pct = get_spy_day_pct()
    movers  = get_top_movers(count=5, min_price=MIN_PRICE, min_volume=MIN_VOLUME)

    lines = ["ORB DIAGNOSTIC — 10:00 AM\n"]
    for m in movers[:3]:
        ticker = m["ticker"]
        try:
            df5 = fetch_5min(ticker)
            orb30 = _orb_from_df(df5) if df5 is not None else None
            orb15 = _orb_15min(ticker, df5) if df5 is not None else None
            orb   = orb15 or orb30
            if orb:
                lines.append(
                    f"{ticker}: ORB OK  H=${orb['high']:.2f}  L=${orb['low']:.2f}  "
                    f"range=${orb['range']:.2f}"
                )
            else:
                src = "no 5m bars" if df5 is None or df5.empty else "9:30 bar missing"
                lines.append(f"{ticker}: ORB MISSING — {src}")
        except Exception as exc:
            lines.append(f"{ticker}: ERROR — {exc}")

    lines.append(f"\nRegime: {regime}  |  SPY: {spy_pct:+.2f}%")
    send_alert("\n".join(lines))
    logger.info("ORB diagnostic sent")


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
    time.sleep(3)
    close_all_positions()
    time.sleep(15)   # wait for paper fills before verification

    # Verify all positions are actually closed; retry stragglers individually
    from alpaca_trader import get_open_positions
    for attempt in range(3):
        remaining = get_open_positions()
        if not remaining:
            break
        syms = [p.symbol for p in remaining]
        logger.warning(f"EOD verify attempt {attempt+1}: {len(remaining)} still open — {syms}")
        for pos in remaining:
            close_position(pos.symbol)
        time.sleep(8)
    else:
        # After 3 attempts, still open — alert so user can manually close
        still_open = get_open_positions()
        if still_open:
            syms = ", ".join(p.symbol for p in still_open)
            msg  = (
                f"EOD CLOSE FAILED — MANUAL ACTION REQUIRED\n\n"
                f"Positions still open after 3 attempts:\n{syms}\n\n"
                f"Please close manually on Alpaca dashboard."
            )
            logger.error(msg)
            send_alert(msg)

    # Capture EOD closure fills so per-trade P&L is accurate in the summary
    try:
        check_fills(send_fn=send_alert, signals_list=_signals_today)
    except Exception as e:
        logger.warning(f"EOD fill capture failed: {e}")
    time.sleep(2)
    acct = get_account()
    send_eod_summary(account=acct, trades_today=_signals_today)
    _brain.update_daily_pnl(get_daily_pnl())
    logger.info(f"EOD: day P&L ${acct.get('day_pnl',0):+,.0f}  week ${_brain.get_weekly_pnl():+,.0f}")

    # Generate daily Instagram reel — saves to Desktop/Trading_Reels/ + sends via Telegram
    try:
        from reel_generator import generate_reel
        generate_reel(signals_today=_signals_today, account=acct)
    except Exception as _reel_err:
        logger.warning(f"Reel generation skipped: {_reel_err}")


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
        f"   Return    : {week_pnl/equity*100:+.2f}%",
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


def run_news_check() -> None:
    """Every 30 min during market hours: check RSS + Truth Social for actionable news."""
    global _news_alerted_ids
    now = datetime.now(_ET)
    if now.weekday() >= 5 or not (9 <= now.hour < 16):
        return

    alerts: list[str] = []

    # ── Trump Truth Social: alert when specific stocks are named ─────────────
    trump = get_trump_catalyst()
    if trump["active"] and trump.get("mentioned_tickers"):
        key = hashlib.md5(trump["summary"].encode()).hexdigest()
        if key not in _news_alerted_ids:
            _news_alerted_ids.add(key)
            tickers = "  ".join(trump["mentioned_tickers"])
            alerts.append(
                f"TRUMP TRUTH SOCIAL\n\n"
                f"STOCKS MENTIONED: {tickers}\n\n"
                f"{trump['summary']}\n\n"
                f"ACTION: Review for momentum — Trump mention = catalyst\n"
                f"{now.strftime('%I:%M %p ET')}"
            )

    # ── RSS feeds: new Trump/macro headlines ─────────────────────────────────
    trump_hits: list[str] = []
    macro_hits: list[str] = []

    for feed_url in NEWS_RSS_FEEDS:
        try:
            r = requests.get(feed_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            root = _ET_xml.fromstring(r.content)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue
                key = hashlib.md5(title.lower().encode()).hexdigest()
                if key in _news_alerted_ids:
                    continue
                tl = title.lower()
                if any(kw in tl for kw in TRUMP_KEYWORDS) and len(trump_hits) < 3:
                    trump_hits.append(title)
                    _news_alerted_ids.add(key)
                elif any(kw in tl for kw in MARKET_EVENT_KEYWORDS) and len(macro_hits) < 3:
                    macro_hits.append(title)
                    _news_alerted_ids.add(key)
        except Exception as exc:
            logger.debug(f"RSS {feed_url}: {exc}")

    if trump_hits or macro_hits:
        lines = ["MARKET NEWS ALERT"]
        if trump_hits:
            lines += ["", "TRUMP / POLITICAL:"]
            for h in trump_hits:
                lines.append(f"  - {h[:100]}")
        if macro_hits:
            lines += ["", "MACRO / FED:"]
            for h in macro_hits:
                lines.append(f"  - {h[:100]}")
        lines.append(f"\n{now.strftime('%I:%M %p ET')}")
        alerts.append("\n".join(lines))

    for msg in alerts:
        send_alert(msg)

    if alerts:
        logger.info(f"News check: sent {len(alerts)} alert(s)")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Alpaca Paper Bot v3 (Full Intelligence) starting...")
    send_startup_message()

    sched = BlockingScheduler(timezone=TIMEZONE)

    # ORB: 9:30–10:30 every 5 min, then every 30 min until 3:30
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=9,  minute="30,35,40,45,50,55", id="orb_9")
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute="*/5",  id="orb_10")
    sched.add_job(run_orb_scan, "cron", day_of_week="mon-fri",
                  hour="11-15", minute=f"*/{INTERVAL_MINUTES}",
                  id="orb_intraday", misfire_grace_time=60)

    # ORB diagnostic: 10:00 AM — confirms IEX data is populating before scans ramp up
    sched.add_job(run_orb_diagnostic, "cron", day_of_week="mon-fri",
                  hour=10, minute=0, id="orb_diag")

    # Swing: 10:00 AM
    sched.add_job(run_swing_scan, "cron", day_of_week="mon-fri",
                  hour=10, minute=0, id="swing")

    # Fill check: every 5 min
    sched.add_job(run_fill_check, "cron", day_of_week="mon-fri",
                  hour=f"9-{MARKET_CLOSE_HOUR}", minute=f"*/{FILL_CHECK_INTERVAL}",
                  id="fills", misfire_grace_time=60)

    # Time stop: every 30 min during market hours
    sched.add_job(run_time_stop_check, "cron", day_of_week="mon-fri",
                  hour="10-15", minute="0,30", id="time_stop")

    # Pre-market sort at 9:20 AM — prioritises today's active stocks in the scan window
    sched.add_job(sort_by_premarket_activity, "cron", day_of_week="mon-fri",
                  hour=9, minute=20, id="premarket_sort")

    # Brain + catalyst refresh at 9:25 AM
    sched.add_job(run_brain_update, "cron", day_of_week="mon-fri",
                  hour=9, minute=25, id="brain_update")

    # News check disabled — whatsapp_signal_bot already sends RSS alerts to same Telegram chat

    # Mid-day status
    sched.add_job(run_position_status, "cron", day_of_week="mon-fri",
                  hour=13, minute=0, id="midday")

    # EOD — fires at 3:55 PM ET (5 min before close) so market orders fill same session
    sched.add_job(run_eod, "cron", day_of_week="mon-fri",
                  hour=MARKET_CLOSE_HOUR - 1, minute=55, id="eod")

    # Weekly summary disabled — whatsapp_signal_bot sends weekly performance to same Telegram chat

    logger.info(
        "Bot v3 ready:\n"
        "  Filters : trend(20MA) + sector ETF + earnings block + Trump/Congress/SEC\n"
        f"  Target  : ${DAILY_PROFIT_TARGET:,.0f}/day → Grade A only after\n"
        f"  Stops   : -${AUTO_MAX_DAILY_LOSS:,.0f} daily | time stop {TIME_STOP_MINUTES}min\n"
        "  Brain   : adaptive confidence + position size + weekly halt\n"
        "  9:25 AM : regime + Trump + Congress briefing sent to Telegram\n"
        "  Every 30min: RSS news (Reuters/MarketWatch/CNBC) + Truth Social alerts"
    )

    if is_market_open():
        run_orb_scan()

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
