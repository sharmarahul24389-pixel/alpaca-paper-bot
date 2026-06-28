import logging
from datetime import datetime

import pytz
import yfinance as yf

from analyzer import _flatten_columns, _rsi, _macd, _atr
from config import (
    TIMEZONE, ACCOUNT_SIZE, MIN_SIGNAL_SCORE, SWING_RISK_PCT,
)
from events import detect_trump_news, detect_market_news, get_all_headlines, get_upcoming_events
from signal_generator import Signal

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)

_PM_MAX_SCORE = 9        # Gap(1) + RSI(3) + MACD(3) + EMA(2)
_PM_ATR_MULT  = 2.0      # Wider stop for pre-market low liquidity
_PM_RR        = 2.0      # 2:1 reward-to-risk

_PREMARKET_WATCHLIST = [
    # Mega-cap & high-beta (most likely to gap on news)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO",
    # Nasdaq-100 high-momentum names
    "ASML", "MU", "PANW", "ANET", "MRVL", "CRWD", "TTD", "APP", "ARM",
    "WDAY", "ZS", "FTNT", "SMCI", "MELI", "DASH", "ABNB", "TEAM", "PLTR",
    # Financials & banks
    "JPM", "BAC", "GS", "C", "MS", "COIN", "HOOD", "SOFI", "PYPL",
    # Consumer / media
    "NFLX", "DIS", "UBER", "F", "GM", "RIVN", "NKE",
    # ETFs for market tone
    "SPY", "QQQ", "IWM", "XLK", "XLE", "GLD", "SLV",
]


def get_premarket_movers(top_n: int = 5) -> tuple[list[dict], list[dict]]:
    try:
        raw = yf.download(
            _PREMARKET_WATCHLIST,
            period="2d",
            interval="1h",
            prepost=True,
            auto_adjust=True,
            progress=False,
        )
        close_df = raw["Close"]

        movers: list[dict] = []
        for ticker in close_df.columns:
            try:
                series = close_df[ticker].dropna()
                if len(series) < 2:
                    continue
                prev_close = float(series.iloc[-2])
                curr_price = float(series.iloc[-1])
                if prev_close <= 0:
                    continue
                gap_pct = (curr_price - prev_close) / prev_close * 100
                movers.append({
                    "ticker":     str(ticker),
                    "price":      round(curr_price, 2),
                    "prev_close": round(prev_close, 2),
                    "gap_pct":    round(gap_pct, 2),
                })
            except Exception:
                continue

        movers.sort(key=lambda x: x["gap_pct"], reverse=True)
        gap_ups   = [m for m in movers if m["gap_pct"] >  0.5][:top_n]
        gap_downs = sorted([m for m in movers if m["gap_pct"] < -0.5], key=lambda x: x["gap_pct"])[:top_n]
        return gap_ups, gap_downs

    except Exception as exc:
        logger.error(f"Pre-market scan failed: {exc}")
        return [], []


# -- Pre-market signal analysis ------------------------------------------------

def analyze_premarket(ticker: str, gap_pct: float) -> Signal | None:
    """RSI/MACD/EMA analysis on pre-market 30-min bars.
    Returns a Signal or None if no actionable setup found."""
    try:
        df = yf.download(
            ticker, period="5d", interval="30m",
            prepost=True, auto_adjust=True, progress=False,
        )
        df = _flatten_columns(df)
        df.dropna(inplace=True)
        if len(df) < 25:
            return None

        # Localise index to ET so we can filter pre-market bars
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert(_ET)
        else:
            df.index = df.index.tz_convert(_ET)

        now_et      = datetime.now(_ET)
        today       = now_et.date()
        pm_open     = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

        today_pm = df[
            (df.index.date == today) &
            (df.index >= pm_open) &
            (df.index < market_open)
        ]
        if today_pm.empty:
            logger.info(f"  {ticker}: no pre-market bars found today")
            return None

        close  = df["Close"]
        volume = df["Volume"]

        rsi                               = _rsi(close)
        macd_line, signal_line, histogram = _macd(close)
        ema20                             = close.ewm(span=20, adjust=False).mean()
        ema50                             = close.ewm(span=50, adjust=False).mean()
        atr                               = _atr(df)

        def s(series, idx=-1):
            return float(series.iloc[idx])

        price      = s(close)
        rsi_val    = s(rsi)
        rsi_prev   = s(rsi, -2)
        atr_val    = s(atr)
        rsi_rising = rsi_val > rsi_prev

        buy_score:    int       = 0
        sell_score:   int       = 0
        buy_reasons:  list[str] = []
        sell_reasons: list[str] = []

        # Gap catalyst (max 1 pt)
        if gap_pct > 1.5:
            buy_score += 1
            buy_reasons.append(f"Gap up {gap_pct:+.1f}% in pre-market")
        elif gap_pct < -1.5:
            sell_score += 1
            sell_reasons.append(f"Gap down {gap_pct:+.1f}% in pre-market")

        # RSI (max 3 pts)
        if rsi_val < 35:
            buy_score += 3
            buy_reasons.append(f"RSI {rsi_val:.0f} -- oversold bounce setup")
        elif rsi_val < 50 and rsi_rising:
            buy_score += 2
            buy_reasons.append(f"RSI {rsi_val:.0f} recovering, rising")
        elif 50 <= rsi_val < 70:
            buy_score += 1
            buy_reasons.append(f"RSI {rsi_val:.0f} -- bullish momentum")

        if rsi_val > 65:
            sell_score += 3
            sell_reasons.append(f"RSI {rsi_val:.0f} -- overbought")
        elif rsi_val > 50 and not rsi_rising:
            sell_score += 2
            sell_reasons.append(f"RSI {rsi_val:.0f} rolling over")
        elif 30 < rsi_val <= 50:
            sell_score += 1
            sell_reasons.append(f"RSI {rsi_val:.0f} -- bearish bias")

        # MACD (max 3 pts)
        macd_cross_up   = s(macd_line, -2) < s(signal_line, -2) and s(macd_line) > s(signal_line)
        macd_cross_down = s(macd_line, -2) > s(signal_line, -2) and s(macd_line) < s(signal_line)

        if macd_cross_up:
            buy_score += 3
            buy_reasons.append("MACD bullish crossover")
        elif s(histogram) > 0 and s(histogram) > s(histogram, -2):
            buy_score += 1
            buy_reasons.append("MACD histogram expanding bullish")

        if macd_cross_down:
            sell_score += 3
            sell_reasons.append("MACD bearish crossover")
        elif s(histogram) < 0 and s(histogram) < s(histogram, -2):
            sell_score += 1
            sell_reasons.append("MACD histogram expanding bearish")

        # EMA trend (max 2 pts)
        if s(close) > s(ema20) > s(ema50):
            buy_score += 2
            buy_reasons.append("Price > EMA20 > EMA50 -- uptrend")
        elif s(close) < s(ema20) < s(ema50):
            sell_score += 2
            sell_reasons.append("Price < EMA20 < EMA50 -- downtrend")

        sl_dist     = atr_val * _PM_ATR_MULT
        risk_amount = ACCOUNT_SIZE * SWING_RISK_PCT  # 0.5% = $500

        def _position(entry: float, sl: float) -> dict:
            risk_unit = abs(entry - sl)
            units     = max(1, int(risk_amount / risk_unit)) if risk_unit > 0.01 else 1
            return {
                "units":          units,
                "position_value": round(units * entry, 2),
                "risk_amount":    round(risk_amount, 2),
                "target_pnl":     round(risk_amount * _PM_RR, 2),
                "pct_of_account": round(units * entry / ACCOUNT_SIZE * 100, 1),
            }

        if buy_score >= MIN_SIGNAL_SCORE and buy_score > sell_score:
            entry     = price
            stop_loss = round(entry - sl_dist, 2)
            target    = round(entry + sl_dist * _PM_RR, 2)
            return Signal(
                ticker=ticker, direction="BUY", entry=entry,
                stop_loss=stop_loss, target=target, rr=_PM_RR,
                confidence=min(100, int(buy_score / _PM_MAX_SCORE * 100)),
                reasons=buy_reasons[:5], position=_position(entry, stop_loss),
            )

        if sell_score >= MIN_SIGNAL_SCORE and sell_score > buy_score:
            entry     = price
            stop_loss = round(entry + sl_dist, 2)
            target    = round(entry - sl_dist * _PM_RR, 2)
            return Signal(
                ticker=ticker, direction="SELL", entry=entry,
                stop_loss=stop_loss, target=target, rr=_PM_RR,
                confidence=min(100, int(sell_score / _PM_MAX_SCORE * 100)),
                reasons=sell_reasons[:5], position=_position(entry, stop_loss),
            )

        return None

    except Exception as exc:
        logger.error(f"Pre-market analyze failed ({ticker}): {exc}")
        return None


_PM_GAP_MIN = 1.5   # minimum gap % to consider
_PM_GAP_MAX = 4.0   # above this = too extended, send warning instead of signal


def run_premarket_signal_scan(min_confidence: int = 35) -> list[Signal]:
    """Scan for pre-market trading signals (4 AM - 9 AM ET).
    Only considers stocks with 1.5-4% gap backed by technicals.
    Stocks already >4% extended get a warning-only message (no entry signal)."""
    gap_ups, gap_downs = get_premarket_movers(top_n=10)

    actionable: list[dict] = []
    extended:   list[dict] = []

    for m in gap_ups:
        if m["gap_pct"] > _PM_GAP_MAX:
            extended.append(m)
        elif m["gap_pct"] >= _PM_GAP_MIN:
            actionable.append(m)

    for m in gap_downs:
        if m["gap_pct"] < -_PM_GAP_MAX:
            extended.append(m)
        elif m["gap_pct"] <= -_PM_GAP_MIN:
            actionable.append(m)

    # Send a single "too extended" warning for stocks that moved >4% already
    if extended:
        names = ", ".join(f"{m['ticker']} ({m['gap_pct']:+.1f}%)" for m in extended[:5])
        from notifier import send_news_alert as _alert
        _alert(
            f"⚠️ PRE-MARKET EXTENDED MOVERS — NO ENTRY\n"
            f"\n"
            f"{names}\n"
            f"\n"
            f"Move already >{_PM_GAP_MAX}% — do NOT chase.\n"
            f"Wait for market open. Look for first 5-min pullback before entering.\n"
            f"ORB signal will fire at 9:45 AM if setup holds."
        )
        logger.info(f"Extended movers (no signal): {names}")

    if not actionable:
        logger.info("Pre-market signal scan: no 1.5-4% gap candidates found")
        return []

    signals: list[Signal] = []
    for m in actionable[:8]:
        ticker = m["ticker"]
        if ticker in ("SPY", "QQQ"):
            continue
        logger.info(f"Pre-market signal: analyzing {ticker}  gap={m['gap_pct']:+.1f}%")
        sig = analyze_premarket(ticker, m["gap_pct"])
        if sig and sig.direction != "WAIT" and sig.confidence >= min_confidence:
            sig.gap_pct    = m["gap_pct"]
            sig.prev_close = m["prev_close"]
            # Warn if gap is in the 2.5-4% range — valid setup but use a limit order
            if abs(m["gap_pct"]) >= 2.5:
                sig.reasons.append(f"Already {m['gap_pct']:+.1f}% — use limit order 0.3% below ask")
            logger.info(f"  PRE-MARKET {sig.direction}  conf={sig.confidence}%")
            signals.append(sig)
        else:
            logger.info(f"  {ticker}: below threshold or no pre-market setup")

    return signals


# -- Pre-market briefing -------------------------------------------------------

def format_premarket_briefing() -> str:
    now         = datetime.now(_ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    mins_left   = max(0, int((market_open - now).total_seconds() / 60))

    gap_ups, gap_downs = get_premarket_movers()
    headlines          = get_all_headlines()
    events             = get_upcoming_events()
    trump_news         = detect_trump_news(headlines)[:2]
    macro_news         = detect_market_news(headlines)[:3]
    all_news           = trump_news + [n for n in macro_news if n not in trump_news]

    lines = [
        f"🌅 PRE-MARKET BRIEFING -- {now.strftime('%a %b %d')}",
        "",
    ]

    if gap_ups:
        lines.append("📈 Gap-Ups:")
        for m in gap_ups:
            lines.append(f"  🟢 {m['ticker']:<6}  {m['gap_pct']:+.1f}%  ${m['price']:.2f}")

    if gap_downs:
        lines.append("")
        lines.append("📉 Gap-Downs:")
        for m in gap_downs:
            lines.append(f"  🔴 {m['ticker']:<6}  {m['gap_pct']:+.1f}%  ${m['price']:.2f}")

    if all_news:
        lines.append("")
        lines.append("📰 Overnight News:")
        for h in all_news[:4]:
            lines.append(f"  • {h[:90]}")

    if events:
        lines.append("")
        lines.append("⚠️ High-Impact Events Today:")
        for e in events:
            extras = ""
            if e.get("forecast"):
                extras += f"  Fcst: {e['forecast']}"
            if e.get("previous"):
                extras += f"  Prev: {e['previous']}"
            lines.append(f"  • {e['title']} -- {e['time']}{extras}")

    lines += [
        "",
        f"⏰ Market opens in {mins_left} min" if mins_left > 0 else "⏰ Market is now open",
        f"⚡ Pre-market signals active until 9:00 AM ET",
        f"📊 Intraday signals start at 9:30 AM ET",
        f"📈 Swing signals at 10:00 AM ET",
    ]

    return "\n".join(lines)

