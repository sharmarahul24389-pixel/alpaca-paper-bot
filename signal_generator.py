"""
Signal Generator v10 — ORB-Momentum Hybrid (Quant layer in quant_signals.py)

Scoring (max 27 pts):
  ORB breakout + volume      6 pts  -- structural catalyst with participation
  ORB breakout (no volume)   3 pts  -- structural only (half credit)
  PDH/PDL level break        2 pts  -- institutional reference
  VWAP bands (trend-aware)   2 pts  -- trend continuation vs counter-trend logic
  Sector alignment           3 pts  -- ETF trending with trade
  Relative Strength vs SPY   3 pts  -- stock outperforming / underperforming market
  4-Hour EMA trend           2 pts  -- medium-term momentum filter
  EMA 20/50 (30-min)         2 pts  -- short-term trend confirmation
  RSI                        1 pt   -- extreme readings only
  Volume surge               2 pts  -- participation confirmation
  Daily trend                2 pts  -- higher-timeframe context
  Options flow               1 pt   -- sentiment confirmation
  Market structure (HH/HL)   1 pt   -- price action trend confirmation (NEW)
  MACD                       removed (lagging, low predictive value)

Grade tiers:
  A+: ORB + sector + RS + strong vol (≥2×) + daily trend + options → 1.1% risk
  A : ORB + sector + RS + strong vol (≥2×)                          → 1.0% risk
  B : 3 of 4 primary (normal vol ≥1.5×)                             → 0.75% risk
  C : fewer                                                          → 0.5% risk

Counter-sector penalty : -3 pts
Earnings protection    : block intraday if earnings TODAY, block swing if <=10 days away
"""
from dataclasses import dataclass, field
from datetime import datetime

import pytz

from config import (
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    VOLUME_RATIO_THRESHOLD,
    RR_RATIO,
    RR_RATIO_B,
    RR_RATIO_C,
    ATR_SL_MULTIPLIER,
    MIN_SIGNAL_SCORE,
)
from position_sizer import calculate_position

_ET        = pytz.timezone("America/New_York")
_MAX_SCORE = 27


@dataclass
class Signal:
    ticker:     str
    direction:  str
    entry:      float
    stop_loss:  float
    target:     float
    rr:         float
    confidence: int
    grade:      str = "C"
    reasons:    list[str] = field(default_factory=list)
    position:   dict      = field(default_factory=dict)
    context:    dict      = field(default_factory=dict)  # market/sector context populated by caller


def _calc_grade(orb_hit: bool, sector_aligned: bool,
                rs_confirmed: bool, strong_vol: bool, vol_surge: bool,
                trend_aligned: bool = False, options_aligned: bool = False) -> str:
    """
    A+: all 4 primary (strong vol ≥2×) + daily trend + options flow  → 1.1%
    A : all 4 primary (strong vol ≥2×)                                → 1.0%
    B : 3 of 4 primary (normal vol ≥1.5×)                             → 0.75%
    C : fewer                                                          → 0.5%
    """
    a_score = sum([orb_hit, sector_aligned, rs_confirmed, strong_vol])
    if a_score == 4:
        return "A"   # A+ merged into A — backtest showed A+ had negative EV (-$183/signal)
    b_score = sum([orb_hit, sector_aligned, rs_confirmed, vol_surge])
    if b_score >= 3:
        return "B"
    return "C"


def generate_signal(analysis: dict) -> Signal:
    ticker    = analysis["ticker"]
    price     = analysis["price"]
    atr       = analysis["atr"]
    rsi       = analysis["rsi"]
    vol_ratio = analysis["volume_ratio"]

    # Context
    daily_trend      = analysis.get("daily_trend",       "NEUTRAL")
    market_regime    = analysis.get("market_regime",     "NEUTRAL")
    options_sent     = analysis.get("options_sentiment", "NEUTRAL")
    sector_trend     = analysis.get("sector_trend",      "NEUTRAL")
    sector_etf       = analysis.get("sector_etf",        "?")
    orb              = analysis.get("orb")
    pdh              = analysis.get("pdh")
    pdl              = analysis.get("pdl")
    vb               = analysis.get("vwap_bands",        {})
    h4_trend         = analysis.get("h4_trend",          "NEUTRAL")
    spy_day_pct      = analysis.get("spy_day_pct",       0.0)
    day_pct          = analysis.get("day_pct",           0.0)
    days_to_earnings = analysis.get("days_to_earnings",  None)
    mkt_struct       = analysis.get("market_structure",  {})
    struct_bullish   = mkt_struct.get("bullish", False)
    struct_bearish   = mkt_struct.get("bearish", False)

    # Earnings protection — block intraday trade on earnings day
    if days_to_earnings is not None and days_to_earnings == 0:
        return Signal(
            ticker=ticker, direction="WAIT", entry=price,
            stop_loss=0.0, target=0.0, rr=0.0, confidence=0, grade="C",
            reasons=["Earnings today — signal blocked"],
            position={},
        )

    buy_score  = 0
    sell_score = 0
    buy_reasons:  list[str] = []
    sell_reasons: list[str] = []

    # Track grade components
    orb_hit_buy  = False
    orb_hit_sell = False
    strong_vol   = vol_ratio >= 2.0              # Grade A/A+ requires ≥2× volume
    vol_surge    = vol_ratio >= VOLUME_RATIO_THRESHOLD  # Grade B requires ≥1.5×

    # =========================================================================
    # 1. ORB BREAKOUT  (6 pts with vol, 3 pts without)
    # =========================================================================
    now_et   = datetime.now(_ET)
    # 15-min ORB ready at 9:45 AM; 30-min ORB fallback ready at 10:00 AM
    orb_ready = now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 45)
    if orb and orb_ready:
        rng_pct = orb["range"] / price * 100
        if 0.25 <= rng_pct <= 4.0:
            vol_pts = 6 if vol_ratio >= VOLUME_RATIO_THRESHOLD else 3
            vol_tag = f"vol {vol_ratio:.1f}x" if vol_ratio >= VOLUME_RATIO_THRESHOLD else "low vol"
            if price > orb["high"] * 1.001:
                extension_pct = (price - orb["high"]) / orb["high"] * 100
                if extension_pct <= 2.0:  # fresh breakout — within 2% of ORB high
                    buy_score += vol_pts
                    buy_reasons.insert(0, f"ORB breakout above ${orb['high']:.2f} ({rng_pct:.1f}% range, {vol_tag})")
                    orb_hit_buy = True
                # else: stale — price already ran >2% past ORB high, skip
            elif price < orb["low"] * 0.999:
                extension_pct = (orb["low"] - price) / orb["low"] * 100
                if extension_pct <= 2.0:  # fresh breakdown
                    sell_score += vol_pts
                    sell_reasons.insert(0, f"ORB breakdown below ${orb['low']:.2f} ({rng_pct:.1f}% range, {vol_tag})")
                    orb_hit_sell = True
                # else: stale — price already ran >2% past ORB low, skip

    # =========================================================================
    # 2. PREVIOUS DAY HIGH / LOW  (2 pts)
    # =========================================================================
    if pdh and price > pdh * 1.001:
        buy_score += 2
        buy_reasons.append(f"Breaking Previous Day High ${pdh:.2f}")
    elif pdl and price < pdl * 0.999:
        sell_score += 2
        sell_reasons.append(f"Breaking Previous Day Low ${pdl:.2f}")

    # =========================================================================
    # 3. VWAP BANDS — trend-aware (2 pts, no countertrend fading on trend days)
    # =========================================================================
    if vb:
        v  = vb["vwap"]
        u2 = vb["upper2"]
        l2 = vb["lower2"]
        strong_up   = analysis["ema20_above_ema50"] and sector_trend == "BULLISH"
        strong_down = not analysis["ema20_above_ema50"] and sector_trend == "BEARISH"

        if price >= u2:
            if strong_up:
                # Trend day: riding +2σ is continuation, not a short
                buy_score += 2
                buy_reasons.append(f"VWAP+2σ momentum continuation (${u2:.2f}) — trend day")
            else:
                sell_score += 2
                sell_reasons.append(f"Price at VWAP+2σ ${u2:.2f} — overbought")
        elif price <= l2:
            if strong_down:
                sell_score += 2
                sell_reasons.append(f"VWAP-2σ momentum continuation (${l2:.2f}) — trend day")
            else:
                buy_score += 2
                buy_reasons.append(f"Price at VWAP-2σ ${l2:.2f} — oversold bounce")
        elif price > v:
            buy_score += 1
            buy_reasons.append(f"Price above VWAP ${v:.2f}")
        else:
            sell_score += 1
            sell_reasons.append(f"Price below VWAP ${v:.2f}")

    # =========================================================================
    # 4. SECTOR ALIGNMENT  (3 pts, -3 counter-sector penalty applied later)
    # =========================================================================
    sector_aligned_buy  = sector_trend == "BULLISH"
    sector_aligned_sell = sector_trend == "BEARISH"

    if sector_aligned_buy:
        buy_score += 3
        buy_reasons.append(f"Sector {sector_etf} bullish (aligned)")
    elif sector_aligned_sell:
        sell_score += 3
        sell_reasons.append(f"Sector {sector_etf} bearish (aligned)")

    # =========================================================================
    # 5. RELATIVE STRENGTH vs blended SPY+QQQ benchmark  (3 pts)
    # =========================================================================
    # Blend matches regime detection weighting — our universe is Nasdaq-heavy
    # so a pure SPY benchmark undersells RS for tech stocks
    qqq_day_pct   = analysis.get("qqq_day_pct", spy_day_pct)
    benchmark_pct = spy_day_pct * 0.4 + qqq_day_pct * 0.6
    rs_vs_bench   = round(day_pct - benchmark_pct, 2)
    rs_buy_confirm  = rs_vs_bench >= 1.0
    rs_sell_confirm = rs_vs_bench <= -1.0

    if rs_buy_confirm:
        buy_score += 3
        buy_reasons.append(f"RS vs benchmark: +{rs_vs_bench:.1f}% (leading SPY+QQQ)")
    elif rs_sell_confirm:
        sell_score += 3
        sell_reasons.append(f"RS vs benchmark: {rs_vs_bench:.1f}% (lagging SPY+QQQ)")

    # =========================================================================
    # 6. 4-HOUR EMA TREND  (2 pts)
    # =========================================================================
    if h4_trend == "BULLISH":
        buy_score += 2
        buy_reasons.append("4H EMA: bullish (EMA20 > EMA50)")
    elif h4_trend == "BEARISH":
        sell_score += 2
        sell_reasons.append("4H EMA: bearish (EMA20 < EMA50)")

    # =========================================================================
    # 7. EMA 20/50 on 30-min  (2 pts)
    # =========================================================================
    if analysis["price_above_ema20"] and analysis["ema20_above_ema50"]:
        buy_score += 2
        buy_reasons.append("Price > EMA20 > EMA50 (uptrend)")
    elif not analysis["price_above_ema20"] and not analysis["ema20_above_ema50"]:
        sell_score += 2
        sell_reasons.append("Price < EMA20 < EMA50 (downtrend)")

    # =========================================================================
    # 8. RSI  (1 pt — extreme readings only)
    # =========================================================================
    if rsi <= RSI_OVERSOLD:
        buy_score += 1
        buy_reasons.append(f"RSI {rsi:.0f} oversold")
    elif rsi >= RSI_OVERBOUGHT:
        sell_score += 1
        sell_reasons.append(f"RSI {rsi:.0f} overbought")

    # =========================================================================
    # 9. VOLUME SURGE  (2 pts)
    # =========================================================================
    if vol_surge:
        if buy_score >= sell_score:
            buy_score += 2
            buy_reasons.append(f"Volume {vol_ratio:.1f}x avg — institutional participation")
        else:
            sell_score += 2
            sell_reasons.append(f"Volume {vol_ratio:.1f}x avg — institutional participation")

    # =========================================================================
    # 10. DAILY TREND  (2 pts)
    # =========================================================================
    if daily_trend == "BULLISH":
        buy_score += 2
        buy_reasons.append("Daily trend: BULLISH")
    elif daily_trend == "BEARISH":
        sell_score += 2
        sell_reasons.append("Daily trend: BEARISH")

    # =========================================================================
    # 11. OPTIONS FLOW  (1 pt)
    # =========================================================================
    if options_sent == "BULLISH":
        buy_score += 1
        buy_reasons.append("Options flow: bullish")
    elif options_sent == "BEARISH":
        sell_score += 1
        sell_reasons.append("Options flow: bearish")

    # =========================================================================
    # 12. MARKET STRUCTURE  (1 pt) — HH+HL (bullish) or LH+LL (bearish)
    # =========================================================================
    if struct_bullish:
        buy_score += 1
        buy_reasons.append("Market structure: higher highs + higher lows")
    elif struct_bearish:
        sell_score += 1
        sell_reasons.append("Market structure: lower highs + lower lows")

    # =========================================================================
    # COUNTER-SECTOR PENALTY  (-3 pts)
    # =========================================================================
    if sector_trend == "BEARISH" and buy_score > sell_score:
        buy_score = max(0, buy_score - 3)
    if sector_trend == "BULLISH" and sell_score > buy_score:
        sell_score = max(0, sell_score - 3)

    # =========================================================================
    # MARKET REGIME GUARD
    # =========================================================================
    regime_penalty = 2 if market_regime in ("BULLISH", "BEARISH") else 0

    # =========================================================================
    # DIRECTION DECISION + GRADE
    # =========================================================================
    sl_dist = atr * ATR_SL_MULTIPLIER

    def _grade_rr(g: str) -> float:
        """Grade-tiered reward-to-risk: A/A+ aim for 3:1, B for 2:1, C for 1.5:1."""
        if g in ("A", "A+"):
            return RR_RATIO
        if g == "B":
            return RR_RATIO_B
        return RR_RATIO_C

    # v9.3: ORB breakout required + counter-sector blocked (hard gates, not just scoring)
    if (buy_score >= MIN_SIGNAL_SCORE and buy_score > sell_score
            and orb_hit_buy                    # Change 1: ORB breakout mandatory
            and sector_trend != "BEARISH"      # Change 2: block counter-sector BUY
            and rsi < 70):                     # Change 3: don't buy very overbought stocks
        if market_regime == "BEARISH" and buy_score < MIN_SIGNAL_SCORE + regime_penalty:
            pass
        else:
            entry      = price
            stop_loss  = round(entry - sl_dist, 2)
            grade      = _calc_grade(orb_hit_buy, sector_aligned_buy,
                                     rs_buy_confirm, strong_vol, vol_surge,
                                     trend_aligned=(daily_trend == "BULLISH"),
                                     options_aligned=(options_sent == "BULLISH"))
            rr         = _grade_rr(grade)
            target     = round(entry + sl_dist * rr, 2)
            confidence = min(100, int(buy_score / _MAX_SCORE * 100))
            return Signal(
                ticker=ticker, direction="BUY", entry=entry,
                stop_loss=stop_loss, target=target, rr=rr,
                confidence=confidence, grade=grade,
                reasons=buy_reasons[:6],
                position=calculate_position(entry, stop_loss, grade),
            )

    if (sell_score >= MIN_SIGNAL_SCORE and sell_score > buy_score
            and orb_hit_sell                   # Change 1: ORB breakout mandatory
            and sector_trend != "BULLISH"      # Change 2: block counter-sector SELL
            and rsi > 30):                     # Change 3: don't short very oversold stocks
        if market_regime == "BULLISH" and sell_score < MIN_SIGNAL_SCORE + regime_penalty:
            pass
        else:
            entry      = price
            stop_loss  = round(entry + sl_dist, 2)
            grade      = _calc_grade(orb_hit_sell, sector_aligned_sell,
                                     rs_sell_confirm, strong_vol, vol_surge,
                                     trend_aligned=(daily_trend == "BEARISH"),
                                     options_aligned=(options_sent == "BEARISH"))
            rr         = _grade_rr(grade)
            target     = round(entry - sl_dist * rr, 2)
            confidence = min(100, int(sell_score / _MAX_SCORE * 100))
            return Signal(
                ticker=ticker, direction="SELL", entry=entry,
                stop_loss=stop_loss, target=target, rr=rr,
                confidence=confidence, grade=grade,
                reasons=sell_reasons[:6],
                position=calculate_position(entry, stop_loss, grade),
            )

    return Signal(
        ticker=ticker, direction="WAIT", entry=price,
        stop_loss=0.0, target=0.0, rr=0.0, confidence=0, grade="C",
        reasons=(buy_reasons + sell_reasons)[:2],
        position={},
    )
