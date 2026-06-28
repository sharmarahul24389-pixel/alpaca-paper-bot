"""
Adaptive brain: tracks performance, detects market regime, adjusts parameters.
State persists in /tmp/brain_state.json between runs (resets only on new deploy).
"""
import json, os, logging
from datetime import date, datetime
from collections import defaultdict

import yfinance as yf
import pandas as pd

log = logging.getLogger(__name__)
STATE_FILE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "brain_state.json")
_state = None  # module-level cache


# ── Persistence ───────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "version": 2,
        "trades": [],        # completed trade records
        "daily_pnl": {},     # date → realized P&L for that day
        "params": _default_params(),
        "last_regime": "UNKNOWN",
        "last_regime_ts": None,
    }

def _default_params() -> dict:
    return {
        "min_confidence_orb":   60,
        "min_confidence_quant": 60,
        "min_confidence_swing": 60,
        "position_size_mult":   1.0,   # multiplies risk_pct
        "skip_types":           [],    # e.g. ["QUANT"] when it's underperforming
    }

def load() -> dict:
    global _state
    if _state is not None:
        return _state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                _state = json.load(f)
            if _state.get("version") != 2:
                _state = _default_state()
        else:
            _state = _default_state()
    except Exception:
        _state = _default_state()
    return _state

def save(state: dict):
    global _state
    _state = state
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.warning(f"Brain save failed: {e}")


# ── Market regime ─────────────────────────────────────────────────────────────

def get_regime(force_refresh=False) -> str:
    """Returns: BULL | BEAR | VOLATILE | CHOPPY"""
    state = load()
    last_ts = state.get("last_regime_ts")
    # Cache for 30 min
    if not force_refresh and last_ts:
        age = (datetime.utcnow() - datetime.fromisoformat(last_ts)).seconds
        if age < 1800:
            return state.get("last_regime", "UNKNOWN")
    try:
        spy = yf.download("SPY",  period="25d", interval="1d",
                          auto_adjust=True, progress=False)
        vix = yf.download("^VIX", period="5d",  interval="1d",
                          auto_adjust=True, progress=False)
        spy_ret = (float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[0]) - 1) * 100
        cur_vix = float(vix["Close"].iloc[-1])

        if cur_vix > 25:
            regime = "VOLATILE"
        elif spy_ret > 3:
            regime = "BULL"
        elif spy_ret < -3:
            regime = "BEAR"
        else:
            regime = "CHOPPY"
    except Exception as e:
        log.warning(f"Regime detection failed: {e}")
        regime = state.get("last_regime", "UNKNOWN")

    state["last_regime"] = regime
    state["last_regime_ts"] = datetime.utcnow().isoformat()
    save(state)
    return regime


# ── Trade recording ───────────────────────────────────────────────────────────

def record_trade(ticker: str, signal_type: str, grade: str,
                 direction: str, pnl: float, result: str):
    """Call when a trade fully closes (fill + stop/target hit)."""
    state = load()
    state["trades"].append({
        "date":        date.today().isoformat(),
        "ticker":      ticker,
        "signal_type": signal_type,
        "grade":       grade,
        "direction":   direction,
        "pnl":         round(pnl, 2),
        "result":      result,   # WIN | LOSS | SCRATCH
        "regime":      get_regime(),
    })
    # Keep last 500 trades only
    state["trades"] = state["trades"][-500:]
    _recalc_params(state)
    save(state)
    log.info(f"Brain recorded: {ticker} {result} ${pnl:+.0f} | "
             f"type={signal_type} grade={grade}")

def update_daily_pnl(today_pnl: float):
    state = load()
    state["daily_pnl"][date.today().isoformat()] = round(today_pnl, 2)
    save(state)


# ── Adaptive parameter tuning ─────────────────────────────────────────────────

def _win_rate(trades: list, signal_type=None, grade=None, lookback=30) -> float:
    filtered = trades
    if signal_type:
        filtered = [t for t in filtered if t.get("signal_type") == signal_type]
    if grade:
        filtered = [t for t in filtered if t.get("grade") == grade]
    filtered = filtered[-lookback:]
    if len(filtered) < 5:
        return 0.55   # assume neutral until we have data
    return sum(1 for t in filtered if t.get("result") == "WIN") / len(filtered)

def _recalc_params(state: dict):
    trades  = state["trades"]
    params  = state["params"]
    regime  = state.get("last_regime", "UNKNOWN")

    # ── Position size multiplier based on overall WR ──────────────────────────
    wr_all = _win_rate(trades, lookback=30)
    if wr_all >= 0.60:
        params["position_size_mult"] = 1.25
    elif wr_all >= 0.50:
        params["position_size_mult"] = 1.0
    elif wr_all >= 0.40:
        params["position_size_mult"] = 0.75
    else:
        params["position_size_mult"] = 0.50

    # Shrink in volatile / bear markets
    if regime == "VOLATILE":
        params["position_size_mult"] = min(params["position_size_mult"], 0.75)
    elif regime == "BEAR":
        params["position_size_mult"] = min(params["position_size_mult"], 0.85)

    # ── Confidence thresholds per signal type ─────────────────────────────────
    for sig_type, key in [("ORB","min_confidence_orb"),
                           ("QUANT","min_confidence_quant"),
                           ("SWING","min_confidence_swing")]:
        wr = _win_rate(trades, signal_type=sig_type, lookback=20)
        if wr < 0.38:
            params[key] = 72     # struggling — tighten a lot
        elif wr < 0.48:
            params[key] = 65     # below break-even — tighten a bit
        else:
            params[key] = 60     # performing well — normal threshold

    # ── Skip signal types that are genuinely broken ───────────────────────────
    skip = []
    for sig_type in ["ORB", "QUANT", "SWING"]:
        wr = _win_rate(trades, signal_type=sig_type, lookback=20)
        count = len([t for t in trades[-20:] if t.get("signal_type") == sig_type])
        if count >= 10 and wr < 0.30:   # 10+ trades, <30% WR → skip
            skip.append(sig_type)
            log.warning(f"Brain: skipping {sig_type} (WR={wr:.0%} over last 20)")
    params["skip_types"] = skip

def get_params() -> dict:
    return load()["params"]


# ── Status summary ────────────────────────────────────────────────────────────

def summary() -> str:
    state  = load()
    trades = state["trades"]
    params = state["params"]
    regime = state.get("last_regime", "UNKNOWN")

    wr_all = _win_rate(trades, lookback=30)
    lines = [
        f"Brain Status | regime={regime} | WR(30)={wr_all:.0%}",
        f"  pos_mult={params['position_size_mult']:.2f}  "
        f"skip={params['skip_types'] or 'none'}",
        f"  min_conf: ORB={params['min_confidence_orb']}  "
        f"QUANT={params['min_confidence_quant']}  "
        f"SWING={params['min_confidence_swing']}",
        f"  total recorded trades: {len(trades)}",
    ]
    return "\n".join(lines)
