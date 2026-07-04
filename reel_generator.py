"""
reel_generator.py — Daily Instagram Reel for Alpaca Paper Bot.

Generates a 1080×1080 MP4 (~30 sec) showing:
  • Animated P&L curve for the day
  • Top-3 trade cards with stock chart + entry/exit markings
  • Final scorecard
  • Full voice commentary via Microsoft Edge TTS

Saves to Desktop/Trading_Reels/YYYY-MM-DD.mp4  AND sends via Telegram.
Called from run_eod() in main.py after send_eod_summary().
"""
import asyncio
import io
import logging
import os
import subprocess
import tempfile
from datetime import datetime, date

import numpy as np
import pytz
import requests
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)
_ET = pytz.timezone("America/New_York")

# ── Video constants ────────────────────────────────────────────────────────────
PX  = 1080          # 1080 × 1080 square (Instagram post / Reel)
DPI = 100
FS  = PX / DPI      # 10.8 inches — at 100 DPI gives exactly 1080 px
FPS = 24

# ── Brand colors (same palette as V4 PDF) ─────────────────────────────────────
BG    = "#0d1b2a"
CARD  = "#0f2035"
BLUE  = "#1e40af"
TEAL  = "#0891b2"
GREEN = "#22c55e"
RED   = "#ef4444"
GOLD  = "#f59e0b"
WHITE = "#f1f5f9"
LGRAY = "#94a3b8"
DGRAY = "#334155"
DRED  = "#991b1b"


# ══════════════════════════════════════════════════════════════════════════════
# Frame helpers
# ══════════════════════════════════════════════════════════════════════════════

def _new_fig():
    fig = plt.figure(figsize=(FS, FS), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def _to_rgb(fig) -> np.ndarray:
    """Render a matplotlib figure → H × W × 3 uint8 numpy array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # buffer_rgba() works in all modern matplotlib versions (3.8+)
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return arr[:, :, :3]  # RGBA → RGB


def _hold(frame: np.ndarray, n: int) -> list:
    return [frame] * n


def _fade(frame_a: np.ndarray, frame_b: np.ndarray, n: int) -> list:
    """Linear cross-fade between two frames over n steps."""
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        frames.append((frame_a * (1 - t) + frame_b * t).astype(np.uint8))
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_fill_times(symbols: list) -> dict:
    """Pull actual fill times from Alpaca for today's symbols."""
    try:
        from alpaca_trader import get_recent_orders
        today = date.today()
        times = {}
        for o in get_recent_orders(status="filled", limit=200):
            sym = o.symbol
            if sym in symbols and sym not in times and o.filled_at:
                t = o.filled_at
                if hasattr(t, "date") and t.date() == today:
                    times[sym] = t.astimezone(_ET)
        return times
    except Exception as e:
        logger.warning(f"Could not get fill times: {e}")
        return {}


def _prepare_trades(signals_today: list) -> list:
    """Return enriched, sorted list of trades that actually filled today."""
    symbols = [s["ticker"] for s in signals_today if s.get("ticker")]
    fill_times = _get_fill_times(symbols)

    trades = []
    for sig in signals_today:
        ticker  = sig.get("ticker", "")
        pnl     = sig.get("pnl", 0) or 0
        fill_px = sig.get("fill_px", 0) or 0
        if fill_px == 0:
            continue  # signal never filled
        ft = fill_times.get(ticker)
        if ft is None:
            now = datetime.now(_ET)
            ft  = now.replace(hour=10, minute=15, second=0, microsecond=0)
        trades.append({
            "ticker":      ticker,
            "direction":   sig.get("direction", "BUY"),
            "grade":       sig.get("grade", "B"),
            "pnl":         pnl,
            "fill_px":     fill_px,
            "stop":        sig.get("stop", 0),
            "r1":          sig.get("r1", 0),
            "r2":          sig.get("r2", 0),
            "signal_type": sig.get("signal_type", "ORB"),
            "reasons":     sig.get("reasons", []),
            "confidence":  sig.get("confidence", 0),
            "cat_score":   sig.get("cat_score", 0),
            "result":      "WIN" if pnl > 10 else ("LOSS" if pnl < -10 else "SCRATCH"),
            "fill_time":   ft,
        })
    trades.sort(key=lambda x: x["fill_time"])
    return trades


def _get_bars(ticker: str):
    """Download today's 5-min bars; return DataFrame or None."""
    try:
        df = yf.download(ticker, period="1d", interval="5m",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SCENE 1 — Intro title card (60 frames = 2.5 s)
# ══════════════════════════════════════════════════════════════════════════════

def scene_intro(today_str: str, day_pnl: float, n_trades: int) -> list:
    pnl_color = GREEN if day_pnl >= 0 else RED
    pnl_str   = f"+${day_pnl:,.0f}" if day_pnl >= 0 else f"-${abs(day_pnl):,.0f}"

    fig = _new_fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Accent bar top
    ax.add_patch(mpatches.Rectangle((0, 0.93), 1, 0.07, color=TEAL, zorder=2))
    ax.text(0.5, 0.965, "ALPACA  PAPER  BOT", ha="center", va="center",
            fontsize=22, fontweight="bold", color=WHITE, zorder=3)

    # Bot logo area
    ax.add_patch(FancyBboxPatch((0.35, 0.60), 0.30, 0.27,
                                boxstyle="round,pad=0.02",
                                facecolor=CARD, edgecolor=TEAL, linewidth=2))
    ax.text(0.50, 0.76, "BOT", ha="center", va="center",
            fontsize=44, fontweight="bold", color=TEAL)
    ax.text(0.50, 0.63, "Auto Trader", ha="center", va="center",
            fontsize=13, color=LGRAY)

    # Date
    ax.text(0.50, 0.53, today_str, ha="center", va="center",
            fontsize=18, color=WHITE, fontweight="bold")

    # Day P&L preview
    ax.text(0.50, 0.42, "Day P&L", ha="center", va="center",
            fontsize=13, color=LGRAY)
    ax.text(0.50, 0.33, pnl_str, ha="center", va="center",
            fontsize=46, color=pnl_color, fontweight="bold")

    # Trades badge
    ax.add_patch(FancyBboxPatch((0.38, 0.22), 0.24, 0.08,
                                boxstyle="round,pad=0.015",
                                facecolor=BLUE, edgecolor="none"))
    ax.text(0.50, 0.26, f"{n_trades} Trades Today", ha="center", va="center",
            fontsize=13, color=WHITE, fontweight="bold")

    # Footer
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.07, color=DGRAY, zorder=2))
    ax.text(0.5, 0.035, "Paper Trading Only  •  Week 2", ha="center", va="center",
            fontsize=11, color=LGRAY, zorder=3)

    base = _to_rgb(fig)

    # Fade in from black
    black = np.zeros_like(base)
    frames = _fade(black, base, 20) + _hold(base, 40)
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# SCENE 2 — Animated P&L curve (210 frames = 8.75 s)
# ══════════════════════════════════════════════════════════════════════════════

def scene_pnl_curve(trades: list, day_pnl: float) -> list:
    """Build up the cumulative P&L line trade by trade."""
    if not trades:
        # No trades — still show $0 flat line
        return _scene_no_trades()

    # Build timeline: (fill_time, cumulative_pnl)
    points = []
    cumulative = 0.0
    for t in trades:
        cumulative += t["pnl"]
        points.append((t["fill_time"], cumulative))

    # Convert times to float hours since 9:30 AM for plotting
    def _to_h(dt):
        return dt.hour + dt.minute / 60 + dt.second / 3600

    xs = [9.5] + [_to_h(p[0]) for p in points] + [16.0]
    ys = [0.0]  + [p[1] for p in points]         + [day_pnl]

    y_abs_max = max(abs(min(ys)), abs(max(ys)), 100)
    y_pad     = y_abs_max * 0.3

    total_frames = 210
    n_pts = len(xs)
    frames = []

    for fi in range(total_frames):
        # How far along are we? (0→1)
        progress = fi / (total_frames - 1)
        # Which segment are we drawing through?
        # We animate the line building from left to right
        x_end = xs[0] + progress * (xs[-1] - xs[0])

        fig = _new_fig()
        ax  = fig.add_axes([0.12, 0.18, 0.83, 0.68])
        ax.set_facecolor(CARD)
        fig.patch.set_facecolor(BG)

        # Build partial xs/ys up to x_end
        px_, py_ = [xs[0]], [ys[0]]
        for i in range(1, n_pts):
            if xs[i] <= x_end:
                px_.append(xs[i])
                py_.append(ys[i])
            else:
                # Interpolate
                frac = (x_end - xs[i-1]) / (xs[i] - xs[i-1])
                px_.append(x_end)
                py_.append(ys[i-1] + frac * (ys[i] - ys[i-1]))
                break

        # Fill under curve
        pos_xs = []; pos_ys = []; neg_xs = []; neg_ys = []
        for xi, yi in zip(px_, py_):
            if yi >= 0:
                pos_xs.append(xi); pos_ys.append(yi)
            else:
                neg_xs.append(xi); neg_ys.append(yi)

        ax.fill_between(px_, 0, py_,
                        where=[y >= 0 for y in py_],
                        color=GREEN, alpha=0.18, interpolate=True)
        ax.fill_between(px_, 0, py_,
                        where=[y < 0 for y in py_],
                        color=RED, alpha=0.18, interpolate=True)

        # Main line
        line_color = GREEN if py_[-1] >= 0 else RED
        ax.plot(px_, py_, color=line_color, linewidth=2.5, zorder=4)

        # Trade dots (filled points)
        for i, (tx, ty) in enumerate(zip(xs[1:-1], ys[1:-1])):
            if tx <= x_end:
                t = trades[i]
                dc = GREEN if t["pnl"] > 0 else (RED if t["pnl"] < -10 else GOLD)
                ax.scatter([tx], [ty], color=dc, s=80, zorder=5, edgecolors=WHITE, linewidths=1)

        # Zero line
        ax.axhline(0, color=LGRAY, linewidth=0.8, alpha=0.5, zorder=2)

        # Axes formatting
        ax.set_xlim(9.3, 16.2)
        ax.set_ylim(-y_abs_max - y_pad, y_abs_max + y_pad)
        ax.set_xticks([9.5, 11, 12.5, 14, 15.5])
        ax.set_xticklabels(["9:30", "11am", "12:30", "2pm", "3:30"], color=LGRAY, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+,.0f}"))
        ax.tick_params(axis="y", colors=LGRAY, labelsize=9)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=DGRAY, alpha=0.4, linewidth=0.5)

        # Running P&L label (top right)
        cur_pnl = py_[-1]
        ax.text(0.98, 0.97, f"${cur_pnl:+,.0f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=22, fontweight="bold",
                color=GREEN if cur_pnl >= 0 else RED)

        # Title bar
        ax_title = fig.add_axes([0, 0.90, 1, 0.10])
        ax_title.set_facecolor(TEAL)
        ax_title.axis("off")
        ax_title.text(0.5, 0.5, "DAY  P&L  TIMELINE",
                      ha="center", va="center", fontsize=17,
                      fontweight="bold", color=WHITE)

        # Footer
        ax_foot = fig.add_axes([0, 0, 1, 0.06])
        ax_foot.set_facecolor(DGRAY)
        ax_foot.axis("off")
        ax_foot.text(0.5, 0.5, "Alpaca Paper Bot  •  Live Paper Trading",
                     ha="center", va="center", fontsize=10, color=LGRAY)

        frames.append(_to_rgb(fig))

    return frames


def _scene_no_trades() -> list:
    fig = _new_fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis("off"); ax.set_facecolor(BG)
    ax.text(0.5, 0.5, "No trades today\n(Market closed or halted)",
            ha="center", va="center", fontsize=22, color=LGRAY)
    return _hold(_to_rgb(fig), 210)


# ══════════════════════════════════════════════════════════════════════════════
# SCENE 3 — Trade card with 5-min chart (100 frames = ~4 s each)
# ══════════════════════════════════════════════════════════════════════════════

def scene_trade_card(trade: dict, bars) -> list:
    ticker    = trade["ticker"]
    direction = trade["direction"]
    grade     = trade["grade"]
    pnl       = trade["pnl"]
    fill_px   = trade["fill_px"]
    result    = trade["result"]

    result_color = GREEN if result == "WIN" else (RED if result == "LOSS" else GOLD)
    dir_color    = GREEN if direction == "BUY" else RED
    pnl_str      = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
    arrow        = "▲" if direction == "BUY" else "▼"

    fig = _new_fig()

    # ── Title bar ──────────────────────────────────────────────────────────────
    ax_title = fig.add_axes([0, 0.90, 1, 0.10])
    ax_title.set_facecolor(BLUE)
    ax_title.axis("off")
    ax_title.text(0.07, 0.5, f"{arrow}  {ticker}  {direction}",
                  ha="left", va="center", fontsize=20,
                  fontweight="bold", color=dir_color)
    ax_title.text(0.93, 0.5, f"Grade {grade}",
                  ha="right", va="center", fontsize=14, color=GOLD)

    # ── Stock chart ────────────────────────────────────────────────────────────
    ax_c = fig.add_axes([0.05, 0.35, 0.90, 0.52])
    ax_c.set_facecolor(CARD)

    if bars is not None and not bars.empty:
        # Draw candlestick bars manually
        opens  = bars["Open"].values.flatten()
        closes = bars["Close"].values.flatten()
        highs  = bars["High"].values.flatten()
        lows   = bars["Low"].values.flatten()
        xs     = np.arange(len(opens))

        for i in range(len(opens)):
            up   = closes[i] >= opens[i]
            col  = "#22c55e" if up else "#ef4444"
            body_lo = min(opens[i], closes[i])
            body_hi = max(opens[i], closes[i])
            ax_c.bar(i, body_hi - body_lo, bottom=body_lo,
                     color=col, width=0.7, alpha=0.85, zorder=3)
            ax_c.plot([i, i], [lows[i], highs[i]],
                      color=col, linewidth=0.9, zorder=2, alpha=0.7)

        price_range = highs.max() - lows.min()
        if price_range < 0.01:
            price_range = fill_px * 0.02

        stop = trade.get("stop", 0)
        r1   = trade.get("r1", 0)
        r2   = trade.get("r2", 0)

        # Dynamic y-range that includes all key levels
        all_levels = [lows.min(), highs.max()]
        for lvl in [stop, r1, r2, fill_px]:
            if lvl and lvl > 0:
                all_levels.append(lvl)
        y_lo = min(all_levels)
        y_hi = max(all_levels)
        y_pad = (y_hi - y_lo) * 0.18 or fill_px * 0.02

        # Entry line (gold dashed)
        ax_c.axhline(fill_px, color=GOLD, linewidth=1.6,
                     linestyle="--", alpha=0.95, zorder=5)
        ax_c.text(len(opens) + 0.3, fill_px, f"ENTRY ${fill_px:.2f}",
                  va="center", color=GOLD, fontsize=8, fontweight="bold")

        # Stop loss line (red)
        if stop and stop > 0:
            ax_c.axhline(stop, color=RED, linewidth=1.4,
                         linestyle=":", alpha=0.9, zorder=5)
            ax_c.text(len(opens) + 0.3, stop, f"STOP ${stop:.2f}",
                      va="center", color=RED, fontsize=7.5)

        # Target 1 line (light green)
        if r1 and r1 > 0:
            ax_c.axhline(r1, color="#86efac", linewidth=1.4,
                         linestyle="-.", alpha=0.9, zorder=5)
            ax_c.text(len(opens) + 0.3, r1, f"T1 ${r1:.2f}",
                      va="center", color="#86efac", fontsize=7.5)

        # Target 2 line (bright green)
        if r2 and r2 > 0:
            ax_c.axhline(r2, color=GREEN, linewidth=1.6,
                         linestyle="-", alpha=0.9, zorder=5)
            ax_c.text(len(opens) + 0.3, r2, f"T2 ${r2:.2f}",
                      va="center", color=GREEN, fontsize=7.5, fontweight="bold")

        ax_c.set_xlim(-1, len(opens) + 5)
        ax_c.set_ylim(y_lo - y_pad, y_hi + y_pad)

    else:
        # No chart data — show placeholder text
        ax_c.text(0.5, 0.5, f"${fill_px:.2f}\n(chart unavailable)",
                  ha="center", va="center", fontsize=16,
                  color=LGRAY, transform=ax_c.transAxes)

    ax_c.spines[:].set_visible(False)
    ax_c.tick_params(colors=LGRAY, labelsize=8, length=0)
    ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax_c.grid(axis="y", color=DGRAY, alpha=0.35, linewidth=0.5)

    # ── Result card ───────────────────────────────────────────────────────────
    ax_r = fig.add_axes([0.05, 0.06, 0.90, 0.27])
    ax_r.set_facecolor(CARD)
    ax_r.axis("off")

    # Result badge
    ax_r.add_patch(FancyBboxPatch((0.03, 0.55), 0.22, 0.38,
                                  boxstyle="round,pad=0.02",
                                  facecolor=result_color,
                                  edgecolor="none",
                                  transform=ax_r.transAxes))
    ax_r.text(0.14, 0.74, result, ha="center", va="center",
              fontsize=18, fontweight="bold", color=WHITE,
              transform=ax_r.transAxes)

    # P&L
    ax_r.text(0.38, 0.74, pnl_str, ha="center", va="center",
              fontsize=36, fontweight="bold",
              color=GREEN if pnl >= 0 else RED,
              transform=ax_r.transAxes)

    # Fill price
    ax_r.text(0.75, 0.80, f"Filled @ ${fill_px:.2f}", ha="center",
              fontsize=11, color=LGRAY, transform=ax_r.transAxes)
    ax_r.text(0.75, 0.55, f"Grade {grade}  •  {direction}",
              ha="center", fontsize=11, color=LGRAY,
              transform=ax_r.transAxes)

    # Divider
    ax_r.axhline(0.48, color=DGRAY, linewidth=0.7, xmin=0.02, xmax=0.98)

    ax_r.text(0.50, 0.22, "Paper Trading Only — Not Financial Advice",
              ha="center", va="center", fontsize=9, color=LGRAY,
              transform=ax_r.transAxes)

    frame = _to_rgb(fig)
    return _hold(frame, 100)


# ══════════════════════════════════════════════════════════════════════════════
# SCENE 4 — Final scorecard (180 frames = 7.5 s)
# ══════════════════════════════════════════════════════════════════════════════

def scene_scorecard(trades: list, day_pnl: float, account_val: float,
                    today_str: str) -> list:
    n_trades = len(trades)
    n_wins   = sum(1 for t in trades if t["result"] == "WIN")
    n_losses = sum(1 for t in trades if t["result"] == "LOSS")
    wr       = (n_wins / n_trades * 100) if n_trades else 0
    pnl_str  = f"+${day_pnl:,.0f}" if day_pnl >= 0 else f"-${abs(day_pnl):,.0f}"
    pnl_color = GREEN if day_pnl >= 0 else RED

    fig = _new_fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Top accent
    ax.add_patch(mpatches.Rectangle((0, 0.93), 1, 0.07, color=TEAL, zorder=2))
    ax.text(0.5, 0.965, "END  OF  DAY  RESULTS",
            ha="center", va="center", fontsize=20,
            fontweight="bold", color=WHITE, zorder=3)

    # Date
    ax.text(0.5, 0.88, today_str, ha="center", fontsize=13, color=LGRAY)

    # Big P&L
    ax.text(0.5, 0.72, "Day P&L", ha="center", fontsize=15, color=LGRAY)
    ax.text(0.5, 0.60, pnl_str, ha="center", fontsize=60,
            fontweight="bold", color=pnl_color)

    # Divider
    ax.add_patch(mpatches.Rectangle((0.08, 0.535), 0.84, 0.002,
                                    color=DGRAY, zorder=2))

    # Stats row
    stats = [
        (f"{n_trades}", "Trades"),
        (f"{n_wins}W / {n_losses}L", "Win / Loss"),
        (f"{wr:.0f}%", "Win Rate"),
    ]
    x_positions = [0.18, 0.50, 0.82]
    for x, (val, lbl) in zip(x_positions, stats):
        ax.text(x, 0.46, val, ha="center", fontsize=22,
                fontweight="bold", color=WHITE)
        ax.text(x, 0.40, lbl, ha="center", fontsize=11, color=LGRAY)

    # Win rate bar
    bar_x, bar_y, bar_w, bar_h = 0.08, 0.34, 0.84, 0.025
    ax.add_patch(mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                    color=DRED if n_trades else DGRAY, zorder=2))
    ax.add_patch(mpatches.Rectangle((bar_x, bar_y), bar_w * wr / 100, bar_h,
                                    color=GREEN, zorder=3))
    ax.text(0.5, bar_y + bar_h + 0.01,
            f"Win Rate Bar  {wr:.0f}%",
            ha="center", fontsize=9, color=LGRAY)

    # Account value
    ax.add_patch(FancyBboxPatch((0.15, 0.19), 0.70, 0.12,
                                boxstyle="round,pad=0.015",
                                facecolor=CARD, edgecolor=TEAL, linewidth=1.5))
    ax.text(0.50, 0.28, "Account Value", ha="center", fontsize=11, color=LGRAY)
    ax.text(0.50, 0.215, f"${account_val:,.0f}", ha="center",
            fontsize=24, fontweight="bold", color=WHITE)

    # Trades list (compact)
    if trades:
        y_start = 0.155
        for t in trades[:4]:
            clr = GREEN if t["pnl"] >= 0 else RED
            pnl_s = f"+${t['pnl']:,.0f}" if t["pnl"] >= 0 else f"-${abs(t['pnl']):,.0f}"
            arrow = "▲" if t["direction"] == "BUY" else "▼"
            ax.text(0.12, y_start,
                    f"{arrow} {t['ticker']}  Gr.{t['grade']}",
                    fontsize=10, color=WHITE, va="center")
            ax.text(0.88, y_start, pnl_s,
                    fontsize=10, color=clr, va="center", ha="right")
            y_start -= 0.028

    # Footer
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.04, color=DGRAY, zorder=2))
    ax.text(0.5, 0.02, "Paper Trading  •  Alpaca Markets  •  Not Financial Advice",
            ha="center", va="center", fontsize=9, color=LGRAY, zorder=3)

    frame = _to_rgb(fig)
    # Fade in
    black = np.zeros_like(frame)
    return _fade(black, frame, 24) + _hold(frame, 156)


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_output_folder() -> str:
    """Return Desktop/Trading_Reels path, creating it if needed."""
    candidates = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "Trading_Reels"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Trading_Reels"),
    ]
    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            test = os.path.join(p, ".test")
            open(test, "w").close()
            os.remove(test)
            return p
        except Exception:
            continue
    # Fallback: Railway tmp folder
    fallback = "/tmp/trading_reels"
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _write_mp4(frames: list, output_path: str) -> None:
    import imageio
    with imageio.get_writer(output_path, fps=FPS, macro_block_size=1,
                            output_params=["-pix_fmt", "yuv420p", "-crf", "22"]) as w:
        for f in frames:
            w.append_data(f)
    logger.info(f"Reel saved: {output_path}  ({len(frames)} frames, "
                f"{len(frames)/FPS:.1f}s)")


def _send_telegram_video(filepath: str, caption: str) -> None:
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        token   = TELEGRAM_BOT_TOKEN
        chat_id = TELEGRAM_CHAT_ID
    except Exception:
        token   = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.error("Telegram credentials missing — cannot send reel")
        return
    url = f"https://api.telegram.org/bot{token}/sendVideo"
    try:
        with open(filepath, "rb") as f:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"video": f},
                timeout=120,
            )
        if r.ok:
            logger.info("Reel sent via Telegram")
        else:
            logger.error(f"Telegram video upload failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.error(f"Telegram video upload error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# COMMENTARY + AUDIO
# ══════════════════════════════════════════════════════════════════════════════

VOICE = "en-US-EricNeural"   # professional, clear — good for finance content


def _build_commentary(trades: list, day_pnl: float, account_val: float,
                      today_str: str) -> str:
    """Dynamically build the full voiceover script for the day's reel."""
    n        = len(trades)
    n_wins   = sum(1 for t in trades if t["result"] == "WIN")
    n_losses = sum(1 for t in trades if t["result"] == "LOSS")
    wr       = int(n_wins / n * 100) if n else 0
    day_name = datetime.now(_ET).strftime("%A")
    pnl_up   = day_pnl >= 0
    pnl_abs  = abs(day_pnl)
    pnl_word = f"up ${pnl_abs:,.0f}" if pnl_up else f"down ${pnl_abs:,.0f}"

    parts = []

    # ── Intro ──────────────────────────────────────────────────────────────────
    parts.append(
        f"Alpaca Paper Bot — {today_str}. Here is your full daily trading recap."
    )

    if n == 0:
        parts.append(
            "No trades today. The market either closed early or conditions "
            "did not meet the bot's entry filters. The bot stays disciplined "
            "and only acts on high-probability setups."
        )
    else:
        parts.append(
            f"The bot placed {n} trade{'s' if n > 1 else ''} today and "
            f"finished the session {pnl_word}. Let me walk you through every trade — "
            f"why we took it, how we managed the risk, and what happened."
        )

    # ── P&L curve narration ────────────────────────────────────────────────────
    if n > 0:
        best  = max(trades, key=lambda t: t["pnl"])
        worst = min(trades, key=lambda t: t["pnl"])
        parts.append(
            f"This chart shows the cumulative P&L building through the session. "
            f"{'Green' if day_pnl >= 0 else 'Red'} means we finished "
            f"{'above' if day_pnl >= 0 else 'below'} zero. "
            f"The best performer was {best['ticker']} at plus ${best['pnl']:,.0f}."
        )
        if n > 1 and worst["pnl"] < -10:
            parts.append(
                f"The toughest trade was {worst['ticker']}, costing ${abs(worst['pnl']):,.0f}. "
                f"The stop was hit cleanly — that is the risk management doing its job."
            )

    # ── Per-trade full narration ───────────────────────────────────────────────
    for i, t in enumerate(trades[:3]):
        ticker      = t["ticker"]
        direction   = "BUY" if t["direction"] == "BUY" else "SELL"
        dir_word    = "long" if direction == "BUY" else "short"
        grade       = t["grade"]
        fill_px     = t["fill_px"]
        stop        = t.get("stop", 0)
        r1          = t.get("r1", 0)
        r2          = t.get("r2", 0)
        pnl         = t["pnl"]
        result      = t["result"]
        signal_type = t.get("signal_type", "ORB")
        reasons     = t.get("reasons", [])
        confidence  = t.get("confidence", 0)
        cat_score   = t.get("cat_score", 0)

        trade_num = ["First", "Second", "Third"][i]

        # Grade description
        grade_desc = {
            "A": "Grade A — our highest-conviction setup. Full position size, all four criteria confirmed.",
            "B": "Grade B — solid setup, three of four criteria met. Three-quarter position size.",
            "C": "Grade C — valid breakout but fewer confirmations. Half position size with tighter targets.",
        }.get(grade, f"Grade {grade}.")

        # WHY we took it
        why_parts = []
        if reasons:
            # Clean up reason strings for speech
            clean = []
            for r in reasons[:4]:
                r = str(r).replace("✅", "").replace("🔴", "").replace("🟢", "")
                r = r.replace("  ", " ").strip()
                if r:
                    clean.append(r)
            if clean:
                why_parts.append("The reasons for entering: " + ". ".join(clean) + ".")
        if confidence:
            why_parts.append(f"Signal confidence was {confidence} percent.")
        if cat_score and cat_score != 0:
            cat_desc = "positive" if cat_score > 0 else "slightly negative but within tolerance"
            why_parts.append(f"Catalyst score was {cat_score:+d}, {cat_desc}.")

        # Entry / risk levels
        risk_per_share = abs(fill_px - stop) if stop else 0
        r1_pct = abs(r1 - fill_px) / fill_px * 100 if r1 and fill_px else 0
        r2_pct = abs(r2 - fill_px) / fill_px * 100 if r2 and fill_px else 0

        level_parts = [f"Entry at ${fill_px:.2f}."]
        if stop:
            level_parts.append(f"Stop loss at ${stop:.2f}, risking ${risk_per_share:.2f} per share.")
        if r1:
            level_parts.append(f"First target at ${r1:.2f} — that is plus {r1_pct:.1f} percent.")
        if r2:
            level_parts.append(f"Second target at ${r2:.2f} — that is plus {r2_pct:.1f} percent.")
        level_parts.append("The bot uses a two-leg bracket: half exits at target one, half rides to target two.")

        # Outcome
        if result == "WIN":
            if pnl > 500:
                outcome = f"This was a strong winner — the trade returned ${pnl:,.0f}. Both legs hit target."
            else:
                outcome = f"A winner. The trade returned ${pnl:,.0f}. Target hit, bot booked the gain."
        elif result == "LOSS":
            outcome = (
                f"The stop was hit. Trade closed for a loss of ${abs(pnl):,.0f}. "
                f"Loss was pre-defined — this is part of the strategy. "
                f"One loss does not change the overall edge."
            )
        else:
            outcome = (
                f"This one was a scratch — essentially breakeven at ${pnl:+,.0f}. "
                f"The time-based stop likely closed it before either target was reached."
            )

        # Assemble trade narration
        parts.append(
            f"{trade_num} trade — {ticker}, {signal_type} {dir_word} signal. {grade_desc}"
        )
        if why_parts:
            parts.append(" ".join(why_parts))
        parts.append(" ".join(level_parts))
        parts.append(f"Outcome: {outcome}")

    # ── Scorecard narration ────────────────────────────────────────────────────
    if n > 0:
        qual = "excellent" if wr >= 70 else ("solid" if wr >= 55 else ("mixed" if wr >= 40 else "tough"))
        parts.append(
            f"Now for the final scorecard on {day_name}. "
            f"Total P&L: {pnl_word}. "
            f"{n} trade{'s' if n != 1 else ''} placed — "
            f"{n_wins} winner{'s' if n_wins != 1 else ''}, "
            f"{n_losses} loss{'es' if n_losses != 1 else ''}. "
            f"A {qual} win rate of {wr} percent. "
            f"The paper account now stands at ${account_val:,.0f}."
        )
    else:
        parts.append(
            f"Final check for {day_name}. No trades taken today. "
            f"Account unchanged at ${account_val:,.0f}."
        )

    parts.append(
        "This is a fully automated paper trading system running live on Alpaca Markets. "
        "Every trade is paper only — no real money at risk. "
        "We are tracking week-by-week performance transparently. "
        "This is not financial advice. See you tomorrow."
    )

    return "  ".join(parts)


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


async def _tts_async(text: str, voice: str, out_path: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice).save(out_path)


def _generate_tts(text: str, out_path: str, voice: str = VOICE) -> None:
    """Generate MP3 voiceover using Microsoft Edge TTS (free, no API key)."""
    asyncio.run(_tts_async(text, voice, out_path))
    logger.info(f"TTS audio generated: {out_path}")


def _merge_audio_video(video_path: str, audio_path: str,
                       output_path: str, video_dur: float) -> None:
    """
    Combine silent video + TTS audio into final MP4.
    Audio is padded with silence if shorter than the video.
    """
    ffmpeg = _ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        # Pad audio with silence so video never gets cut short
        "-af", f"apad=whole_dur={video_dur:.3f}",
        "-t", str(video_dur),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg merge failed: {result.stderr[-500:]}")
        raise RuntimeError("ffmpeg audio merge failed")
    logger.info(f"Final reel with audio: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_reel(signals_today: list, account: dict) -> None:
    """
    Called from run_eod() in main.py.
    signals_today — same list used in send_eod_summary()
    account       — dict from get_account() with 'day_pnl', 'equity', etc.
    """
    try:
        today_str   = datetime.now(_ET).strftime("%B %d, %Y")
        date_str    = datetime.now(_ET).strftime("%Y-%m-%d")
        day_pnl     = account.get("day_pnl", 0) or 0
        account_val = float(account.get("equity", 100_000) or 100_000)

        logger.info("Reel: preparing trade data…")
        trades = _prepare_trades(signals_today)

        # Build all frames
        logger.info("Reel: rendering frames…")
        frames = []

        # Scene 1 — intro (2.5 s)
        frames += scene_intro(today_str, day_pnl, len(trades))

        # Scene 2 — P&L curve (8.75 s)
        frames += scene_pnl_curve(trades, day_pnl)

        # Scene 3 — trade cards (max 3, 4 s each)
        for trade in trades[:3]:
            bars = _get_bars(trade["ticker"])
            frames += scene_trade_card(trade, bars)

        # Scene 4 — scorecard (7.5 s)
        frames += scene_scorecard(trades, day_pnl, account_val, today_str)

        # Write silent video to temp file
        tmp      = tempfile.gettempdir()
        silent   = os.path.join(tmp, f"reel_silent_{date_str}.mp4")
        audio_mp3= os.path.join(tmp, f"reel_audio_{date_str}.mp3")
        video_dur= len(frames) / FPS

        _write_mp4(frames, silent)

        # Generate commentary audio
        logger.info("Reel: generating commentary audio...")
        commentary = _build_commentary(trades, day_pnl, account_val, today_str)
        logger.info(f"Commentary ({len(commentary)} chars):\n{commentary[:200]}...")
        _generate_tts(commentary, audio_mp3)

        # Merge audio onto video → final file
        folder      = _get_output_folder()
        output_path = os.path.join(folder, f"{date_str}.mp4")
        _merge_audio_video(silent, audio_mp3, output_path, video_dur)

        # Cleanup temp files
        for f in [silent, audio_mp3]:
            try: os.remove(f)
            except Exception: pass

        # Send via Telegram
        pnl_str = f"+${day_pnl:,.0f}" if day_pnl >= 0 else f"-${abs(day_pnl):,.0f}"
        caption = (f"Alpaca Paper Bot - {today_str}\n"
                   f"Day P&L: {pnl_str}  |  {len(trades)} trades\n"
                   f"Account: ${account_val:,.0f}")
        _send_telegram_video(output_path, caption)
        logger.info("Reel generation complete")

    except Exception as e:
        logger.error(f"Reel generation failed: {e}", exc_info=True)
