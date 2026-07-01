import os
from dotenv import load_dotenv

load_dotenv()

# ── Alpaca paper trading ───────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER      = os.getenv("ALPACA_PAPER", "true").lower() != "false"

# Fill monitor — check for order fills every N minutes
FILL_CHECK_INTERVAL = 5

# Auto-trade filters
AUTO_MIN_GRADE        = os.getenv("AUTO_MIN_GRADE", "B")
AUTO_MIN_CONFIDENCE   = int(os.getenv("AUTO_MIN_CONFIDENCE", "60"))
AUTO_MAX_DAILY_LOSS   = float(os.getenv("AUTO_MAX_DAILY_LOSS", "1000"))
# No hard signal count limit — brain governs quality, loss limit governs risk
AUTO_MAX_SIGNALS      = int(os.getenv("AUTO_MAX_SIGNALS", "999"))

# Daily profit target: after hitting this, only Grade A signals are taken
DAILY_PROFIT_TARGET      = float(os.getenv("DAILY_PROFIT_TARGET",      "1000"))
# Profit protection: once target is hit, halt if P&L gives back this much
PROFIT_PROTECT_DRAWDOWN  = float(os.getenv("PROFIT_PROTECT_DRAWDOWN",  "500"))
GRADE_A_ONLY_LABEL    = "A"

# Time-based stop: close flat positions after this many minutes (0 = disabled)
TIME_STOP_MINUTES     = int(os.getenv("TIME_STOP_MINUTES", "90"))

# Catalyst filters
CATALYST_HARD_SKIP_SCORE  = int(os.getenv("CATALYST_HARD_SKIP_SCORE", "-3"))   # skip signal
CATALYST_GRADE_A_SCORE    = int(os.getenv("CATALYST_GRADE_A_SCORE",  "-2"))    # require Grade A
TREND_FILTER_ENABLED      = os.getenv("TREND_FILTER_ENABLED", "true").lower() == "true"
SECTOR_FILTER_ENABLED     = os.getenv("SECTOR_FILTER_ENABLED", "true").lower() == "true"
SEC_8K_FILTER_ENABLED     = os.getenv("SEC_8K_FILTER_ENABLED", "true").lower() == "true"

# Earnings block: skip ORB/Quant within N days of earnings
EARNINGS_BLOCK_DAYS   = int(os.getenv("EARNINGS_BLOCK_DAYS", "3"))

# Correlation filter: max trades per sector per day (0 = unlimited)
MAX_SECTOR_SIGNALS    = int(os.getenv("MAX_SECTOR_SIGNALS", "2"))

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Scanner ────────────────────────────────────────────────────────────────────
TOP_MOVERS_COUNT    = 15
MIN_PRICE           = 15.0
MIN_VOLUME          = 750_000
MAX_TICKERS_TO_SCAN = 40

# ── Technical thresholds ───────────────────────────────────────────────────────
RSI_OVERSOLD             = 35
RSI_OVERBOUGHT           = 65
VOLUME_RATIO_THRESHOLD   = 2.0

# ── Risk / Reward ──────────────────────────────────────────────────────────────
RR_RATIO          = 2.0
RR_RATIO_B        = 2.0
RR_RATIO_C        = 1.5
ATR_SL_MULTIPLIER = 1.0

# ── Daily profit locks ─────────────────────────────────────────────────────────
DAILY_PROFIT_SOFT_LOCK = 700
DAILY_PROFIT_HARD_LOCK = 1000

# ── Grade-based position sizing ────────────────────────────────────────────────
RISK_GRADE_A_PLUS = 0.011
RISK_GRADE_A      = 0.010
RISK_GRADE_B      = 0.0075
RISK_GRADE_C      = 0.005

# ── Signal scoring ─────────────────────────────────────────────────────────────
MIN_SIGNAL_SCORE = 3
MIN_CONFIDENCE   = int(os.getenv("MIN_CONFIDENCE", "55"))

# ── Market hours (US Eastern) ──────────────────────────────────────────────────
MARKET_OPEN_HOUR    = 9
MARKET_OPEN_MINUTE  = 30
MARKET_CLOSE_HOUR   = 16
MARKET_CLOSE_MINUTE = 0
TIMEZONE            = "America/New_York"
INTERVAL_MINUTES    = 30

# ── Account & risk management ──────────────────────────────────────────────────
ACCOUNT_SIZE                = 100_000
RISK_PER_TRADE_PCT          = 0.01
MAX_DAILY_SIGNALS           = 3
MAX_DAILY_WATCHLIST_SIGNALS = 4

# ── Swing trade settings ───────────────────────────────────────────────────────
SWING_RISK_PCT   = 0.005
SWING_RR_RATIO   = 3.0
SWING_ATR_SL     = 2.0
MAX_SWING_SIGNALS = 2

# ── News & events ──────────────────────────────────────────────────────────────
TRUMP_KEYWORDS = [
    "trump", "tariff", "trade war", "sanctions", "executive order",
    "white house", "truth social", "trump tariff", "trump says",
]
MARKET_EVENT_KEYWORDS = [
    "federal reserve", "fed rate", "interest rate", "cpi", "inflation",
    "jobs report", "nfp", "non-farm", "gdp", "recession", "fomc",
    "powell", "rate cut", "rate hike", "debt ceiling", "treasury",
]
NEWS_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "https://finance.yahoo.com/rss/headline?s=SPY",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",   # CNBC Top News
    "https://www.cnbc.com/id/20409666/device/rss/rss.html",    # CNBC Markets
]
TRUTH_SOCIAL_RSS = "https://truthsocial.com/@realDonaldTrump.rss"

# ── Misc ───────────────────────────────────────────────────────────────────────
COMMODITY_SYMBOLS           = {}
MACRO_SYMBOLS               = {"DXY": "DX-Y.NYB", "TNX": "^TNX"}
COMMODITY_OPEN_HOUR         = 18
COMMODITY_CLOSE_HOUR        = 17
COMMODITY_DAILY_BREAK_START = 17
COMMODITY_DAILY_BREAK_END   = 18
NEWS_FEEDS                  = []
BACKTEST_DAYS               = 30
BACKTEST_LOOKAHEAD          = 20
DRY_RUN                     = True
MT5_LOGIN                   = 0
MT5_PASSWORD                = ""
MT5_SERVER                  = ""
MT5_SYMBOL_MAP              = {}
AUTO_TRADE_RISK_PCT         = 0.01
AUTO_SCAN_INTERVAL          = 30
CRYPTO_SYMBOLS              = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
