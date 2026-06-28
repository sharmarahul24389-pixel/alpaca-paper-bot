import calendar
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import feedparser
import pytz
import requests

from config import TIMEZONE, TRUMP_KEYWORDS, MARKET_EVENT_KEYWORDS, NEWS_RSS_FEEDS, TRUTH_SOCIAL_RSS

logger = logging.getLogger(__name__)
_ET = pytz.timezone(TIMEZONE)


# ── Economic calendar ──────────────────────────────────────────────────────────

def get_upcoming_events() -> list[dict]:
    """Fetch high-impact USD events in the next 48 hours from Forex Factory."""
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()

        now    = datetime.now(_ET)
        cutoff = now + timedelta(hours=48)
        events = []

        for e in data:
            if e.get("currency") != "USD" or e.get("impact") != "High":
                continue
            try:
                dt = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(_ET)
                if now <= dt <= cutoff:
                    events.append({
                        "title":    e.get("title", ""),
                        "time":     dt.strftime("%a %b %d %I:%M %p ET"),
                        "forecast": e.get("forecast", "N/A"),
                        "previous": e.get("previous", "N/A"),
                    })
            except Exception:
                continue

        logger.info(f"Upcoming high-impact events: {len(events)}")
        return events[:5]
    except Exception as exc:
        logger.warning(f"Economic calendar failed: {exc}")
        return []


def format_events_warning(events: list[dict]) -> str:
    if not events:
        return ""
    lines = ["⚠️ High-impact events next 48h:"]
    for e in events:
        lines.append(f"  • {e['title']} — {e['time']}")
    return "\n".join(lines)


# ── News fetcher ───────────────────────────────────────────────────────────────

def _fetch_with_doh_fallback(url: str, timeout: int = 8) -> requests.Response | None:
    """GET with automatic Google DoH fallback when Railway DNS fails."""
    from urllib.parse import urlparse
    import urllib3
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        return requests.get(url, timeout=timeout, headers=headers)
    except requests.exceptions.ConnectionError as exc:
        if "Failed to resolve" not in str(exc) and "NameResolution" not in str(exc):
            raise
    hostname = urlparse(url).hostname
    try:
        doh = requests.get(
            "https://dns.google/resolve",
            params={"name": hostname, "type": "A"},
            timeout=5, headers=headers,
        )
        ip = next(
            (a["data"] for a in doh.json().get("Answer", []) if a.get("type") == 1),
            None,
        )
        if not ip:
            return None
        ip_url = url.replace(f"://{hostname}", f"://{ip}", 1)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(
            ip_url, timeout=timeout,
            headers={**headers, "Host": hostname},
            verify=False,
        )
        logger.info(f"RSS DoH fallback: {hostname} → {ip}  HTTP {resp.status_code}")
        return resp
    except Exception as exc2:
        logger.warning(f"RSS DoH fallback failed ({hostname}): {exc2}")
        return None


def _fetch_rss_headlines(url: str) -> list[str]:
    try:
        resp = _fetch_with_doh_fallback(url, timeout=8)
        if resp is None:
            return []
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        return [item.findtext("title", "").strip() for item in root.iter("item")][:15]
    except Exception as exc:
        logger.warning(f"RSS failed ({url}): {exc}")
        return []


def get_all_headlines() -> list[str]:
    seen: set[str] = set()
    headlines: list[str] = []
    for feed in NEWS_RSS_FEEDS:
        for h in _fetch_rss_headlines(feed):
            key = h.lower()
            if h and key not in seen:
                seen.add(key)
                headlines.append(h)
    return headlines[:30]


def detect_trump_news(headlines: list[str]) -> list[str]:
    return [h for h in headlines if any(kw in h.lower() for kw in TRUMP_KEYWORDS)][:3]


def detect_market_news(headlines: list[str]) -> list[str]:
    return [h for h in headlines if any(kw in h.lower() for kw in MARKET_EVENT_KEYWORDS)][:3]


def get_truth_social_posts(since_hours: float = 48.0) -> list[dict]:
    """Fetch Trump Truth Social posts from public RSS. Returns ALL posts — no keyword filter.
    Uses requests so we can set browser headers (feedparser alone gets blocked by Cloudflare).
    since_hours is a soft safety net; the caller's dedup dict is the real duplicate guard."""
    try:
        resp = requests.get(
            TRUTH_SOCIAL_RSS,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            },
            timeout=15,
        )
        logger.info(f"Truth Social HTTP {resp.status_code} — {len(resp.content)} bytes")
        if resp.status_code != 200:
            logger.warning(f"Truth Social feed returned HTTP {resp.status_code}")
            return []

        feed = feedparser.parse(resp.content)
        if feed.bozo and not feed.entries:
            logger.warning(f"Truth Social feed parse error: {feed.bozo_exception}")
            return []

        if not feed.entries:
            logger.info("Truth Social: feed parsed but no entries found")
            return []

        cutoff = datetime.now(pytz.utc) - timedelta(hours=since_hours)
        posts: list[dict] = []

        for entry in feed.entries[:30]:
            text = (entry.get("title") or entry.get("summary") or "").strip()
            if not text:
                continue

            pub = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime.fromtimestamp(
                    calendar.timegm(entry.published_parsed), tz=pytz.utc
                )

            if pub and pub < cutoff:
                continue

            posts.append({
                "text": text[:400],
                "link": entry.get("link", ""),
                "published": pub,
            })

        logger.info(f"Truth Social: {len(posts)} post(s) within {since_hours}h window, {len(feed.entries)} total in feed")
        return posts
    except Exception as exc:
        logger.warning(f"Truth Social fetch failed: {exc}")
        return []


def format_truth_social_alert(post: dict) -> str:
    now = datetime.now(_ET).strftime("%I:%M %p ET")
    text = post["text"]
    return "\n".join([
        "🇺🇸 TRUMP — TRUTH SOCIAL",
        "",
        f'  "{text}"',
        "",
        f"⏰ {now}",
        "⚠️ Check for market impact: tariffs / trade / sanctions",
    ])


def format_news_alert(trump_news: list[str], market_news: list[str]) -> str | None:
    if not trump_news and not market_news:
        return None

    lines = ["📰 MARKET ALERT"]
    if trump_news:
        lines.append("")
        lines.append("🇺🇸 Trump / Political:")
        for h in trump_news:
            lines.append(f"  • {h[:100]}")
    if market_news:
        lines.append("")
        lines.append("📊 Macro / Fed:")
        for h in market_news:
            lines.append(f"  • {h[:100]}")

    now = datetime.now(_ET).strftime("%I:%M %p ET")
    lines.append(f"\n⏰ {now}")
    return "\n".join(lines)
