"""
Twilio incoming WhatsApp webhook.

Use case: forward any WhatsApp message sent to your Twilio number back
as a formatted market alert.

Railway setup (one-time):
  1. Deploy this code — Railway will expose PORT automatically.
  2. In Twilio Console → Messaging → Senders → WhatsApp Senders →
     click your number → "A message comes in":
     set to: POST https://<your-railway-domain>/webhook/whatsapp
"""
import logging

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from config import WHATSAPP_TO, TRUMP_KEYWORDS, MARKET_EVENT_KEYWORDS
from notifier import send_news_alert

logger = logging.getLogger(__name__)
app = Flask(__name__)

_MARKET_KEYWORDS = list({*TRUMP_KEYWORDS, *MARKET_EVENT_KEYWORDS, "earnings", "ipo",
                          "merger", "acquisition", "layoffs", "bankruptcy", "guidance",
                          "beat", "miss", "revenue", "profit", "loss", "upgrade", "downgrade"})


def _format_relay_alert(body: str) -> str:
    body_lower = body.lower()
    found = [kw for kw in _MARKET_KEYWORDS if kw in body_lower]

    lines = [
        "📨 FORWARDED — MARKET UPDATE",
        "",
        body[:300],
    ]
    if found:
        lines += [
            "",
            f"🔍 Keywords: {', '.join(sorted(found)[:8])}",
            "⚠️ Check for trading impact",
        ]
    return "\n".join(lines)


@app.route("/webhook/whatsapp", methods=["POST"])
def whatsapp_webhook():
    from_number = request.form.get("From", "")
    body        = request.form.get("Body", "").strip()
    twiml       = MessagingResponse()

    if from_number != WHATSAPP_TO:
        logger.info(f"Ignoring message from {from_number} (not user number)")
        return str(twiml), 200

    if not body:
        return str(twiml), 200

    logger.info(f"Received forwarded message ({len(body)} chars) from user")
    try:
        alert = _format_relay_alert(body)
        send_news_alert(alert)
    except Exception as exc:
        logger.error(f"Relay alert failed: {exc}")

    return str(twiml), 200


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200
