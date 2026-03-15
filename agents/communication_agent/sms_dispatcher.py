"""
sms_dispatcher.py
-----------------
Sends SMS and WhatsApp messages to field teams using Twilio.

Two functions:
  send_sms()       — sends a plain SMS to a phone number
  send_whatsapp()  — sends a WhatsApp message (uses Twilio sandbox)

Credentials are read from environment variables — never hardcoded.

Setup required:
  1. Create a free Twilio account at https://www.twilio.com/try-twilio
  2. Get your Account SID, Auth Token, and a Twilio phone number
  3. Add to your .env file:
       TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_AUTH_TOKEN=your_auth_token_here
       TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
       YOUR_PHONE_NUMBER=+91xxxxxxxxxx
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Credentials ────────────────────────────────────────────────────────────
_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID")
_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
_FROM_NUMBER   = os.getenv("TWILIO_PHONE_NUMBER")
_DEFAULT_TO    = os.getenv("YOUR_PHONE_NUMBER")


def _get_client():
    """
    Lazy-load the Twilio client so the file can be imported
    even if credentials are not yet configured.
    """
    if not all([_ACCOUNT_SID, _AUTH_TOKEN]):
        raise EnvironmentError(
            "Twilio credentials missing. Add TWILIO_ACCOUNT_SID and "
            "TWILIO_AUTH_TOKEN to your .env file."
        )
    from twilio.rest import Client
    return Client(_ACCOUNT_SID, _AUTH_TOKEN)


# ── SMS ────────────────────────────────────────────────────────────────────

def send_sms(message: str, to_number: str = None) -> dict:
    """
    Send a plain SMS message via Twilio.

    Parameters
    ----------
    message   : text content to send
    to_number : recipient phone number in E.164 format e.g. '+919876543210'
                defaults to YOUR_PHONE_NUMBER from .env

    Returns
    -------
    dict with keys:
        success : bool
        sid     : Twilio message SID (if success)
        error   : error message (if failed)
    """

    to = to_number or _DEFAULT_TO

    if not to:
        return {
            "success": False,
            "sid": None,
            "error": "No recipient number provided. Set YOUR_PHONE_NUMBER in .env"
        }

    if not _FROM_NUMBER:
        return {
            "success": False,
            "sid": None,
            "error": "TWILIO_PHONE_NUMBER not set in .env"
        }

    try:
        client = _get_client()
        msg = client.messages.create(
            body=message,
            from_=_FROM_NUMBER,
            to=to
        )
        print(f"[SMS] Sent to {to} — SID: {msg.sid}")
        return {"success": True, "sid": msg.sid, "error": None}

    except Exception as e:
        print(f"[SMS] Failed to send: {e}")
        return {"success": False, "sid": None, "error": str(e)}


# ── WhatsApp ───────────────────────────────────────────────────────────────

def send_whatsapp(message: str, to_number: str = None) -> dict:
    """
    Send a WhatsApp message via Twilio sandbox.

    Note: The recipient must have joined the Twilio WhatsApp sandbox first.
    Instructions: https://www.twilio.com/docs/whatsapp/sandbox

    Parameters
    ----------
    message   : text content to send
    to_number : recipient number e.g. '+919876543210'
                defaults to YOUR_PHONE_NUMBER from .env

    Returns
    -------
    dict with keys:
        success : bool
        sid     : Twilio message SID (if success)
        error   : error message (if failed)
    """

    to = to_number or _DEFAULT_TO

    if not to:
        return {
            "success": False,
            "sid": None,
            "error": "No recipient number provided."
        }

    try:
        client = _get_client()
        msg = client.messages.create(
            body=message,
            from_="whatsapp:+14155238886",   # Twilio sandbox number
            to=f"whatsapp:{to}"
        )
        print(f"[WhatsApp] Sent to {to} — SID: {msg.sid}")
        return {"success": True, "sid": msg.sid, "error": None}

    except Exception as e:
        print(f"[WhatsApp] Failed to send: {e}")
        return {"success": False, "sid": None, "error": str(e)}