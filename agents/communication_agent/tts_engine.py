"""
tts_engine.py
-------------
Converts dispatch messages to audio MP3 files using gTTS
(Google Text-to-Speech). No API key required — gTTS is free.

One function:
  text_to_speech() — converts a string to an MP3 file and
                     saves it to the outputs/ folder.
"""

import os
from datetime import datetime
from gtts import gTTS


# ── Output directory ───────────────────────────────────────────────────────
_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "audio_outputs"
)
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ── TTS ────────────────────────────────────────────────────────────────────

def text_to_speech(
    message: str,
    filename: str = None,
    language: str = "en",
) -> dict:
    """
    Convert a text message to an MP3 audio file.

    Parameters
    ----------
    message  : the text to convert to speech
    filename : output filename without extension
               defaults to 'dispatch_YYYYMMDD_HHMMSS'
    language : language code — 'en' for English, 'hi' for Hindi,
               'ta' for Tamil, 'te' for Telugu, 'bn' for Bengali
               Full list: https://gtts.readthedocs.io/en/latest/module.html

    Returns
    -------
    dict with keys:
        success   : bool
        filepath  : absolute path to the saved MP3 (if success)
        error     : error message (if failed)
    """

    if not message or not message.strip():
        return {
            "success": False,
            "filepath": None,
            "error": "Empty message — nothing to convert."
        }

    # build filename
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dispatch_{timestamp}"

    filepath = os.path.join(_OUTPUT_DIR, f"{filename}.mp3")

    try:
        tts = gTTS(text=message, lang=language, slow=False)
        tts.save(filepath)
        print(f"[TTS] Audio saved → {filepath}")
        return {"success": True, "filepath": filepath, "error": None}

    except Exception as e:
        print(f"[TTS] Failed: {e}")
        return {"success": False, "filepath": None, "error": str(e)}


# ── Language code helper ───────────────────────────────────────────────────

LANGUAGE_CODES = {
    "english": "en",
    "hindi":   "hi",
    "tamil":   "ta",
    "telugu":  "te",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati":"gu",
    "punjabi": "pa",
    "urdu":    "ur",
}

def get_language_code(language_name: str) -> str:
    """
    Convert a language name to a gTTS language code.
    Returns 'en' as fallback if not found.
    """
    return LANGUAGE_CODES.get(language_name.lower(), "en")