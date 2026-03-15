# """
# gemini_client.py
# ----------------
# All Gemini Flash interactions for the Communication Agent.

# Responsibilities:
#   1. generate_dispatch_instruction() — converts a raw task into clear
#      step-by-step field instructions for rescue teams
#   2. summarize_field_reports()       — condenses N field reports into
#      a 2-line situation summary for commanders
#   3. translate_message()             — rewrites a message in a target
#      language (Hindi, Tamil, etc.) using Gemini

# All functions are stateless — pass in text, get text back.
# No LangGraph state is touched here.
# """

# import os
# import json
# from google import generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# # ── Init Gemini client ─────────────────────────────────────────────────────
# _API_KEY = os.getenv("GOOGLE_API_KEY")
# if not _API_KEY:
#     raise EnvironmentError(
#         "GOOGLE_API_KEY not found in environment. "
#         "Add it to your .env file."
#     )

# genai.configure(api_key=_API_KEY)
# _MODEL = genai.GenerativeModel("gemini-1.5-flash")


# # ── 1. Dispatch instruction generator ─────────────────────────────────────

# def generate_dispatch_instruction(
#     zone: str,
#     resource_type: str,
#     unit_count: int,
#     severity: str,
#     victim_count: int,
#     route_summary: str,
#     language: str = "English",
# ) -> str:
#     """
#     Convert a raw task into clear, numbered field instructions.

#     Parameters
#     ----------
#     zone          : e.g. 'Z12'
#     resource_type : e.g. 'ambulance', 'rescue_team', 'boat'
#     unit_count    : how many units are being dispatched
#     severity      : 'Critical' | 'Moderate' | 'Low'
#     victim_count  : estimated victims in the zone
#     route_summary : human-readable route string e.g.
#                     'City Hospital → Road B → Zone Z12 (2.3 km, 8 min)'
#     language      : output language (default English)

#     Returns
#     -------
#     str — numbered step-by-step dispatch instruction
#     """

#     prompt = f"""
# You are an emergency dispatch coordinator generating field instructions
# for a crisis response team during a disaster.

# Task details:
# - Destination zone : {zone}
# - Resource type    : {resource_type}
# - Units dispatched : {unit_count}
# - Zone severity    : {severity}
# - Estimated victims: {victim_count}
# - Route            : {route_summary}

# Write clear, concise, numbered step-by-step instructions for the field team.
# Use simple language. Be direct. Include: what to do on arrival, how many
# people to expect, and any safety warnings based on severity level.
# Maximum 6 steps.
# Write in {language}.
# """

#     response = _MODEL.generate_content(prompt)
#     return response.text.strip()


# # ── 2. Field report summarizer ─────────────────────────────────────────────

# def summarize_field_reports(reports: list[str]) -> str:
#     """
#     Condense a list of field reports into a 2-line situation summary.

#     Parameters
#     ----------
#     reports : list of raw text reports from field teams

#     Returns
#     -------
#     str — 2-line summary for commanders
#     """

#     if not reports:
#         return "No field reports received."

#     numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reports))

#     prompt = f"""
# You are an emergency operations analyst.
# Below are field reports from rescue teams on the ground during a disaster.

# Field Reports:
# {numbered}

# Write a 2-line summary for the incident commander.
# Be factual, concise, and highlight the most critical information.
# Do not add any preamble or explanation — just the 2 lines.
# """

#     response = _MODEL.generate_content(prompt)
#     return response.text.strip()


# # ── 3. Message translator ──────────────────────────────────────────────────

# def translate_message(message: str, target_language: str) -> str:
#     """
#     Translate a dispatch message into the target language.

#     Parameters
#     ----------
#     message         : original message in English
#     target_language : e.g. 'Hindi', 'Tamil', 'Telugu', 'Bengali'

#     Returns
#     -------
#     str — translated message
#     """

#     prompt = f"""
# Translate the following emergency dispatch message into {target_language}.
# Keep the numbered format if present. Use clear, simple language.
# Only return the translated text — no explanation or preamble.

# Message:
# {message}
# """

#     response = _MODEL.generate_content(prompt)
#     return response.text.strip()


"""
gemini_client.py
----------------
All Gemini Flash interactions for the Communication Agent.

Responsibilities:
  1. generate_dispatch_instruction() — converts a raw task into clear
     step-by-step field instructions for rescue teams
  2. summarize_field_reports()       — condenses N field reports into
     a 2-line situation summary for commanders
  3. translate_message()             — rewrites a message in a target
     language (Hindi, Tamil, etc.) using Gemini

All functions are stateless — pass in text, get text back.
No LangGraph state is touched here.
"""
import os
import time
from google import genai
from dotenv import load_dotenv

# Free tier = 5 requests/min. Add a small delay between calls.
_REQUEST_DELAY_S = 13   # 60s / 5 requests = 12s minimum, 13 to be safe

load_dotenv()

# ── Init Gemini client ─────────────────────────────────────────────────────
_API_KEY = os.getenv("GOOGLE_API_KEY")
if not _API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found in environment. "
        "Add it to your .env file."
    )

_CLIENT = genai.Client(api_key=_API_KEY)
_MODEL  = "gemini-2.5-flash"


# ── 1. Dispatch instruction generator ─────────────────────────────────────

def generate_dispatch_instruction(
    zone: str,
    resource_type: str,
    unit_count: int,
    severity: str,
    victim_count: int,
    route_summary: str,
    language: str = "English",
) -> str:
    """
    Convert a raw task into clear, numbered field instructions.

    Parameters
    ----------
    zone          : e.g. 'Z12'
    resource_type : e.g. 'ambulance', 'rescue_team', 'boat'
    unit_count    : how many units are being dispatched
    severity      : 'Critical' | 'Moderate' | 'Low'
    victim_count  : estimated victims in the zone
    route_summary : human-readable route string e.g.
                    'City Hospital → Road B → Zone Z12 (2.3 km, 8 min)'
    language      : output language (default English)

    Returns
    -------
    str — numbered step-by-step dispatch instruction
    """

    prompt = f"""
You are an emergency dispatch coordinator generating field instructions
for a crisis response team during a disaster.

Task details:
- Destination zone : {zone}
- Resource type    : {resource_type}
- Units dispatched : {unit_count}
- Zone severity    : {severity}
- Estimated victims: {victim_count}
- Route            : {route_summary}

Write clear, concise, numbered step-by-step instructions for the field team.
Use simple language. Be direct. Include: what to do on arrival, how many
people to expect, and any safety warnings based on severity level.
Maximum 6 steps.
Write in {language}.
"""

    response = _CLIENT.models.generate_content(model=_MODEL, contents=prompt)
    time.sleep(_REQUEST_DELAY_S)
    return response.text.strip()


# ── 2. Field report summarizer ─────────────────────────────────────────────

def summarize_field_reports(reports: list[str]) -> str:
    """
    Condense a list of field reports into a 2-line situation summary.

    Parameters
    ----------
    reports : list of raw text reports from field teams

    Returns
    -------
    str — 2-line summary for commanders
    """

    if not reports:
        return "No field reports received."

    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reports))

    prompt = f"""
You are an emergency operations analyst.
Below are field reports from rescue teams on the ground during a disaster.

Field Reports:
{numbered}

Write a 2-line summary for the incident commander.
Be factual, concise, and highlight the most critical information.
Do not add any preamble or explanation — just the 2 lines.
"""

    response = _CLIENT.models.generate_content(model=_MODEL, contents=prompt)
    time.sleep(_REQUEST_DELAY_S)
    return response.text.strip()


# ── 3. Message translator ──────────────────────────────────────────────────

def translate_message(message: str, target_language: str) -> str:
    """
    Translate a dispatch message into the target language.

    Parameters
    ----------
    message         : original message in English
    target_language : e.g. 'Hindi', 'Tamil', 'Telugu', 'Bengali'

    Returns
    -------
    str — translated message
    """

    prompt = f"""
Translate the following emergency dispatch message into {target_language}.
Keep the numbered format if present. Use clear, simple language.
Only return the translated text — no explanation or preamble.

Message:
{message}
"""

    response = _CLIENT.models.generate_content(model=_MODEL, contents=prompt)
    time.sleep(_REQUEST_DELAY_S)
    return response.text.strip()