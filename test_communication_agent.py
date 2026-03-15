"""
test_communication_agent.py
----------------------------
Test the Communication Agent in isolation using fake data.
Run from the project root:

    python test_communication_agent.py

Tests (in order):
  1. Gemini instruction generation
  2. Gemini field report summarization
  3. Gemini translation (Hindi)
  4. TTS audio generation
  5. Full dispatch_all() pipeline (no SMS)
  6. SMS send (only if Twilio is configured)
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ── Fake data (mirrors what Route Agent would produce) ─────────────────────

FAKE_ROUTE_PLANS = [
    {
        "zone":               "Z12",
        "resource_type":      "ambulance",
        "origin":             "City Hospital",
        "destination_latlon": (25.435, 81.846),
        "success":            True,
        "distance_km":        2.3,
        "eta_minutes":        8.0,
        "waypoints":          [],
        "unit_count":         3,
    },
    {
        "zone":               "Z34",
        "resource_type":      "rescue_team",
        "origin":             "Rescue Station A",
        "destination_latlon": (25.440, 81.852),
        "success":            True,
        "distance_km":        1.1,
        "eta_minutes":        4.5,
        "waypoints":          [],
        "unit_count":         2,
    },
    {
        "zone":               "Z55",
        "resource_type":      "boat",
        "origin":             "Boat Depot",
        "destination_latlon": (25.430, 81.840),
        "success":            False,
        "distance_km":        0.0,
        "eta_minutes":        0.0,
        "waypoints":          [],
        "unit_count":         1,
    },
]

FAKE_ZONE_METADATA = {
    "Z12": {"severity": "Critical", "victim_count": 9},
    "Z34": {"severity": "Moderate", "victim_count": 4},
    "Z55": {"severity": "Low",      "victim_count": 1},
}

FAKE_FIELD_REPORTS = [
    "Team Alpha reporting from Z12: building partially collapsed, "
    "at least 6 people trapped on second floor. Need cutting tools.",
    "Team Bravo at Z34: flood water receding, 4 survivors on rooftop. "
    "Ladder required for extraction.",
    "Team Charlie: Road on sector 7 completely blocked by debris. "
    "Rerouting via sector 9.",
    "Medical unit at Z12: 2 critical injuries, requesting immediate "
    "evacuation to City Hospital.",
]


# ── Test 1: Instruction generation ────────────────────────────────────────
def test_instruction_generation():
    print("\n" + "="*50)
    print("TEST 1: Gemini instruction generation")
    print("="*50)

    from agents.communication_agent.gemini_client import generate_dispatch_instruction

    instruction = generate_dispatch_instruction(
        zone          = "Z12",
        resource_type = "ambulance",
        unit_count    = 3,
        severity      = "Critical",
        victim_count  = 9,
        route_summary = "City Hospital → Zone Z12 (2.3 km, ~8 min)",
        language      = "English",
    )
    print(instruction)
    assert len(instruction) > 50, "Instruction too short"
    print("\n✓ PASSED")


# ── Test 2: Field report summarization ────────────────────────────────────
def test_summarization():
    print("\n" + "="*50)
    print("TEST 2: Field report summarization")
    print("="*50)

    from agents.communication_agent.gemini_client import summarize_field_reports

    summary = summarize_field_reports(FAKE_FIELD_REPORTS)
    print(summary)
    assert len(summary) > 20, "Summary too short"
    print("\n✓ PASSED")


# ── Test 3: Translation ────────────────────────────────────────────────────
def test_translation():
    print("\n" + "="*50)
    print("TEST 3: Translation to Hindi")
    print("="*50)

    from agents.communication_agent.gemini_client import translate_message

    original = "Proceed to Zone Z12 immediately. 9 victims reported. Use Route B."
    translated = translate_message(original, "Hindi")
    print(f"Original : {original}")
    print(f"Hindi    : {translated}")
    assert len(translated) > 10, "Translation too short"
    print("\n✓ PASSED")


# ── Test 4: TTS ────────────────────────────────────────────────────────────
def test_tts():
    print("\n" + "="*50)
    print("TEST 4: Text-to-speech (English)")
    print("="*50)

    from agents.communication_agent.tts_engine import text_to_speech

    result = text_to_speech(
        message  = "Attention all units. Proceed to Zone Zed 12 immediately. "
                   "Nine victims reported. This is a critical situation.",
        filename = "test_dispatch",
        language = "en",
    )
    print(result)
    assert result["success"], f"TTS failed: {result['error']}"
    assert os.path.exists(result["filepath"]), "Audio file not created"
    print(f"\n✓ PASSED — audio saved to: {result['filepath']}")


# ── Test 5: Full pipeline (no SMS) ─────────────────────────────────────────
def test_full_pipeline():
    print("\n" + "="*50)
    print("TEST 5: Full dispatch_all() pipeline")
    print("="*50)

    from agents.communication_agent.communication_agent import dispatch_all

    result = dispatch_all(
        route_plans     = FAKE_ROUTE_PLANS,
        zone_metadata   = FAKE_ZONE_METADATA,
        field_reports   = FAKE_FIELD_REPORTS,
        dispatch_config = {
            "language":       "English",
            "send_sms":       False,   # no real SMS in this test
            "generate_audio": True,
        }
    )

    print("\n── Instructions ──")
    for zone, instr in result["instructions"].items():
        print(f"\nZone {zone}:\n{instr[:200]}...")

    print("\n── Summary ──")
    print(result["summary"])

    print("\n── Audio files ──")
    for f in result["audio_files"]:
        print(f"  {f}")

    assert "Z12" in result["instructions"], "Missing Z12 instruction"
    assert len(result["summary"]) > 0,      "Empty summary"
    print("\n✓ PASSED")


# ── Test 6: SMS (only if Twilio configured) ────────────────────────────────
def test_sms():
    print("\n" + "="*50)
    print("TEST 6: SMS dispatch")
    print("="*50)

    sid   = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    to    = os.getenv("YOUR_PHONE_NUMBER")

    if not all([sid, token, to]):
        print("⚠ Twilio not configured — skipping SMS test.")
        print("  Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, YOUR_PHONE_NUMBER to .env")
        return

    from agents.communication_agent.sms_dispatcher import send_sms

    result = send_sms(
        message   = "[TEST] Crisis Management System — SMS dispatch working correctly.",
        to_number = to,
    )
    print(result)
    assert result["success"], f"SMS failed: {result['error']}"
    print("\n✓ PASSED")


# ── Run all tests ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚨 COMMUNICATION AGENT — TEST SUITE")
    print("====================================")

    test_instruction_generation()
    test_summarization()
    test_translation()
    test_tts()
    test_full_pipeline()
    test_sms()

    print("\n\n✅ ALL TESTS COMPLETE")