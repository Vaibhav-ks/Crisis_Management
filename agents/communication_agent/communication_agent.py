"""
communication_agent.py
-----------------------
MAIN ENTRY POINT for the Communication Agent.

Call  dispatch_all()  from the Master Coordinator.

This function:
  1. Takes route plans from the Route Agent + zone metadata
  2. Generates human-readable dispatch instructions via Gemini
  3. Optionally translates them into a local language
  4. Sends SMS to field teams via Twilio
  5. Generates a TTS audio file for radio broadcast
  6. Summarizes all field reports for commanders

Input format (what the Coordinator passes in)
─────────────────────────────────────────────

route_plans = [
    {
        "zone":               "Z12",
        "resource_type":      "ambulance",
        "origin":             "City Hospital",
        "destination_latlon": (25.435, 81.846),
        "success":            True,
        "distance_km":        2.3,
        "eta_minutes":        8.0,
        "waypoints":          [...],
        "unit_count":         2,
    },
    ...
]

zone_metadata = {
    "Z12": {"severity": "Critical", "victim_count": 9},
    "Z34": {"severity": "Moderate", "victim_count": 3},
}

field_reports = [
    "Team Alpha: Road blocked near Zone Z12, rerouting.",
    "Team Bravo: 6 survivors found on rooftop in Z34.",
]

dispatch_config = {
    "language":       "English",     # instruction language
    "send_sms":       True,          # whether to send real SMS
    "generate_audio": True,          # whether to generate TTS MP3
    "to_number":      "+91xxxxxxxx", # override default recipient
}
"""



from .gemini_client   import (
    generate_dispatch_instruction,
    summarize_field_reports,
    translate_message,
)
from .sms_dispatcher  import send_sms, send_whatsapp
from .tts_engine      import text_to_speech, get_language_code


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_route_summary(plan: dict) -> str:
    """Build a human-readable route string from a route plan dict."""
    if not plan.get("success"):
        return "Route unavailable — direct navigation required."
    return (
        f"{plan['origin_name']} → Zone {plan['zone']} "
        f"({plan['distance_km']} km, ~{plan['eta_minutes']} min)"
    )


# ── Main public function ───────────────────────────────────────────────────

def dispatch_all(
    route_plans:    list,
    zone_metadata:  dict,
    field_reports:  list  = None,
    dispatch_config: dict = None,
) -> dict:
    """
    Run the full communication pipeline for all route plans.

    Returns
    -------
    dict with keys:
        instructions  : { zone: instruction_text }
        sms_results   : [ { zone, success, sid, error } ]
        audio_files   : [ filepath, ... ]
        summary       : str — 2-line field report summary
    """

    # defaults
    if field_reports is None:
        field_reports = []
    if dispatch_config is None:
        dispatch_config = {}

    language      = dispatch_config.get("language",       "English")
    do_sms        = dispatch_config.get("send_sms",        False)
    do_audio      = dispatch_config.get("generate_audio",  True)
    to_number     = dispatch_config.get("to_number",       None)

    print("\n" + "="*60)
    print("  COMMUNICATION AGENT  —  dispatching")
    print("="*60)

    instructions = {}
    sms_results  = []
    audio_files  = []

    # ── Step 1: Generate + send instructions per route ─────────────────────
    for plan in route_plans:

        zone          = plan.get("zone", "Unknown")
        resource_type = plan.get("resource_type", "team")
        unit_count    = plan.get("unit_count", 1)
        route_summary = _build_route_summary(plan)

        # get severity and victim count from metadata
        meta         = zone_metadata.get(zone, {})
        severity     = meta.get("severity",     "Unknown")
        victim_count = meta.get("victim_count",  0)

        print(f"\n[CommAgent] Processing: {unit_count}x {resource_type} → Zone {zone}")

        # generate instruction via Gemini
        instruction = generate_dispatch_instruction(
            zone          = zone,
            resource_type = resource_type,
            unit_count    = unit_count,
            severity      = severity,
            victim_count  = victim_count,
            route_summary = route_summary,
            language      = language,
        )

        # translate if not English
        if language.lower() != "english":
            instruction = translate_message(instruction, language)
        
        instructions[zone] = instruction
        print(f"  Instruction generated ({len(instruction)} chars)")

        # send SMS
        if do_sms:
            sms_body = (
                f"[CRISIS DISPATCH] Zone {zone} | {unit_count}x {resource_type}\n"
                f"{instruction[:300]}"   # SMS has 160-char limit per segment
            )
            result = send_sms(sms_body, to_number)
            result["zone"] = zone
            sms_results.append(result)
        else:
            print(f"  SMS skipped (send_sms=False)")

        # generate audio for critical zones only (to keep demo clean)
        if do_audio and severity == "Critical":
            lang_code = get_language_code(language)
            audio_result = text_to_speech(
                message  = instruction,
                filename = f"dispatch_{zone}_{resource_type}",
                language = lang_code,
            )
            if audio_result["success"]:
                audio_files.append(audio_result["filepath"])

    # ── Step 2: Summarize field reports ────────────────────────────────────
    print(f"\n[CommAgent] Summarizing {len(field_reports)} field report(s)...")
    summary = summarize_field_reports(field_reports)
    print(f"  Summary: {summary}")

    print(f"\n[CommAgent] Done. "
          f"{len(instructions)} instruction(s), "
          f"{len(sms_results)} SMS(es), "
          f"{len(audio_files)} audio file(s).\n")

    return {
        "instructions": instructions,
        "sms_results":  sms_results,
        "audio_files":  audio_files,
        "summary":      summary,
    }