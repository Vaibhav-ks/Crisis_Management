import json
import re
from utils.gemini_llm import llm


def allocate_rescue_resources_llm(zone_map, people_counts, zones, available_resources=None):

    # Default available resources
    if available_resources is None:
        available_resources = {
            "boats": 3,
            "ambulances": 2,
            "rescue_teams": 4
        }

    print("\n[RESCUE LLM] Available resources:", available_resources)

    zone_data = []

    for z in zones:

        data = zone_map.get(z, {})

        zone_data.append({
            "zone": z,
            "flood_score": data.get("flood_score", 0),
            "damage_score": data.get("damage_score", 0),
            "severity": data.get("severity", 0),
            "people_count": people_counts.get(z, 0)
        })

    prompt = f"""
You are an emergency disaster response planner.

Your task is to allocate rescue resources to zones.

ZONE DATA:
{zone_data}

AVAILABLE RESOURCES:
{available_resources}

Rules:
1. Do NOT exceed available resources.
2. Flooded areas need boats.
3. High people_count zones need ambulances.
4. High damage_score zones need rescue teams.

IMPORTANT:
Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.

Example output:

{{
 "Z12": {{"boats":2,"ambulances":1,"rescue_teams":2}},
 "Z21": {{"boats":1,"ambulances":0,"rescue_teams":1}}
}}
"""

    response = llm.invoke(prompt)

    content = response.content

    print("\n[LLM RAW RESPONSE]")
    print(content)

    try:
        rescue_plan = json.loads(content)

    except:

        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)

            if json_match:
                rescue_plan = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")

        except Exception as e:
            print("[LLM] Failed to parse response:", e)
            rescue_plan = {}

    return rescue_plan