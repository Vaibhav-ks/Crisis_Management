"""
config.py
---------
Single source of truth for default deployment-site coordinates and fleet size.

These are REAL coordinates, not invented placeholders. They cover Velachery,
Chennai — a documented, real flood-prone neighbourhood: badly flooded in 2015
and again in December 2023 (Cyclone Michaung), with NDRF boat rescues recorded
around the overflowing Velachery Lake. The base locations below were verified
via Google Places as of this writing (currently-operating, 24-hour hospital
and an active fire station).

DEFAULT_IMAGE_META's center point IS Velachery Lake (12.988496, 80.212960) —
the actual flood epicenter. If you center a real satellite screenshot here
(Google Maps Satellite view, or the Mapbox Static Images API), the image
content will genuinely correspond to the coordinates the rest of the pipeline
uses for routing — which is the thing that was broken before: synthetic
images with made-up coordinates that didn't correspond to anything real.

HOW TO GET A MATCHING REAL SATELLITE IMAGE
---------------------------------------------
1. Go to Google Maps, search "Velachery Lake, Chennai", switch to Satellite
   view, zoom until roads/buildings are clearly visible (zoom ~16-17).
2. Screenshot the visible area. Note the screenshot's pixel dimensions.
3. Update width_px / height_px below (or pass image_meta explicitly in
   Streamlit Stage 1) to match your actual screenshot's dimensions —
   coverage_km below assumes a roughly 1.6km-wide capture; adjust if your
   zoom level captured more or less area.

WHY THIS FILE EXISTS (architecture note)
-------------------------------------------
Before this fix, the exact same coordinate dict was hardcoded twice, byte for
byte, in master_agent/master_nodes.py and streamlit_app.py, and a third,
independent default (fleet size) was hardcoded inside
agents/resource_agent/rescue_decision_llm.py. Centralizing them here means
updating a location during a demo can't silently desync one screen from
another.
"""

DEFAULT_IMAGE_META = {
    "center_lat":  12.988496,    # Velachery Lake, Chennai — real, documented flood zone
    "center_lon":  80.212960,
    # ⚠️ APPROXIMATE — see "VERIFY coverage_km" note below before trusting
    # route distances/ETAs for anything beyond a demo.
    "coverage_km": 2.0,
    "width_px":    1746,         # matches Images_for_testing/velachery_satellite.png exactly
    "height_px":   769,
}

# VERIFY coverage_km FOR REAL ACCURACY
# --------------------------------------
# coverage_km can't be measured from the image file itself — it depends on
# the zoom level you screenshotted at, which isn't stored in the PNG. 2.0 is
# a reasonable estimate (Velachery Lake's long axis is roughly 1.6-1.8km, and
# this screenshot includes a fair margin of surrounding streets on both
# sides), good enough to demo the pipeline. For real accuracy:
#   1. On Google Maps, right-click the LEFT edge of your screenshot's area →
#      "Measure distance" → click the RIGHT edge of the same area.
#   2. Replace coverage_km above with that real number (in km).
# This matters because every zone's lat/lon (and therefore every route
# distance/ETA) is computed FROM this single number — get it close, or your
# zones will be geographically offset from where they look like they are.

DEFAULT_BASE_LOCATIONS = {
    "ambulance":   {"name": "Prashanth Super Speciality Hospital", "lat": 12.978498,  "lon": 80.221318},
    "rescue_team": {"name": "Velachery Fire Station",              "lat": 12.976802,  "lon": 80.227605},
    "boat":        {"name": "Velachery Lake (launch point)",       "lat": 12.988496,  "lon": 80.212960},
}

# Used by rescue_decision_llm.py whenever no real fleet inventory is supplied.
DEFAULT_AVAILABLE_RESOURCES = {
    "boats":        3,
    "ambulances":   2,
    "rescue_teams": 4,
}