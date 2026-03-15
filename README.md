# Crisis Management AI System

An end-to-end multi-agent AI pipeline that takes a satellite or aerial image of a disaster area, analyses it using computer vision, deploys virtual drones to count survivors, allocates rescue resources using an LLM, and plans optimal road routes — all with human-in-the-loop approval gates.

---

## System Flow

```
Satellite Image
      │
      ▼
┌─────────────┐
│ Vision Agent│  ← UNet flood model + YOLO debris model
└──────┬──────┘
       │  zone_map  (100 zones, flood_score / damage_score / severity)
       │  flood_prob_map (raw float array for road blocking)
       ▼
┌─────────────┐
│  Store Zone │  → Writes all zone scores to crisis.db (SQLite)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Drone Analysis  │  ← Reads crisis.db, picks top-5 most-affected zones by severity
└──────┬───────────┘
       │  most_affected_zones: ["Z35", "Z72", ...]
       ▼
┌──────────────────┐
│  Drone Decision  │  ← Assigns one drone per zone
└──────┬───────────┘
       │  drone_allocation: {"drone_1": "Z35", ...}
       ▼
┌──────────────────┐
│  Drone Dispatch  │  ← Maps each zone to a real image from zone_images/ folder
└──────┬───────────┘
       │  zone_image_map: {"Z35": "zone_images/img1.jpg", ...}
       ▼
┌──────────────────┐
│  Drone Vision    │  ← YOLO COCO model counts people, animals, vehicles per zone
└──────┬───────────┘
       │  people_counts: {"Z35": 12, "Z72": 5, ...}
       ▼
┌──────────────────┐
│  Update People   │  → Updates people_count column in crisis.db
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Rescue Decision  │  ← Gemini LLM reads zone_map + people_counts
│   (LLM node)     │    allocates boats / ambulances / rescue_teams per zone
└──────┬───────────┘
       │  rescue_plan: {"Z35": {"ambulances": 2, "boats": 1}, ...}
       ▼
┌──────────────────┐
│  Admin Approval  │  ← Human operator reviews the rescue plan (y/n)
│  (Resource Gate) │    Rejected → loops back to Rescue Decision
└──────┬───────────┘
       │  approved
       ▼
┌──────────────────┐
│  Route Planner   │  ← Takes rescue_plan + image_meta + flood_prob_map
│  (Route Agent)   │    Plans Dijkstra routes on road graph per resource per zone
└──────┬───────────┘
       │  route_plan: [{zone, resource_type, waypoints, distance_km, eta_minutes}, ...]
       ▼
┌──────────────────┐
│  Admin Approval  │  ← Human operator reviews the route plan (y/n)
│  (Route Gate)    │
└──────┬───────────┘
       │  approved
       ▼
┌──────────────────┐
│  Communication   │  ← Gemini generates dispatch instructions per zone
│     Agent        │    Twilio sends SMS to field teams
└──────┬───────────┘    gTTS converts critical alerts to MP3 audio
       │  dispatch_result: {instructions, sms_results, audio_files, summary}
       ▼
     END
```

---

## Agents

### Vision Agent (`agents/vision_agent/`)

Takes a single aerial image and produces a structured `zone_map` for 100 zones (10×10 grid).

| File | What it does |
|------|-------------|
| `preprocess.py` | Loads image via OpenCV, converts BGR → RGB, resizes if > 1024px |
| `flood_segmentation.py` | Runs UNet (`unet_flood_model1.2.pth`) → float probability map (0–1) per pixel |
| `earthquake.py` | Runs YOLO (`best_debris.pt`) → bounding boxes of damaged buildings |
| `grid_mapper.py` | Divides image into 10×10 grid, aggregates flood scores and damage scores per zone |
| `severity.py` | Computes `severity = 0.6 × flood_score + 0.4 × damage_score` per zone |
| `victim_counter.py` | Runs YOLO COCO (`yolov8n.pt`) → counts people, animals, vehicles |
| `vision_agent.py` | Orchestrates the above; returns `zone_map` + `flood_prob_map` |

**Zone naming:** Zones are named `Z{row}{col}` with 0-based indices. `Z00` is top-left, `Z99` is bottom-right.

**Output:**
```python
{
    "zone_map": {
        "Z00": {"flood_score": 0.12, "damage_score": 0.0, "severity": 0.072},
        "Z35": {"flood_score": 0.85, "damage_score": 0.72, "severity": 0.799},
        ...  # 100 zones total
    },
    "flood_prob_map": <numpy array H×W, float32, values 0–1>
}
```

---

### Resource Agent (`agents/resource_agent/`)

Decides which zones matter most and what resources to send.

| File | What it does |
|------|-------------|
| `drone_analysis.py` | Queries `crisis.db`, sorts zones by severity + flood_score, returns top-N |
| `rescue_decision_llm.py` | Calls Gemini LLM with zone data + people counts, returns resource allocation JSON |
| `priority_model.py` | (Legacy) Numeric priority scoring — used by `resource_agent.py` standalone |
| `optimizer.py` | (Legacy) OR-Tools LP solver for ambulance allocation |

**LLM Output format** (what flows into the Route Agent):
```python
{
    "Z35": {"boats": 2, "ambulances": 1, "rescue_teams": 2},
    "Z72": {"boats": 0, "ambulances": 1, "rescue_teams": 1}
}
```

---

### Drone Agent (`agents/drone_agent/`)

| File | What it does |
|------|-------------|
| `drone_nodes.py` | Allocates drones to zones (`allocate_drones`), maps zones to image files |
| `drone_vision.py` | Runs victim detection on each zone image, saves annotated output to `zone_results/` |

**Required:** A `zone_images/` folder at project root with one image per zone. Images are matched to zones in order (first image → first zone, etc).

---

### Route Agent (`agents/route_agent/`)

Plans the actual road routes for every approved resource dispatch.

| File | What it does |
|------|-------------|
| `geo_reference.py` | Converts pixel coordinates ↔ GPS using the image's geographic metadata |
| `zone_coordinates.py` | Parses zone names (e.g. `Z35`) → pixel position → GPS coordinate |
| `road_network.py` | Downloads OSM road graph (or builds synthetic one), applies flood masks to block roads |
| `router.py` | Dijkstra shortest path on the road graph, builds waypoint list |
| `route_agent.py` | Main entry point: `plan_all_routes()` — ties everything together |

**Is it limited to Prayagraj?** No. The Route Agent works for any city worldwide. All coordinates come from `image_meta` passed in at runtime — there are no hardcoded city coordinates in the routing logic. The synthetic road graph is always built centred on `image_meta["center_lat/lon"]`, so it correctly represents whatever area the image covers.

**How it knows where to route:**
1. `image_meta` tells it the GPS centre and coverage area of the image
2. Zone names (`Z35`) are converted to GPS using the image pixel dimensions
3. Base locations (hospital, NDRF station, etc.) are caller-provided — not hardcoded
4. The road graph is built around the image centre, then Dijkstra finds the shortest path

**With real OSM roads** (`use_real_osm=True`): downloads actual OpenStreetMap roads for the exact lat/lon area, caches to disk so subsequent runs are instant.

**With synthetic graph** (`use_real_osm=False`): builds a 3×3 grid of 9 road nodes centred on the image location — good for testing and offline use, gives rough distance estimates.

---


### Communication Agent (`agents/communication_agent/`)

Generates human-readable dispatch instructions, sends SMS alerts, and produces audio files for radio broadcast.

| File | What it does |
|------|-------------|
| `gemini_client.py` | Three Gemini calls: generate field instructions, summarize reports, translate to local language |
| `sms_dispatcher.py` | Twilio SMS and WhatsApp dispatch to field team phone numbers |
| `tts_engine.py` | gTTS text-to-speech — converts Critical zone alerts to MP3 audio files saved in `audio_outputs/` |
| `communication_agent.py` | Main entry point: `dispatch_all()` — ties everything together |

**Input** (from Route Agent via Master State):
```python
route_plans     # list of route dicts from plan_all_routes()
zone_metadata   # {"Z35": {"severity": "Critical", "victim_count": 12}, ...}
field_reports   # optional list of ground team report strings
dispatch_config # {"language": "English", "send_sms": False, "generate_audio": True}
```

**Output:**
```python
{
    "instructions": {"Z35": "1. Proceed to Zone Z35...", ...},
    "sms_results":  [{"zone": "Z35", "success": True, "sid": "SM..."}],
    "audio_files":  ["agents/communication_agent/audio_outputs/dispatch_Z35_ambulances.mp3"],
    "summary":      "Z35: 12 trapped, critical injuries. Z72: flood receding, 5 survivors."
}
```

**Languages supported:** English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Punjabi, Urdu.

**SMS:** Disabled by default (`send_sms=False`). Enable by setting `send_sms=True` in `dispatch_config` and adding Twilio credentials to `.env`.

**Audio:** Only generated for `Critical` severity zones to keep output clean.

---

### Master Agent (`master_agent/`)

| File | What it does |
|------|-------------|
| `master_state.py` | LangGraph `TypedDict` — all shared state fields across the pipeline |
| `master_nodes.py` | Every LangGraph node function (vision_node, rescue_decision_node, route_planner_node, ...) |
| `master_graph.py` | Builds and compiles the `StateGraph` with all edges and conditional branches |

---

## Database

`crisis.db` (SQLite) — single table:

```sql
CREATE TABLE zones (
    zone_id      TEXT PRIMARY KEY,  -- e.g. "Z35"
    flood_score  REAL,
    damage_score REAL,
    severity     REAL,
    people_count INTEGER,
    drone_deployed INTEGER,
    last_updated TEXT
)
```

Written by: `update_from_vision.py` (after Vision Agent), `update_people_count.py` (after Drone Vision).
Read by: `load_zone_state.py` (Drone Analysis), `rescue_decision_llm.py` (Resource LLM).

---

## ML Models (not in repo — load from disk)

| File | Architecture | Used by |
|------|-------------|---------|
| `agents/vision_agent/unet_flood_model1.2.pth` | UNet (ResNet-34 encoder, segmentation_models_pytorch) | `flood_segmentation.py` |
| `agents/vision_agent/best_debris.pt` | YOLOv8 (custom-trained) | `earthquake.py` |
| `yolov8n.pt` | YOLOv8n COCO (detects 80 classes) | `victim_counter.py` — auto-downloaded by ultralytics on first run |

---

## Running

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Set your Gemini API key
cp sample.env .env
# Edit .env → GOOGLE_API_KEY=your_key_here

# 3. Initialise the database
python db/init_db.py

# 4. Run the system
python run_system.py
# Enter: path/to/your/disaster/image.jpg
```

When prompted by admin approval gates, type `y` to approve or `n` to reject and re-plan.

---

## Running Tests

```bash
python test_route_agent.py
```

Tests cover: geo-transform accuracy, zone coordinate parsing (all 100 zones), road graph structure, Dijkstra routing, resource key normalisation, flood mask handling, multi-city routing (Delhi / Mumbai / Chennai / Prayagraj), and dynamic rerouting after blockages.

Shapely-dependent tests (flood polygon generation) are automatically skipped if `shapely` is not installed. Install with `pip install shapely` to enable them.

---

## Project Structure

```
Crisis_Management/
├── agents/
│   ├── vision_agent/          # Image analysis (flood + debris + victim detection)
│   ├── resource_agent/        # Zone prioritisation + LLM resource allocation
│   ├── drone_agent/           # Drone dispatch + victim counting per zone
│   └── route_agent/           # GPS routing for every resource deployment
├── master_agent/              # LangGraph pipeline orchestration
├── db/                        # SQLite helpers (init, read, write)
├── zone_images/               # (create this) one image per zone for drone vision
├── zone_results/              # Auto-created: annotated drone vision outputs
├── crisis.db                  # Runtime database
├── run_system.py              # Entry point
├── test_route_agent.py        # Test suite (36 tests, 0 failures)
└── requirements.txt
```

---

## Key Design Decisions

**Why not delete blocked roads?** Deleting an edge can disconnect the graph entirely, making Dijkstra fail even when a longer safe route exists. Instead, blocked edges get `travel_time = 1e9` seconds — Dijkstra still finds a path but will always prefer any unblocked road over it.

**Why 0-based zone IDs?** The Vision Agent's `grid_mapper.py` uses `f"Z{gy}{gx}"` where `gy, gx` start at 0. Zone `Z00` is the top-left cell. `Z99` is the bottom-right.

**Why synthetic graph for offline?** The 3×3 grid with 9 nodes gives real multi-hop paths (up to 5 hops, ~2.5 km) that prove routing logic works end-to-end without requiring an internet connection. The graph is always centred on the actual crisis location from `image_meta`.

**Why LLM for resource allocation?** The allocation problem is multi-constraint (flood needs boats, people need ambulances, debris needs rescue teams, can't exceed available units). An LLM with a clear prompt handles this more flexibly than a hardcoded LP formulation and can incorporate qualitative factors.