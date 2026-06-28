"""
vision_agent.py
---------------
Flood-only vision pipeline (earthquake removed).

Runs in sequence:
  1. load_image       — load + resize the satellite image
  2. detect_flood     — UNet flood segmentation → float prob map (0–1)
  3. build_zone_map   — aggregate scores into 10×10 grid
  4. add_severity     — compute weighted severity score per zone
"""

import os
from .preprocess            import load_image
from .flood_segmentation    import detect_flood
from .grid_mapper           import build_zone_map
from .severity              import add_severity
from .building_segmentation import detect_buildings
from .earthquake            import detect_damage
from .visualizer            import draw_zone_grid


def analyze_image(image_path: str) -> dict:
    """
    Run the full vision pipeline on one satellite/aerial image.
    Flood-only: no earthquake / YOLO debris detection.

    Parameters
    ----------
    image_path : str  path to the image file

    Returns
    -------
    dict:
        "zone_map"       : dict  — 100 zones with flood_score, severity
        "flood_prob_map" : ndarray — raw float flood probability map (H x W, 0-1)
    """
    # 1. Load image
    image = load_image(image_path)

    # 2. Flood segmentation → float probability map (H x W, values 0.0-1.0)
    flood_prob_map = detect_flood(image)

    # 3. Building detection for severity weighting
    building_prob_map = detect_buildings(image)

    # 4. Build 10x10 zone grid with flood + building scores
    zone_map = build_zone_map(
        image             = image,
        flood_prob_map    = flood_prob_map,
        damage_detections = detect_damage(image),            # no earthquake detections
        building_prob_map = building_prob_map,
    )

    # 5. Add composite severity score to each zone
    zone_map = add_severity(zone_map)

    # Save grid image
    output_dir = "zone_results"
    os.makedirs(output_dir, exist_ok=True)
    draw_zone_grid(image, zone_map, os.path.join(output_dir, "grid_output.jpg"))

    return {
        "zone_map":       zone_map,
        "flood_prob_map": flood_prob_map,
    }