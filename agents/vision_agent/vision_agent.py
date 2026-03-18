
"""
vision_agent.py
---------------
Entry point for the Vision Agent.

Runs in sequence:
  1. load_image       — load + resize the satellite image
  2. detect_flood     — UNet flood segmentation → float prob map (0–1)
  3. detect_damage    — YOLO debris detection   → bounding box list
  4. build_zone_map   — aggregate scores into 10×10 grid
  5. add_severity     — compute weighted severity score per zone
"""
from streamlit import image

from .visualizer import draw_zone_grid
import os
from .preprocess        import load_image
from .flood_segmentation import detect_flood
from .earthquake        import detect_damage
from .grid_mapper       import build_zone_map
from .severity          import add_severity


def analyze_image(image_path: str) -> dict:
    """
    Run the full vision pipeline on one satellite/aerial image.

    Parameters
    ----------
    image_path : str — path to the image file

    Returns
    -------
    dict with keys:
        "zone_map"       : dict  — 100 zones with flood_score, damage_score, severity
        "flood_prob_map" : ndarray — raw float flood probability map (H×W, 0–1)
                           ADDED: needed by Route Agent for road blocking
    """

    # 1. Load image
    image = load_image(image_path)

    # 2. Flood segmentation → float probability map (H×W, values 0.0–1.0)
    flood_prob_map = detect_flood(image)

    # 3. Detect damaged buildings/debris → list of bbox dicts
    damage_detections = detect_damage(image)

    # 4. Build 10×10 zone grid with flood + damage scores
    zone_map = build_zone_map(
        image             = image,
        flood_prob_map    = flood_prob_map,
        damage_detections = damage_detections,
    )

    # 5. Add composite severity score to each zone
    zone_map = add_severity(zone_map)

    # create output folder
    output_dir = "zone_results"
    os.makedirs(output_dir, exist_ok=True)

    # save grid image
    output_path = os.path.join(output_dir, "grid_output.jpg")

    draw_zone_grid(image, zone_map, output_path)

    return {
        "zone_map":       zone_map,
        "flood_prob_map": flood_prob_map,  
    }






# from .preprocess import load_image
# from .flood_segmentation import detect_flood
# from .earthquake import detect_damage
# from .grid_mapper import build_zone_map
# from .severity import add_severity


# def analyze_image(image_path):

#     # Load image
#     image = load_image(image_path)

#     # Flood segmentation
#     flood_prob_map = detect_flood(image)

#     # Detect damaged buildings
#     damage_detections = detect_damage(image)

#     # Build zone-wise scores
#     zone_map = build_zone_map(
#         image=image,
#         flood_prob_map=flood_prob_map,
#         damage_detections=damage_detections
#     )

#     # Add severity score
#     zone_map = add_severity(zone_map)

#     return {
#         "zone_map": zone_map
#     }


# # from .preprocess import load_image
# # from .detector import detect_objects
# # from .flood_segmentation import detect_flood
# # from .grid_mapper import build_heatmap
# # from .severity import classify_severity
# # from .victim_counter import count_victims_by_zone


# # def analyze_image(image_path):

# #     image = load_image(image_path)

# #     detections = detect_objects(image)

# #     flood_mask = detect_flood(image)

# #     heatmap = build_heatmap(image, detections, flood_mask)

# #     severity = classify_severity(heatmap)

# #     victims = count_victims_by_zone(detections, image)

# #     return {
# #         "heatmap": heatmap.tolist(),
# #         "severity": severity,
# #         "victims": victims
# #     }
