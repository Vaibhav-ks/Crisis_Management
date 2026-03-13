from .preprocess import load_image
from .flood_segmentation import detect_flood
from .earthquake import detect_damage
from .grid_mapper import build_zone_map
from .severity import add_severity


def analyze_image(image_path):

    # Load image
    image = load_image(image_path)

    # Flood segmentation
    flood_prob_map = detect_flood(image)

    # Detect damaged buildings
    damage_detections = detect_damage(image)

    # Build zone-wise scores
    zone_map = build_zone_map(
        image=image,
        flood_prob_map=flood_prob_map,
        damage_detections=damage_detections
    )

    # Add severity score
    zone_map = add_severity(zone_map)

    return {
        "zone_map": zone_map
    }


# from .preprocess import load_image
# from .detector import detect_objects
# from .flood_segmentation import detect_flood
# from .grid_mapper import build_heatmap
# from .severity import classify_severity
# from .victim_counter import count_victims_by_zone


# def analyze_image(image_path):

#     image = load_image(image_path)

#     detections = detect_objects(image)

#     flood_mask = detect_flood(image)

#     heatmap = build_heatmap(image, detections, flood_mask)

#     severity = classify_severity(heatmap)

#     victims = count_victims_by_zone(detections, image)

#     return {
#         "heatmap": heatmap.tolist(),
#         "severity": severity,
#         "victims": victims
#     }