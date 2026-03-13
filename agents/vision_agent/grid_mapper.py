import numpy as np


def build_zone_map(image,
                   flood_prob_map,
                   damage_detections,
                   grid_size=10):

    """
    Build zone-wise crisis map using flood probabilities and
    damaged building detections.
    """

    height, width = image.shape[:2]

    cell_w = width // grid_size
    cell_h = height // grid_size

    zone_map = {}

    # initialize zones
    for gy in range(grid_size):
        for gx in range(grid_size):

            zone_id = f"Z{gy}{gx}"

            zone_map[zone_id] = {
                "flood_score": 0.0,
                "damage_score": 0.0
            }

    # -----------------------------
    # Flood score per zone
    # -----------------------------
    for gy in range(grid_size):
        for gx in range(grid_size):

            x_start = gx * cell_w
            x_end = (gx + 1) * cell_w

            y_start = gy * cell_h
            y_end = (gy + 1) * cell_h

            flood_cell = flood_prob_map[y_start:y_end, x_start:x_end]

            flood_score = float(np.mean(flood_cell))

            zone_id = f"Z{gy}{gx}"

            zone_map[zone_id]["flood_score"] = round(flood_score, 3)

    # -----------------------------
    # Damage score from detections
    # -----------------------------
    for det in damage_detections:

        x1, y1, x2, y2 = det["bbox"]
        confidence = det.get("confidence", 1.0)

        # center of bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        gx = cx // cell_w
        gy = cy // cell_h

        if gx < grid_size and gy < grid_size:

            zone_id = f"Z{gy}{gx}"

            zone_map[zone_id]["damage_score"] += confidence

    # normalize damage scores
    max_damage = max([z["damage_score"] for z in zone_map.values()] + [1])

    for zone_id in zone_map:

        zone_map[zone_id]["damage_score"] = round(
            zone_map[zone_id]["damage_score"] / max_damage,
            3
        )

    return zone_map


# import numpy as np

# def build_heatmap(image, detections, flood_prob_map, grid_size=10):

#     height, width = image.shape[:2]

#     cell_w = width // grid_size
#     cell_h = height // grid_size

#     heatmap = np.zeros((grid_size, grid_size))

#     # -------------------------------
#     # Process object detections
#     # -------------------------------
#     for det in detections:

#         x1, y1, x2, y2 = det["bbox"]

#         cx = int((x1 + x2) / 2)
#         cy = int((y1 + y2) / 2)

#         gx = cx // cell_w
#         gy = cy // cell_h

#         if gx < grid_size and gy < grid_size:
#             heatmap[gy][gx] += det["confidence"]

#     # -------------------------------
#     # Process flood probabilities
#     # -------------------------------
#     for gy in range(grid_size):
#         for gx in range(grid_size):

#             x_start = gx * cell_w
#             x_end = (gx + 1) * cell_w

#             y_start = gy * cell_h
#             y_end = (gy + 1) * cell_h

#             cell = flood_prob_map[y_start:y_end, x_start:x_end]

#             flood_intensity = np.mean(cell)

#             heatmap[gy][gx] += flood_intensity

#     return heatmap