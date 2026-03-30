"""
flood_classifier.py
-------------------
Lightweight flood classifier used by the automated folder watcher.

Uses the same pixel-level thresholding as scan_folder.py and the
main pipeline — NOT a simple mean() comparison — so results are
consistent across the whole system.

Detection logic (matches scan_folder.py exactly):
    pixel flooded  →  UNet prob >= PIXEL_THRESHOLD (default 0.45)
    image flooded  →  flooded_pixels / total_pixels >= FLOOD_FRACTION (default 0.10)
"""

import numpy as np
from .flood_segmentation import detect_flood

PIXEL_THRESHOLD = 0.45   # per-pixel UNet cutoff
FLOOD_FRACTION  = 0.10   # min fraction of flooded pixels to declare flood


def classify_flood(image, pixel_threshold: float = PIXEL_THRESHOLD,
                   flood_fraction: float = FLOOD_FRACTION) -> dict:
    """
    Classify whether a flood exists in the image.

    Args:
        image           : numpy array (RGB) — already loaded via preprocess.load_image
        pixel_threshold : float — UNet probability cutoff per pixel
        flood_fraction  : float — minimum fraction of flooded pixels to trigger FLOOD

    Returns:
        {
            "label"              : "Flood" | "No Flood",
            "is_flood"           : bool,
            "flood_probability"  : float  (mean UNet prob across all pixels),
            "flooded_fraction"   : float  (fraction of pixels above pixel_threshold),
            "flooded_pixels"     : int,
            "total_pixels"       : int,
            "max_prob"           : float,
            "flood_prob_map"     : ndarray (H x W float32)
        }
    """
    flood_prob_map = detect_flood(image)           # H x W float32

    total   = flood_prob_map.size
    flooded = int((flood_prob_map >= pixel_threshold).sum())
    frac    = flooded / total
    is_flood = frac >= flood_fraction

    return {
        "label":             "Flood" if is_flood else "No Flood",
        "is_flood":          is_flood,
        "flood_probability": round(float(np.mean(flood_prob_map)), 4),
        "flooded_fraction":  round(frac, 4),
        "flooded_pixels":    flooded,
        "total_pixels":      total,
        "max_prob":          round(float(flood_prob_map.max()), 4),
        "flood_prob_map":    flood_prob_map,
    }