"""
geo_reference.py
----------------
Converts image pixel coordinates to real-world (latitude, longitude).

Two supported cases:
  Case A: Image has GeoTIFF metadata  → uses rasterio to read transform directly.
  Case B: Image has no metadata       → caller provides center_lat, center_lon,
          coverage_km and we compute the bounding box ourselves.

Every other file in the Route Agent calls geo_reference to turn pixel/zone
positions into real GPS coordinates before touching any road data.

All returned values are explicitly cast to native Python float
so that LangGraph's MemorySaver (msgpack) can serialise them without raising
"Type is not msgpack serializable: numpy.float64".
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Case B helper  (most common in demos / drone feeds without GeoTIFF)
# ---------------------------------------------------------------------------

def build_geo_transform(center_lat: float, center_lon: float, coverage_km: float,
                        image_width_px: int, image_height_px: int) -> dict:
    """
    Build a simple affine-like transform dictionary from image metadata.

    Parameters
    ----------
    center_lat      : latitude  of the image center
    center_lon      : longitude of the image center
    coverage_km     : how many kilometres the image width covers on the ground
    image_width_px  : pixel width  of the image
    image_height_px : pixel height of the image

    Returns
    -------
    dict with keys:
        top_left_lat, top_left_lon,
        lat_per_pixel, lon_per_pixel,
        image_width_px, image_height_px

    All float values are native Python float (NOT numpy.float64).
    """
    # Use plain math functions to avoid numpy scalar types entirely
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(float(center_lat)))

    total_lon_span = float(coverage_km) / km_per_deg_lon
    aspect_ratio   = float(image_height_px) / float(image_width_px)
    total_lat_span = (float(coverage_km) * aspect_ratio) / km_per_deg_lat

    # Top-left corner (north-west)
    top_left_lat = float(center_lat) + (total_lat_span / 2.0)
    top_left_lon = float(center_lon) - (total_lon_span / 2.0)

    # Degrees per pixel
    lat_per_pixel = total_lat_span / float(image_height_px)
    lon_per_pixel = total_lon_span / float(image_width_px)

    return {
        "top_left_lat":    float(top_left_lat),
        "top_left_lon":    float(top_left_lon),
        "lat_per_pixel":   float(lat_per_pixel),
        "lon_per_pixel":   float(lon_per_pixel),
        "image_width_px":  int(image_width_px),
        "image_height_px": int(image_height_px),
    }


def pixel_to_latlon(px: float, py: float, transform: dict) -> tuple:
    """
    Convert a single (px, py) pixel coordinate to (latitude, longitude).

    px = column index (x, left → right)
    py = row    index (y, top  → bottom)

    Returns
    -------
    (lat, lon) tuple of native Python float values.
    """
    lat = transform["top_left_lat"] - float(py) * transform["lat_per_pixel"]
    lon = transform["top_left_lon"] + float(px) * transform["lon_per_pixel"]
    # round() returns float when input is float, but explicitly cast to be safe
    return float(round(lat, 6)), float(round(lon, 6))


# ---------------------------------------------------------------------------
# Case A helper  (GeoTIFF — needs rasterio installed)
# ---------------------------------------------------------------------------

def build_geo_transform_from_geotiff(tiff_path: str) -> dict:
    """
    Read the affine transform from a GeoTIFF file.
    Only used when the Vision Agent feeds real satellite imagery with metadata.

    Requires:  pip install rasterio
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Install rasterio to use GeoTIFF mode:  pip install rasterio")

    with rasterio.open(tiff_path) as src:
        t = src.transform
        w, h = src.width, src.height

    # t.c = top-left lon, t.f = top-left lat
    # t.a = lon per pixel, t.e = lat per pixel (negative → going south)
    return {
        "top_left_lat":    float(t.f),
        "top_left_lon":    float(t.c),
        "lat_per_pixel":   float(abs(t.e)),
        "lon_per_pixel":   float(t.a),
        "image_width_px":  int(w),
        "image_height_px": int(h),
    }