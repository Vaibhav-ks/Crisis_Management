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
"""

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
    """
    # 1 degree latitude  ≈ 111 km  (constant anywhere on Earth)
    # 1 degree longitude ≈ 111 * cos(lat) km  (shrinks toward the poles)

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))

    total_lon_span = coverage_km / km_per_deg_lon
    aspect_ratio   = image_height_px / image_width_px
    total_lat_span = (coverage_km * aspect_ratio) / km_per_deg_lat

    # Top-left corner (north-west)
    top_left_lat = center_lat + (total_lat_span / 2)
    top_left_lon = center_lon - (total_lon_span / 2)

    # Degrees per pixel
    # Rows increase downward; latitude decreases going south → we subtract when converting.
    lat_per_pixel = total_lat_span / image_height_px
    lon_per_pixel = total_lon_span / image_width_px

    return {
        "top_left_lat":    top_left_lat,
        "top_left_lon":    top_left_lon,
        "lat_per_pixel":   lat_per_pixel,
        "lon_per_pixel":   lon_per_pixel,
        "image_width_px":  image_width_px,
        "image_height_px": image_height_px,
    }


def pixel_to_latlon(px: float, py: float, transform: dict) -> tuple:
    """
    Convert a single (px, py) pixel coordinate to (latitude, longitude).

    px = column index (x, left → right)
    py = row    index (y, top  → bottom)

    Image rows increase downward while latitude decreases southward,
    so we subtract py * lat_per_pixel from the top-left latitude.
    Longitude increases eastward, so we add px * lon_per_pixel.
    """
    lat = transform["top_left_lat"] - py * transform["lat_per_pixel"]
    lon = transform["top_left_lon"] + px * transform["lon_per_pixel"]
    return round(lat, 6), round(lon, 6)


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
        "top_left_lat":    t.f,
        "top_left_lon":    t.c,
        "lat_per_pixel":   abs(t.e),
        "lon_per_pixel":   t.a,
        "image_width_px":  w,
        "image_height_px": h,
    }