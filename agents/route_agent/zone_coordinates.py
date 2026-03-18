"""
zone_coordinates.py
-------------------
The Vision Agent divides an image into a 10×10 grid and names each cell
using 0-based row/col indices:

    grid_mapper.py:  zone_id = f"Z{gy}{gx}"   # gy=0..9, gx=0..9

Zone names produced are:
    Z00, Z01, Z02 ... Z09
    Z10, Z11, Z12 ... Z19
    ...
    Z90, Z91, Z92 ... Z99

This file converts those names into real-world GPS coordinates (zone centre)
so the Route Agent knows WHERE to navigate.

Row  = first digit after Z  (0 = top row,    9 = bottom row)
Col  = second digit(s)       (0 = left col,   9 = right col)

Example:  "Z35" → row=3, col=5 (4th row from top, 6th col from left)
"""

from .geo_reference import pixel_to_latlon

GRID_ROWS = 10
GRID_COLS = 10


# ── Zone name parser ──────────────────────────────────────────────────────────

def parse_zone_name(zone_name: str) -> tuple:
    """
    Parse a zone name string to (row_idx, col_idx), both 0-based.

    Formats accepted:
        "Z35"   → row=3, col=5
        "Z3_10" → row=3, col=10  (underscore used when col ≥ 10)

    Raises ValueError for malformed names.
    """
    name = zone_name.strip().upper()
    if not name.startswith("Z"):
        raise ValueError(f"Zone name must start with 'Z', got: {zone_name!r}")

    body = name[1:]

    if "_" in body:
        parts = body.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed zone name with underscore: {zone_name!r}")
        row, col = int(parts[0]), int(parts[1])
    else:
        if len(body) < 2:
            raise ValueError(
                f"Zone name {zone_name!r} too short — "
                "expected at least 2 digits after 'Z' (e.g. 'Z00', 'Z35')."
            )
        row = int(body[0])
        col = int(body[1:])

    return row, col  # 0-based, matching Vision Agent's gy, gx


# ── Zone centre in pixels ─────────────────────────────────────────────────────

def zone_center_pixels(row: int, col: int,
                       image_width_px: int, image_height_px: int) -> tuple:
    """
    Return the pixel coordinate (px, py) of the CENTRE of a grid cell.

    Parameters
    ----------
    row, col         : 0-based grid indices  (0 = top/left)
    image_width_px   : full image width in pixels
    image_height_px  : full image height in pixels

    Returns
    -------
    (px, py) floats — pixel coordinates of the cell centre
    """
    cell_w = image_width_px  / GRID_COLS
    cell_h = image_height_px / GRID_ROWS

    # 0-based: col=0 → left edge at 0, centre at cell_w/2
    px = col * cell_w + cell_w / 2
    py = row * cell_h + cell_h / 2

    return px, py


# ── Main public function ──────────────────────────────────────────────────────

def get_zone_latlon(zone_name: str, geo_transform: dict) -> tuple:
    """
    Full pipeline: zone name → (latitude, longitude) of the zone's centre.

    Parameters
    ----------
    zone_name     : e.g. "Z35" or "Z3_10"  (0-based, Vision Agent format)
    geo_transform : dict returned by build_geo_transform()

    Returns
    -------
    (lat, lon) tuple
    """
    row, col = parse_zone_name(zone_name)
    px, py   = zone_center_pixels(
        row, col,
        geo_transform["image_width_px"],
        geo_transform["image_height_px"],
    )
    return pixel_to_latlon(px, py, geo_transform)


def get_all_zone_coordinates(geo_transform: dict) -> dict:
    """
    Pre-compute GPS coordinates for all 100 zones (Z00–Z99).

    Returns
    -------
    { "Z00": (lat, lon), "Z01": (lat, lon), ..., "Z99": (lat, lon) }
    """
    coords = {}
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            name = f"Z{row}{col}"
            coords[name] = get_zone_latlon(name, geo_transform)
    return coords