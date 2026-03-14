"""
zone_coordinates.py
-------------------
The Vision Agent divides an image into a 10X10 grid and names each cell
like  Z11, Z12 ... Z1_10, Z21 ... Z10_10.

This file converts those zone names into real-world GPS coordinates
(the center of each zone) so the Route Agent knows WHERE to navigate.

Row index = first number  (1 = top row)
Column index= second number (1 = left col)

Example:  Z12  ->  row 1, col 2
"""

from .geo_reference import pixel_to_latlon, build_geo_transform


GRID_ROWS = 10
GRID_COLS = 10


# ---------------------------------------------------------------------------
# Zone name parser
# ---------------------------------------------------------------------------

def parse_zone_name(zone_name: str) -> tuple:
    """
    Parse a zone name string like 'Z12' or 'Z1_2' into (row_idx, col_idx).
    Row and col are 1-based (Z11 = row 1, col 1).

    Accepted formats:
        'Z12'   → row=1, col=2   (only works up to col 9)
        'Z1_10' → row=1, col=10  (use underscore for col ≥ 10)
        'Z10_5' → row=10, col=5
    """
    name = zone_name.strip().upper()
    if not name.startswith("Z"):
        raise ValueError(f"Zone name must start with 'Z', got: {zone_name}")

    body = name[1:]   # strip the leading Z

    if "_" in body:
        parts = body.split("_")
        row, col = int(parts[0]), int(parts[1])
    else:
        # single-digit row and col: 'Z12' → row=1, col=2
        row = int(body[0])
        col = int(body[1:]) if len(body) > 1 else 1

    return row, col


# ---------------------------------------------------------------------------
# Zone center in pixels
# ---------------------------------------------------------------------------

def zone_center_pixels(row: int, col: int, image_width_px: int,
                       image_height_px: int) -> tuple:
    """
    Return the pixel coordinate of the CENTER of a grid cell.

    row, col are 1-based.
    """
    cell_w = image_width_px  / GRID_COLS
    cell_h = image_height_px / GRID_ROWS

    # center of the cell
    px = (col - 1) * cell_w + cell_w / 2
    py = (row - 1) * cell_h + cell_h / 2

    return px, py


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_zone_latlon(zone_name: str, geo_transform: dict) -> tuple:
    """
    Full pipeline: zone name → (latitude, longitude) of zone center.

    Parameters
    ----------
    zone_name     : e.g. 'Z12', 'Z3_10'
    geo_transform : dict returned by build_geo_transform()

    Returns
    -------
    (lat, lon) tuple
    """
    row, col = parse_zone_name(zone_name)

    px, py = zone_center_pixels(
        row, col,
        geo_transform["image_width_px"],
        geo_transform["image_height_px"]
    )

    lat, lon = pixel_to_latlon(px, py, geo_transform)
    return lat, lon


def get_all_zone_coordinates(geo_transform: dict) -> dict:
    """
    Build a dictionary of ALL 100 zone centers at once.
    Useful for pre-computing during startup.

    Returns
    -------
    { 'Z11': (lat, lon), 'Z12': (lat, lon), ... 'Z10_10': (lat, lon) }
    """
    coords = {}
    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            if col <= 9:
                name = f"Z{row}{col}"
            else:
                name = f"Z{row}_{col}"

            coords[name] = get_zone_latlon(name, geo_transform)

    return coords