"""
scan_folder.py
--------------
Flood-only satellite image folder scanner.

Usage:
    python scan_folder.py                        # prompts for folder path
    python scan_folder.py satellite_images
    python scan_folder.py satellite_images --pixel-threshold 0.45 --flood-fraction 0.10

Detection logic:
    pixel flagged flooded  →  UNet prob >= --pixel-threshold (default 0.45)
    image declared flooded →  >= --flood-fraction pixels flagged (default 10%)
"""

import os, sys, argparse, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PIXEL_THRESHOLD = 0.45
FLOOD_FRACTION  = 0.10


class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m"
    RED    = "\033[91m"; GREEN  = "\033[92m"
    YELLOW = "\033[93m"; CYAN   = "\033[96m"
    GREY   = "\033[90m"; WHITE  = "\033[97m"

def banner(text, colour=C.CYAN):
    w = 60
    print(f"\n{colour}{C.BOLD}{'─'*w}\n  {text}\n{'─'*w}{C.RESET}")

def info(msg):        print(f"{C.GREY}[INFO ]{C.RESET} {msg}")
def ok(msg):          print(f"{C.GREEN}[ OK  ]{C.RESET} {msg}")
def warn(msg):        print(f"{C.YELLOW}[WARN ]{C.RESET} {msg}")
def err(msg):         print(f"{C.RED}[ERROR]{C.RESET} {msg}")
def flood_alert(msg): print(f"\n{C.RED}{C.BOLD}🚨  {msg}{C.RESET}\n")
def clear_line(msg):  print(f"{C.GREEN}[CLEAR]{C.RESET} {msg}")


banner("Loading UNet flood-segmentation model...", C.CYAN)
try:
    from agents.vision_agent.preprocess         import load_image
    from agents.vision_agent.flood_classifier   import classify_flood
    ok("UNet model loaded (flood_classifier.py)")
except Exception as e:
    err(f"Could not load vision model: {e}")
    sys.exit(1)


def collect_images(folder: Path) -> list:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _bar(fraction: float, width: int = 20) -> str:
    filled   = int(fraction * width)
    thresh_i = int(FLOOD_FRACTION * width)
    bar = ""
    for i in range(width):
        if i < filled:
            bar += "█" if i < thresh_i else "▓"
        else:
            bar += "░"
    colour = C.RED if fraction >= FLOOD_FRACTION else C.GREEN
    return f"{colour}[{bar}]{C.RESET}"


def run_pipeline(image_path: Path):
    banner("Invoking master_graph crisis pipeline", C.RED)
    try:
        from master_agent.master_graph import master_graph
    except Exception as e:
        err(f"Could not import master_graph: {e}")
        raise

    info(f"master_graph.invoke(satellite_image={image_path.name!r})")
    print()
    master_graph.invoke({
        "satellite_image": str(image_path),
        "field_reports":   [],
        "dispatch_config": {
            "send_sms":       True,
            "generate_audio": True,
            "language":       "English",
        },
    })
    banner("Pipeline complete", C.GREEN)


def scan_folder(folder: Path, pixel_threshold: float, flood_fraction: float):
    global PIXEL_THRESHOLD, FLOOD_FRACTION
    PIXEL_THRESHOLD = pixel_threshold
    FLOOD_FRACTION  = flood_fraction

    images = collect_images(folder)
    if not images:
        warn(f"No image files found in: {folder}")
        return

    banner(f"Satellite folder scan — {len(images)} image(s)", C.CYAN)
    info(f"Folder          : {folder}")
    info(f"Pixel threshold : UNet prob >= {pixel_threshold}  →  pixel flagged")
    info(f"Flood trigger   : flagged pixels >= {flood_fraction*100:.0f}% of image area")
    print()

    scanned_dir = folder.parent / "scanned_images"
    scanned_dir.mkdir(exist_ok=True)

    flood_found = False

    for idx, img_path in enumerate(images, start=1):
        print(f"{C.WHITE}[{idx:02d}/{len(images):02d}]{C.RESET} {img_path.name}  ", end="", flush=True)

        t0 = time.time()
        try:
            image = load_image(str(img_path))
            r     = classify_flood(image, pixel_threshold, flood_fraction)
        except Exception as exc:
            print()
            err(f"Failed on {img_path.name}: {exc}")
            continue

        elapsed = time.time() - t0
        frac    = r["flooded_fraction"]
        stats   = (
            f"flooded={frac*100:5.1f}%  "
            f"max={r['max_prob']:.3f}  "
            f"{_bar(frac)}  ({elapsed:.1f}s)"
        )

        if r["is_flood"]:
            print(f"{C.RED}{stats}{C.RESET}")
            flood_alert(
                f"FLOOD DETECTED — \"{img_path.name}\"  "
                f"| {frac*100:.1f}% pixels flooded  "
                f"| peak prob={r['max_prob']:.3f}"
            )
            remaining = images[idx:]
            if remaining:
                warn(f"Skipping {len(remaining)} unscanned: "
                     f"{', '.join(p.name for p in remaining)}")
            flood_found = True
            run_pipeline(img_path)
            # Move processed image to scanned_images after pipeline
            dest = scanned_dir / img_path.name
            img_path.rename(dest)
            info(f"Moved to scanned_images: {dest}")
            break
        else:
            print(f"{C.GREEN}{stats}{C.RESET}")
            clear_line(
                f"\"{img_path.name}\"  "
                f"{frac*100:.1f}% flooded pixels  "
                f"< {flood_fraction*100:.0f}% trigger  →  no flood"
            )
            # Move non-flood image out of satellite_images
            dest = scanned_dir / img_path.name
            img_path.rename(dest)
            info(f"Moved to scanned_images: {dest}")

    if not flood_found:
        print()
        banner("All images scanned — NO FLOOD DETECTED", C.GREEN)
        ok(f"Processed {len(images)} image(s). None hit the {flood_fraction*100:.0f}% trigger.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan satellite images for flood only (earthquake removed).\n"
            "Fires full crisis pipeline on the first flooded image.\n\n"
            "Detection logic:\n"
            "  pixel flagged flooded  →  UNet output >= --pixel-threshold (default 0.45)\n"
            "  image declared flooded →  >= --flood-fraction pixels flagged (default 10%%)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", nargs="?", default=None,
                        help="Satellite image folder. Prompted if omitted.")
    parser.add_argument("--pixel-threshold", type=float, default=PIXEL_THRESHOLD)
    parser.add_argument("--flood-fraction",  type=float, default=FLOOD_FRACTION)
    args = parser.parse_args()

    folder = (
        Path(args.folder).resolve() if args.folder
        else Path(input("\nEnter satellite image folder path: ").strip()).resolve()
    )

    if not folder.exists() or not folder.is_dir():
        err(f"Invalid folder: {folder}")
        sys.exit(1)

    scan_folder(folder, args.pixel_threshold, args.flood_fraction)


if __name__ == "__main__":
    main()