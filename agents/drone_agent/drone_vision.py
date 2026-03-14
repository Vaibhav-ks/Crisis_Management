import cv2
import os
import torch
from ultralytics import YOLO


def load_zone_image(image_path):
    """
    Load a zone image from disk and convert to RGB numpy array.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"[Drone Vision] Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def create_output_folder():
    """
    Create zone_results folder if it doesn't exist.
    """
    output_dir = "zone_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def draw_detections_on_image(image_cv, detections, zone_id, people_count):
    """
    Draw bounding boxes on image for all detected people.
    
    Args:
        image_cv: OpenCV image (BGR)
        detections: List of detection dicts with class, confidence, bbox
        zone_id: Zone identifier for labeling
        people_count: Number of people detected
    
    Returns:
        Annotated image (BGR)
    """
    img_annotated = image_cv.copy()
    
    # Draw zone label at top
    h, w = img_annotated.shape[:2]
    label_text = f"Zone: {zone_id} | People: {people_count}"
    cv2.putText(
        img_annotated,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    # Draw detections
    for det in detections:
        if det['class'] != 'person':
            continue  # Only draw people
        
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = det['confidence']
        
        # Draw blue rectangle for person
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw confidence score
        label = f"Person {conf:.2f}"
        cv2.putText(
            img_annotated,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
    
    return img_annotated


def drone_vision_node(state):
    """
    LangGraph node: Processes one image per affected zone using the victim counter.
    Sends each zone image to detect_victims_and_vehicles and collects people counts.
    Saves annotated images with bounding boxes to zone_results/ folder.

    Reads:
        state["zone_image_map"]: { zone_id: image_path }

    Returns:
        { "people_counts": { zone_id: int } }
    """
    from agents.vision_agent.victim_counter import detect_victims_and_vehicles

    zone_image_map = state.get("zone_image_map", {})

    if not zone_image_map:
        print("[DRONE VISION] ⚠️  No zone images to process!")
        return {"people_counts": {}}

    print(f"\n[DRONE VISION] Processing {len(zone_image_map)} zone images for victim detection")

    # Create output folder
    output_dir = create_output_folder()
    print(f"[DRONE VISION] 📁 Output folder: {output_dir}/")

    people_counts = {}

    for zone_id, image_path in zone_image_map.items():
        print(f"\n[DRONE VISION] Analyzing zone {zone_id} → {image_path}")

        try:
            # Load image
            image_rgb = load_zone_image(image_path)
            image_cv = cv2.imread(image_path)  # For saving

            # Run victim detection
            result = detect_victims_and_vehicles(image_rgb)

            people = result.get("people", 0)
            animals = result.get("animals", 0)
            vehicles = result.get("vehicles", 0)
            detections = result.get("detections", [])

            people_counts[zone_id] = people

            print(f"[DRONE VISION] Zone {zone_id}: "
                  f"👤 {people} people | 🐾 {animals} animals | 🚗 {vehicles} vehicles")

            # Draw annotations and save
            img_annotated = draw_detections_on_image(image_cv, detections, zone_id, people)
            
            # Save output image
            output_filename = f"{output_dir}/{zone_id}_analysis.jpg"
            cv2.imwrite(output_filename, img_annotated)
            print(f"[DRONE VISION] ✓ Saved: {output_filename}")

        except Exception as e:
            print(f"[DRONE VISION] Error processing zone {zone_id}: {e}")
            people_counts[zone_id] = 0

    print(f"\n[DRONE VISION] ✅ Final people counts per zone: {people_counts}")
    print(f"[DRONE VISION] 📁 All results saved to: {output_dir}/")

    return {"people_counts": people_counts}