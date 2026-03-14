import cv2
import torch
from ultralytics import YOLO
from agents.vision_agent.vision_agent import analyze_image

# Analyze image
result = analyze_image("new_testing_image.jpg")

# Print full zone map
print("\n[Result] Zone Map:")
for zone_id, data in result["zone_map"].items():
    if data["severity"] > 0:  # Only print non-zero zones to reduce noise
        print(f"  {zone_id}: flood={data['flood_score']}, damage={data['damage_score']}, severity={data['severity']}")

# ============================================================================
# Draw bounding boxes with confidence scores
# ============================================================================
print("\n[Visualization] Drawing bounding boxes...")

# Load model and image
MODEL_PATH = r"agents\vision_agent\best_debris.pt"
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

# Load original image
img_cv = cv2.imread("new_testing_image.jpg")
if img_cv is None:
    print("❌ Could not load image")
else:
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Run YOLO prediction
    results = model.predict(
        source=img_rgb,
        conf=0.01,
        device=device,
        verbose=False
    )
    
    # Draw boxes on BGR copy (for cv2.imwrite)
    img_draw = img_cv.copy()
    
    if results and len(results) > 0:
        result = results[0]
        detections_count = 0
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]
                
                detections_count += 1
                
                # Convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle (BGR format for cv2)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label_text = f"{label} {confidence:.3f}"
                cv2.putText(
                    img_draw,
                    label_text,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                print(f"  ✓ Debris #{detections_count}: bbox=({x1},{y1},{x2},{y2}), conf={confidence:.3f}")
        
        print(f"\n✅ Total detections: {detections_count}")
    else:
        print("⚠️ No detections found")
    
    # Save annotated image
    cv2.imwrite("image_with_detections.jpg", img_draw)
    print(f"✓ Saved annotated image: image_with_detections.jpg")
    
    # Also display stats
    h, w = img_draw.shape[:2]
    print(f"📷 Image size: {w}x{h} pixels")