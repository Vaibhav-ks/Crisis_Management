"""
Direct YOLO test without preprocessing - debug model directly
"""
import cv2
import torch
from ultralytics import YOLO
import os

print("=" * 60)
print("DIRECT YOLO DEBRIS DETECTION TEST")
print("=" * 60)

# Device detection
device = 0 if torch.cuda.is_available() else "cpu"
print(f"\n[Device] Using: {'GPU' if device == 0 else 'CPU'}")

# Model loading
MODEL_PATH = r"C:\Users\shree\Downloads\best (3).pt"
print(f"[Model] Loading from: {MODEL_PATH}")
print(f"[Model] File exists: {os.path.exists(MODEL_PATH)}")

try:
    model = YOLO(MODEL_PATH)
    print(f"[Model] ✓ Loaded successfully")
    print(f"[Model] Task: {model.task}")
    print(f"[Model] Classes: {model.names}")
except Exception as e:
    print(f"[Model] ✗ Failed: {e}")
    exit()

# Test image
test_images = ["new_testing_image.jpg", "debris.jpg"]
for img_name in test_images:
    img_path = img_name
    if not os.path.exists(img_path):
        print(f"\n[Image] {img_path} NOT FOUND")
        continue
    
    print(f"\n{'='*60}")
    print(f"[Image] Testing: {img_path}")
    print(f"[Image] File size: {os.path.getsize(img_path) / (1024*1024):.2f} MB")
    
    # Load with OpenCV
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"[Image] ✗ Failed to load with cv2")
        continue
    
    print(f"[Image] Shape (BGR): {img_cv.shape}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    print(f"[Image] Shape (RGB): {img_rgb.shape}")
    
    # Test prediction WITHOUT resizing
    print(f"\n[YOLO] Predicting (conf=0.25)...")
    results = model.predict(
        source=img_rgb,
        conf=0.25,
        device=device,
        verbose=False
    )
    
    if results and len(results) > 0:
        result = results[0]
        n_boxes = len(result.boxes) if result.boxes is not None else 0
        print(f"[YOLO] Detections found: {n_boxes}")
        
        if n_boxes > 0:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                print(f"  [{i}] bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), conf={conf:.3f}, class={model.names[cls_id]}")
    
    # Try with lower confidence
    print(f"\n[YOLO] Predicting (conf=0.10)...")
    results = model.predict(
        source=img_rgb,
        conf=0.10,
        device=device,
        verbose=False
    )
    
    if results and len(results) > 0:
        result = results[0]
        n_boxes = len(result.boxes) if result.boxes is not None else 0
        print(f"[YOLO] Detections found: {n_boxes}")
        
        if n_boxes > 0:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                print(f"  [{i}] bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), conf={conf:.3f}, class={model.names[cls_id]}")

print(f"\n{'='*60}")
print("TEST COMPLETE")
print(f"{'='*60}")
