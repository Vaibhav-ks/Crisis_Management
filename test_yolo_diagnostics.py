"""
Comprehensive YOLO model diagnostics
"""
import os
import torch
from ultralytics import YOLO
import cv2

MODEL_PATH = r"C:\Users\shree\Downloads\best (3).pt"

print("=" * 70)
print("YOLO MODEL DIAGNOSTICS")
print("=" * 70)

# 1. File checks
print(f"\n[1] FILE CHECKS:")
print(f"    Path: {MODEL_PATH}")
print(f"    Exists: {os.path.exists(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
    print(f"    Size: {size_mb:.2f} MB")
    
    # Check if it's a valid PyTorch file
    try:
        state = torch.load(MODEL_PATH, map_location='cpu')
        print(f"    ✓ Valid PyTorch file")
        if isinstance(state, dict):
            print(f"    Keys: {list(state.keys())[:5]}")
    except Exception as e:
        print(f"    ✗ NOT a valid PyTorch file: {e}")

# 2. Load model
print(f"\n[2] MODEL LOADING:")
try:
    model = YOLO(MODEL_PATH)
    print(f"    ✓ Model loaded")
    print(f"    Task: {model.task}")
    print(f"    Architecture: {model.model}")
    print(f"    Classes: {model.names}")
    print(f"    Num classes: {model.nc}")
except Exception as e:
    print(f"    ✗ Failed to load: {e}")
    exit()

# 3. Test with file path (not numpy array)
print(f"\n[3] PREDICT WITH FILE PATH:")
test_img = "image.png"
if os.path.exists(test_img):
    print(f"    Image: {test_img}")
    try:
        results = model.predict(source=test_img, conf=0.25, verbose=True)
        if results and len(results) > 0:
            n = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"    Detections: {n}")
            if n > 0:
                for box in results[0].boxes:
                    print(f"      conf={float(box.conf[0]):.3f}")
    except Exception as e:
        print(f"    Error: {e}")

# 4. Test with numpy array (RGB)
print(f"\n[4] PREDICT WITH NUMPY ARRAY (RGB):")
if os.path.exists(test_img):
    try:
        img = cv2.imread(test_img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"    Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        
        results = model.predict(source=img_rgb, conf=0.25, verbose=True)
        if results and len(results) > 0:
            n = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"    Detections: {n}")
    except Exception as e:
        print(f"    Error: {e}")

# 5. Test with different confidence levels
print(f"\n[5] CONFIDENCE THRESHOLD TEST:")
thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
if os.path.exists(test_img):
    for conf in thresholds:
        try:
            results = model.predict(source=test_img, conf=conf, verbose=False)
            n = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"    conf={conf:.2f}: {n} detections")
        except Exception as e:
            print(f"    conf={conf:.2f}: Error - {e}")

# 6. Test with actual best.pt from Colab if available
print(f"\n[6] ALTERNATIVE MODEL PATHS TO TRY:")
alt_paths = [
    r"C:\Users\shree\Downloads\best_debris.pt",
    r"agents\vision_agent\best (3).pt",
    r"best.pt",
    r"runs\detect\debris_yolov8\weights\best.pt",
]
for path in alt_paths:
    exists = os.path.exists(path)
    print(f"    {path}: {'✓ EXISTS' if exists else 'NOT FOUND'}")

print(f"\n{'='*70}")
print("END DIAGNOSTICS")
print(f"{'='*70}")
