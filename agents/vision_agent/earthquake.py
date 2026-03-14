import torch
from ultralytics import YOLO
import numpy as np

device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Debris Detection] Using device: {'GPU' if device == 0 else 'CPU'}")

# ── Load model with verification ─────────────────────────────────────────────
MODEL_PATH = r"agents\vision_agent\best_debris.pt"
try:
    model = YOLO(MODEL_PATH)
    model.to(device)
    model.eval()
    print(f"[YOLO] Model loaded successfully from: {MODEL_PATH}")
    print(f"[YOLO] Model task: {model.task}")
    print(f"[YOLO] Model classes: {model.names}")
except Exception as e:
    print(f"[YOLO] FAILED to load model: {e}")
    model = None


def detect_damage(image):
    """
    Detect debris/damage in image using YOLO.
    Args:
        image: numpy array (RGB format)
    Returns:
        List of {"bbox": [x1, y1, x2, y2], "confidence": float}
    """

    if model is None:
        print("[YOLO] Model not loaded. Skipping detection.")
        return []

    if image is None:
        print("[YOLO] No image provided.")
        return []

    print(f"[YOLO] Running inference on image shape: {image.shape}, dtype: {image.dtype}")

    try:
        # YOLO expects RGB — which load_image already provides
        results = model.predict(
            source=image,
            conf=0.01,        # Very low threshold - model outputs 0.01-0.10 range
            iou=0.45,
            device=device,
            verbose=False
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Print raw result for debugging
            print(f"[YOLO] Raw result boxes: {result.boxes}")
            print(f"[YOLO] Speed: {result.speed}")

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = model.names[class_id]

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 3),
                        "label": label
                    })

                print(f"[YOLO] ✅ Found {len(detections)} debris detections:")
                for i, d in enumerate(detections):
                    print(f"   Det {i+1}: {d['label']} | conf={d['confidence']} | bbox={[round(v,1) for v in d['bbox']]}")

            else:
                print("[YOLO] ⚠️ No detections above confidence threshold.")
                print(f"[YOLO] Try lowering conf below current threshold.")

        return detections

    except Exception as e:
        import traceback
        print(f"[YOLO] Error during inference: {e}")
        traceback.print_exc()
        return []