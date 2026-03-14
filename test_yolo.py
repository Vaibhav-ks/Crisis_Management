# test_yolo.py  — run this directly
from ultralytics import YOLO
import cv2

model = YOLO(r"agents\vision_agent\best_debris.pt")
print("Classes:", model.names)

img = cv2.imread("Gemini_Generated_Image_ao9zwjao9zwjao9z.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model.predict(source=img_rgb, conf=0.10, verbose=True)
print("Boxes found:", len(results[0].boxes))