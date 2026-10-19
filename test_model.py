import os
from ultralytics import YOLO

model = YOLO("models/best.pt")
folder = "test_images"

for name in sorted(os.listdir(folder)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue
    results = model.predict(os.path.join(folder, name), conf=0.4, save=True,
                            project="test_output", name="run", exist_ok=True, verbose=False)
    boxes = results[0].boxes
    confs = [round(float(c), 2) for c in boxes.conf]
    print(f"{name}: {len(boxes)} pothole(s) {confs}")