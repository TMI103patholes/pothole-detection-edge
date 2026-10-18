# training/training.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    project="runs",
    name="pothole_yolov8n"
)