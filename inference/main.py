import cv2
import os
import datetime
from ultralytics import YOLO
from tracking import PotholeTracker
from severity import classify_severity
from db import init_db, log_pothole

model = YOLO("models/best.pt")
tracker = PotholeTracker()
conn = init_db()
os.makedirs("data/logged_images", exist_ok=True)

cap = cv2.VideoCapture(0)  # 0 = first camera; try 1 if this is the wrong one
if not cap.isOpened():
    print("Could not open the camera. Check permissions and the camera index.")
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    res = model.predict(frame, conf=0.4, verbose=False)[0]
    dets = [(tuple(b.xyxy[0].tolist()), float(b.conf[0])) for b in res.boxes]

    for tid, bbox, conf, is_new in tracker.update(dets):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if is_new:
            sev = classify_severity(bbox, w, h)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/logged_images/pothole_{stamp}_{tid}.jpg"
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(path, crop)
            log_pothole(conn, tid, sev, conf, path)
            print(f"NEW pothole id={tid} severity={sev} conf={conf:.2f}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()