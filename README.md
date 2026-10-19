   # Pothole Detection (Edge)

   Real-time pothole detection: USB camera -> YOLOv8n -> tracker (no double counting) -> severity -> local storage.

   ## Status
   - Model trained (YOLOv8n, Roboflow pothole-vhmow v18, mAP50 about 0.86). Weights in `models/best.pt` (not in git, shared separately).
   - Inference code in `inference/` is NOT written or tested yet.

   ## Setup on a new PC
   1. Install Python (tick "Add Python to PATH"), VS Code, GitHub Desktop.
   2. Clone this repo, then `pip install -r requirements.txt`.
   3. Put `best.pt` in `models/`.
   4. Test: put a pothole photo in `test_images/`, run `python test_model.py`.

   ## Coding order
   1. `inference/severity.py` - small/medium/large from box size
   2. `inference/tracking.py` - same pothole across frames counts once
   3. `inference/db.py` - add timestamp, store each unique pothole, export to Excel
   4. `inference/main.py` - camera + model + tracker + severity + storage loop
   5. GPS (`gps_reader.py`) - last, plugs into main.py where a new pothole is logged

   ## Notes
   - Severity is a rough heuristic based on box area, not calibrated.
   - Validation score is slightly optimistic: the dataset has augmented copies of the same photos.
   - Target hardware: Raspberry Pi 5. Export the model to NCNN/ONNX before deploying.