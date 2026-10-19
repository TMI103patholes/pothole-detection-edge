class PotholeTracker:
    def __init__(self, iou_threshold=0.3, max_missed_frames=15,
                 min_hits=3, max_unconfirmed_missed=2):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.max_unconfirmed_missed = max_unconfirmed_missed

    def _iou(self, a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def update(self, detections):
        # detections: list of (bbox, conf)
        # returns list of (track_id, bbox, conf, is_new)
        # is_new is True only ONCE per pothole: on the frame where it
        # has been seen min_hits times (confirmed).
        matched, results = set(), []
        for bbox, conf in detections:
            best_id, best_iou = None, 0
            for tid, t in self.tracks.items():
                iou = self._iou(bbox, t["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is not None and best_iou >= self.iou_threshold and best_id not in matched:
                t = self.tracks[best_id]
                t.update(bbox=bbox, missed=0)
                t["hits"] += 1
                is_new = False
                if not t["confirmed"] and t["hits"] >= self.min_hits:
                    t["confirmed"] = True
                    is_new = True
                matched.add(best_id)
                results.append((best_id, bbox, conf, is_new))
            else:
                tid = self.next_id
                self.next_id += 1
                confirmed = self.min_hits <= 1
                self.tracks[tid] = {"bbox": bbox, "missed": 0,
                                    "hits": 1, "confirmed": confirmed}
                matched.add(tid)
                results.append((tid, bbox, conf, confirmed))
        for tid in list(self.tracks):
            if tid not in matched:
                t = self.tracks[tid]
                t["missed"] += 1
                limit = self.max_missed_frames if t["confirmed"] else self.max_unconfirmed_missed
                if t["missed"] > limit:
                    del self.tracks[tid]
        return results