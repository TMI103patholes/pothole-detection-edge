def classify_severity(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
    if area_ratio < 0.01:
        return "small"
    elif area_ratio < 0.04:
        return "medium"
    return "large"