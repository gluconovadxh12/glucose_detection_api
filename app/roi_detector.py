import cv2
import numpy as np


OUTER_THRESHOLDS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
PAD = 10
INNER_MARGIN = 20
CENTER_PERCENT = 70


def _order_boxes(boxes):
    """Return boxes ordered as TL, TR, BL, BR."""
    if len(boxes) < 4:
        return boxes

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    top = sorted(boxes[:2], key=lambda b: b[0])
    bottom = sorted(boxes[2:], key=lambda b: b[0])
    return top + bottom


def _find_outer_frame(orig_img, threshold):
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, threshold)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w <= 0 or h <= 0:
        return None

    frame_crop = orig_img[y:y + h, x:x + w].copy()
    frame_h, frame_w = frame_crop.shape[:2]

    if frame_h > frame_w:
        frame_crop = cv2.rotate(frame_crop, cv2.ROTATE_90_CLOCKWISE)

    return frame_crop, (x, y)


def _detect_inner_candidates(frame_crop):
    gray2 = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray2, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_h, frame_w = frame_crop.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if h == 0:
            continue

        aspect = w / h

        if (
            x < INNER_MARGIN
            or y < INNER_MARGIN
            or x + w > frame_w - INNER_MARGIN
            or y + h > frame_h - INNER_MARGIN
        ):
            continue

        if area > 5000 and 1.5 < aspect < 4:
            candidates.append((x, y, w, h, area))

    if len(candidates) < 4:
        return []

    candidates = sorted(candidates, key=lambda b: b[4], reverse=True)[:4]
    ordered = _order_boxes(candidates)
    areas = [box[4] for box in ordered]
    consistency = min(areas) / max(areas) if max(areas) > 0 else 0.0
    return ordered, consistency


def _to_original_frame_coords(box, rotated, frame_shape):
    x, y, w, h, area = box
    if not rotated:
        return x, y, w, h, area

    frame_h, frame_w = frame_shape[:2]
    x_orig = y
    y_orig = frame_h - (x + w)
    w_orig = h
    h_orig = w
    return x_orig, y_orig, w_orig, h_orig, area


def detect_roi_boxes(image_path):
    """Return the four ROI boxes in original-image coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    best_boxes = None
    best_consistency = -1.0

    for threshold in OUTER_THRESHOLDS:
        frame_result = _find_outer_frame(img, threshold)
        if frame_result is None:
            continue

        frame_crop, offset = frame_result
        rotated = frame_crop.shape[0] > frame_crop.shape[1]
        detection = _detect_inner_candidates(frame_crop)
        if not detection:
            continue

        boxes, consistency = detection
        if consistency <= 0.5:
            continue

        if consistency > best_consistency:
            best_consistency = consistency
            best_boxes = []
            for x, y, w, h, area in boxes:
                if rotated:
                    x, y, w, h, area = _to_original_frame_coords((x, y, w, h, area), True, frame_crop.shape)
                best_boxes.append((x + offset[0], y + offset[1], w, h, area))

    if best_boxes is None:
        fallback = _find_outer_frame(img, 120)
        if fallback is None:
            return None

        frame_crop, offset = fallback
        rotated = frame_crop.shape[0] > frame_crop.shape[1]
        detection = _detect_inner_candidates(frame_crop)
        if not detection:
            return None

        boxes, _ = detection
        best_boxes = []
        for x, y, w, h, area in boxes:
            if rotated:
                x, y, w, h, area = _to_original_frame_coords((x, y, w, h, area), True, frame_crop.shape)
            best_boxes.append((x + offset[0], y + offset[1], w, h, area))

    best_boxes = sorted(best_boxes, key=lambda b: (b[1], b[0]))
    top = sorted(best_boxes[:2], key=lambda b: b[0])
    bottom = sorted(best_boxes[2:], key=lambda b: b[0])
    ordered_boxes = top + bottom

    return [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, _ in ordered_boxes[:4]
    ]


def detect_rois(image_path):
    """
    Detect the four ROI boxes from a full smartphone image.

    Returns:
        list of cropped ROI images in the order:
        ROI 1, ROI 2, ROI 3, ROI 4.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    box_list = detect_roi_boxes(image_path)
    if box_list is None or len(box_list) != 4:
        return None

    target_width = max(width for _, _, width, _ in box_list)
    target_height = max(height for _, _, _, height in box_list)
    cropped_rois = []
    for x, y, width, height in box_list:
        roi = img[y + PAD:y + height - PAD, x + PAD:x + width - PAD]
        if roi.size == 0:
            return None

        roi = cv2.resize(
            roi,
            (target_width - 2 * PAD, target_height - 2 * PAD),
            interpolation=cv2.INTER_AREA,
        )

        height, width = roi.shape[:2]
        crop_width = int(width * CENTER_PERCENT / 100)
        crop_height = int(height * CENTER_PERCENT / 100)

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        center_cropped_roi = roi[top:bottom, left:right]
        if center_cropped_roi.size == 0:
            return None

        cropped_rois.append(center_cropped_roi)

    return cropped_rois


__all__ = ["detect_rois", "detect_roi_boxes"]