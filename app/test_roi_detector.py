import os

import cv2

from roi_detector import detect_roi_boxes, detect_rois


IMAGE_PATH = "test_images/full/Mob1_100_1_A.png"
OUTPUT_DIR = "test_images/detected_rois"
VISUALIZATION_PATH = os.path.join(OUTPUT_DIR, "roi_detection_visualization.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("==========================================")
print("ROI DETECTOR VALIDATION")
print("==========================================")
print("FULL IMAGE:")
print(IMAGE_PATH)

rois = detect_rois(IMAGE_PATH)
boxes = detect_roi_boxes(IMAGE_PATH)

if rois is None or boxes is None or len(rois) != 4 or len(boxes) != 4:
    print("\nROI DETECTION FAILED.")
    raise SystemExit(1)

print("\nROI DETECTION:")
for index, (x, y, w, h) in enumerate(boxes, start=1):
    print(f"ROI {index} bounding box: ({x}, {y}, {w}, {h})")

print("\nROI DIMENSIONS:")
for index, roi in enumerate(rois, start=1):
    height, width = roi.shape[:2]
    print(f"ROI {index}: {width} x {height} pixels")

    output_path = os.path.join(OUTPUT_DIR, f"roi_{index}.png")
    cv2.imwrite(output_path, roi)
    print(f"Saved: {output_path}")

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise SystemExit("Unable to read source image.")

annotated = image.copy()
for index, (x, y, w, h) in enumerate(boxes, start=1):
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.putText(
        annotated,
        f"ROI {index}",
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )

cv2.imwrite(VISUALIZATION_PATH, annotated)
print(f"\nSaved visualization: {VISUALIZATION_PATH}")

print("\n==========================================")
print("ROI DETECTOR VALIDATION COMPLETE")
print("==========================================")