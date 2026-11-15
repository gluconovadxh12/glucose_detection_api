import os
import sys
import tempfile

import cv2
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.features import extract_features
from app.roi_detector import CENTER_PERCENT, detect_roi_boxes, detect_rois


SAMPLE_IMAGES = [
    "Mob1_100_1_A.png",
    "Mob1_150_1_A.png",
    "Mob1_200_1_A.png",
    "Mob2_100_1_A.png",
    "Mob3_110_2_D.png",
]

FULL_IMAGE_DIR = os.path.join(ROOT, "test_images", "full")
OUTPUT_DIR = os.path.join(ROOT, "test_images", "original_equivalent_validation")


def center_crop(image):
    height, width = image.shape[:2]
    crop_width = int(width * CENTER_PERCENT / 100)
    crop_height = int(height * CENTER_PERCENT / 100)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image[top:top + crop_height, left:left + crop_width]


def save_box_visualization(image, boxes, output_path):
    annotated = image.copy()
    for index, (x, y, width, height) in enumerate(boxes, start=1):
        resized_width = max(width - 20, 1)
        resized_height = max(height - 20, 1)
        crop_width = int(resized_width * CENTER_PERCENT / 100)
        crop_height = int(resized_height * CENTER_PERCENT / 100)
        left = x + 10 + (resized_width - crop_width) // 2
        top = y + 10 + (resized_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 0, 0), 4)
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 4)
        cv2.putText(
            annotated,
            f"ROI {index}",
            (left, max(30, top - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(output_path, annotated)


def predict(roi, model, selector, scaler):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary_file:
            temporary_path = temporary_file.name
        cv2.imwrite(temporary_path, roi)
        features = extract_features(temporary_path)
        values = pd.DataFrame([features])
        selected = selector.transform(values)
        scaled = scaler.transform(selected)
        prediction = int(model.predict(scaled)[0])
        confidence = float(model.predict_proba(scaled)[0].max() * 100)
        return prediction, confidence
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = joblib.load(os.path.join(ROOT, "models", "extra_trees_classifier.pkl"))
    selector = joblib.load(os.path.join(ROOT, "models", "feature_selector.pkl"))
    scaler = joblib.load(os.path.join(ROOT, "models", "minmax_scaler.pkl"))

    print("ORIGINAL-EQUIVALENT FULL-IMAGE VALIDATION")
    print(f"CENTER_PERCENT={CENTER_PERCENT}; no resize")

    for filename in SAMPLE_IMAGES:
        image_path = os.path.join(FULL_IMAGE_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"{filename}: FAILURE image unreadable")
            continue

        boxes = detect_roi_boxes(image_path)
        rois = detect_rois(image_path)
        if boxes is None or rois is None or len(boxes) != 4 or len(rois) != 4:
            print(f"{filename}: FAILURE expected 4 physical boxes and crops")
            continue

        base = os.path.splitext(filename)[0]
        output_subdir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(output_subdir, exist_ok=True)
        visualization_path = os.path.join(output_subdir, "roi_overlay.png")
        save_box_visualization(image, boxes, visualization_path)

        print(f"\nIMAGE: {filename}")
        print(f"Visualization: {visualization_path}")
        for index, ((x, y, width, height), roi) in enumerate(zip(boxes, rois), start=1):
            crop_path = os.path.join(output_subdir, f"roi_{index}_center_crop_70.png")
            cv2.imwrite(crop_path, roi)
            prediction, confidence = predict(roi, model, selector, scaler)
            print(
                f"ROI {index}: box=({x}, {y}, {width}, {height}), "
                f"crop={roi.shape[1]} x {roi.shape[0]}, "
                f"prediction={prediction} mg/dL, confidence={confidence:.2f}%, "
                f"saved={crop_path}"
            )


if __name__ == "__main__":
    main()