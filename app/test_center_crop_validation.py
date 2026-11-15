import os
import sys
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.features import extract_features
from app.roi_detector import detect_roi_boxes, detect_rois


SAMPLE_IMAGES = [
    "Mob1_100_1_A.png",
    "Mob1_150_1_A.png",
    "Mob1_200_1_A.png",
    "Mob2_100_1_A.png",
    "Mob3_110_2_D.png",
]

FULL_IMAGE_DIR = os.path.join(ROOT, "test_images", "full")
OUTPUT_DIR = os.path.join(ROOT, "test_images", "roi_validation", "center_crop_70")

MODEL_PATH = os.path.join(ROOT, "models", "extra_trees_classifier.pkl")
SELECTOR_PATH = os.path.join(ROOT, "models", "feature_selector.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "minmax_scaler.pkl")


def save_box_visualization(image, boxes, output_path):
    annotated = image.copy()
    for index, (x, y, width, height) in enumerate(boxes, start=1):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            4,
        )
        cv2.putText(
            annotated,
            f"ROI {index}",
            (x, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(output_path, annotated)


def save_center_crop_visualization(rois, output_path):
    label_height = 48
    gap = 20
    canvas_height = max(roi.shape[0] for roi in rois) + label_height
    canvas_width = sum(roi.shape[1] for roi in rois) + gap * (len(rois) - 1)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    x_offset = 0
    for index, roi in enumerate(rois, start=1):
        height, width = roi.shape[:2]
        canvas[label_height:label_height + height, x_offset:x_offset + width] = roi
        cv2.putText(
            canvas,
            f"ROI {index}: {width} x {height}",
            (x_offset, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        x_offset += width + gap

    cv2.imwrite(output_path, canvas)


def predict_single_roi(roi, model, selector, scaler):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary_file:
            temporary_path = temporary_file.name

        if not cv2.imwrite(temporary_path, roi):
            return None

        features = extract_features(temporary_path)
        if features is None:
            return None

        feature_frame = pd.DataFrame([features])
        selected_features = selector.transform(feature_frame)
        scaled_features = scaler.transform(selected_features)

        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = float(probabilities.max() * 100)
        return prediction, confidence
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    scaler = joblib.load(SCALER_PATH)

    for filename in SAMPLE_IMAGES:
        image_path = os.path.join(FULL_IMAGE_DIR, filename)
        image = cv2.imread(image_path)
        print("\n" + "=" * 72)
        print(f"IMAGE: {filename}")
        print("=" * 72)

        if image is None:
            print("RESULT: FAILURE - full image could not be read")
            continue

        boxes = detect_roi_boxes(image_path)
        rois = detect_rois(image_path)
        if boxes is None or rois is None or len(boxes) != 4 or len(rois) != 4:
            print("RESULT: FAILURE - detector did not return exactly 4 ROIs")
            print(f"Boxes: {boxes}")
            print(f"ROI count: {None if rois is None else len(rois)}")
            continue

        output_subdir = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(filename)[0],
        )
        os.makedirs(output_subdir, exist_ok=True)

        annotated_path = os.path.join(output_subdir, "physical_roi_boxes.png")
        save_box_visualization(image, boxes, annotated_path)

        montage_path = os.path.join(output_subdir, "center_cropped_rois.png")
        save_center_crop_visualization(rois, montage_path)

        print("RESULT: exactly 4 physical ROIs detected")
        print("Physical ROI boxes:")
        for index, (x, y, width, height) in enumerate(boxes, start=1):
            print(f"  ROI {index}: ({x}, {y}, {width}, {height})")

        print("70% center-cropped ROI dimensions:")
        for index, roi in enumerate(rois, start=1):
            height, width = roi.shape[:2]
            crop_path = os.path.join(output_subdir, f"roi_{index}_center_crop_70.png")
            cv2.imwrite(crop_path, roi)
            print(f"  ROI {index}: {width} x {height} pixels")
            print(f"    saved: {crop_path}")

        print(f"Physical-box visualization: {annotated_path}")
        print(f"Center-crop visualization: {montage_path}")

        print("Independent Extra Trees predictions:")
        for index, roi in enumerate(rois, start=1):
            result = predict_single_roi(roi, model, selector, scaler)
            if result is None:
                print(f"  ROI {index} -> prediction failed")
                continue

            prediction, confidence = result
            print(
                f"  ROI {index} -> {prediction} mg/dL, "
                f"{confidence:.2f}% confidence"
            )


if __name__ == "__main__":
    main()