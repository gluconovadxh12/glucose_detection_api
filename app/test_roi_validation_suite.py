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
    "test_images/full/Mob1_50_1_A.png",
    "test_images/full/Mob1_100_1_A.png",
    "test_images/full/Mob1_150_1_A.png",
    "test_images/full/Mob1_200_1_A.png",
    "test_images/full/Mob2_100_1_A.png",
    "test_images/full/Mob3_110_2_D.png",
]

OUTPUT_DIR = "test_images/roi_validation"

MODEL_PATH = "models/extra_trees_classifier.pkl"
SELECTOR_PATH = "models/feature_selector.pkl"
SCALER_PATH = "models/minmax_scaler.pkl"


def save_visualization(image_path, boxes, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return

    annotated = image.copy()
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            4,
        )
        cv2.putText(
            annotated,
            f"ROI {index}",
            (x, max(0, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(output_path, annotated)


def predict_single_roi(roi, model, selector, scaler):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        cv2.imwrite(temp_path, roi)
        features = extract_features(temp_path)

        if features is None:
            return None

        X = pd.DataFrame([features])
        X_selected = selector.transform(X)
        X_scaled = scaler.transform(X_selected)

        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(probabilities.max() * 100)
        return prediction, confidence
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def get_candidate_summary(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = []

    for threshold in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
        mask = cv2.inRange(gray, 0, threshold)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        frame_crop = img[y:y + h, x:x + w].copy()
        if frame_crop.shape[0] > frame_crop.shape[1]:
            frame_crop = cv2.rotate(frame_crop, cv2.ROTATE_90_CLOCKWISE)

        gray2 = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        _, inner_mask = cv2.threshold(gray2, 120, 255, cv2.THRESH_BINARY)
        inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours2, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = frame_crop.shape[:2]
        for cnt in contours2:
            area = cv2.contourArea(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect = bw / bh
            if (
                bx < 20
                or by < 20
                or bx + bw > frame_w - 20
                or by + bh > frame_h - 20
            ):
                continue
            if area > 5000 and 1.5 < aspect < 4:
                candidates.append((bx, by, bw, bh, area))

        if len(candidates) >= 4:
            break

    return len(candidates)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    scaler = joblib.load(SCALER_PATH)

    results = []
    failures = []

    for image_path in SAMPLE_IMAGES:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n========================================")
        print(f"IMAGE: {image_path}")
        print(f"========================================")

        boxes = detect_roi_boxes(image_path)
        rois = detect_rois(image_path)
        candidate_count = get_candidate_summary(image_path)

        if boxes is None or rois is None or len(boxes) != 4 or len(rois) != 4:
            failures.append({
                "filename": os.path.basename(image_path),
                "reason": "ROI detection failed or not exactly four ROIs",
                "details": {
                    "boxes": boxes,
                    "roi_count": None if rois is None else len(rois),
                    "candidate_count": candidate_count,
                },
            })
            print("RESULT: FAILURE")
            print(f"Detected boxes: {boxes}")
            print(f"ROI count: {None if rois is None else len(rois)}")
            print(f"Candidate count before final selection: {candidate_count}")
            continue

        area_values = [w * h for _, _, w, h in boxes]
        area_consistency = min(area_values) / max(area_values) if max(area_values) > 0 else 0.0

        detected_layout_ok = True
        if area_consistency < 0.5:
            detected_layout_ok = False
            failures.append({
                "filename": os.path.basename(image_path),
                "reason": "ROI sizes are highly inconsistent",
                "details": {"area_consistency": area_consistency, "areas": area_values},
            })

        if candidate_count > 4:
            detected_layout_ok = False
            failures.append({
                "filename": os.path.basename(image_path),
                "reason": "More than four candidate regions found before filtering",
                "details": {"candidate_count": candidate_count},
            })

        if not detected_layout_ok:
            print("RESULT: GEOMETRY WARNING")
        else:
            print("RESULT: 4 valid ROIs detected")

        print("Bounding boxes:")
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            print(f"  ROI {idx}: ({x}, {y}, {w}, {h})")

        print("ROI dimensions:")
        for idx, roi in enumerate(rois, start=1):
            height, width = roi.shape[:2]
            print(f"  ROI {idx}: {width} x {height} pixels")

        crop_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(crop_dir, exist_ok=True)

        for idx, roi in enumerate(rois, start=1):
            crop_path = os.path.join(crop_dir, f"roi_{idx}.png")
            cv2.imwrite(crop_path, roi)

        vis_path = os.path.join(crop_dir, f"{base_name}_annotated.png")
        save_visualization(image_path, boxes, vis_path)
        print(f"Saved visualization: {vis_path}")

        for idx, roi in enumerate(rois, start=1):
            prediction = predict_single_roi(roi, model, selector, scaler)
            if prediction is None:
                failures.append({
                    "filename": os.path.basename(image_path),
                    "reason": f"ROI {idx} did not pass the existing Extra Trees pipeline",
                    "details": {"roi_index": idx},
                })
                print(f"ROI {idx}: prediction failed")
                continue

            glucose_value, confidence = prediction
            print(f"ROI {idx} -> {glucose_value} mg/dL, {confidence:.2f}% confidence")

        results.append({
            "filename": os.path.basename(image_path),
            "boxes": boxes,
            "roi_count": len(rois),
            "candidate_count": candidate_count,
            "area_consistency": area_consistency,
            "visualization": vis_path,
        })

    print("\n========================================")
    print("SUMMARY")
    print("========================================")
    print(f"Images tested: {len(SAMPLE_IMAGES)}")
    valid_count = len([r for r in results if r["roi_count"] == 4])
    print(f"Images with 4 valid ROIs: {valid_count}")
    print(f"Images with failures: {len(failures)}")

    if failures:
        print("Failure details:")
        for item in failures:
            print(f"  - {item['filename']}: {item['reason']}")
            if item.get('details'):
                print(f"      {item['details']}")

    print("\nSaved visualizations:")
    for item in results:
        print(f"  - {item['filename']}: {item['visualization']}")

    print("\nPer-image prediction summary:")
    for item in results:
        print(f"  - {item['filename']} -> boxes={item['boxes']}")

    with open(os.path.join(OUTPUT_DIR, "validation_summary.txt"), "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Images tested: {len(SAMPLE_IMAGES)}\n")
        summary_file.write(f"Images with 4 valid ROIs: {valid_count}\n")
        summary_file.write(f"Images with failures: {len(failures)}\n")
        if failures:
            summary_file.write("Failures:\n")
            for item in failures:
                summary_file.write(f"  - {item['filename']}: {item['reason']}\n")
        summary_file.write("Saved visualizations:\n")
        for item in results:
            summary_file.write(f"  - {item['filename']}: {item['visualization']}\n")


if __name__ == "__main__":
    main()
