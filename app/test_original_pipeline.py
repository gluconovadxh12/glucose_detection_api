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


IMAGE_PATH = os.path.join(ROOT, "test_images", "full", "Mob1_100_1_A.png")
OUTPUT_DIR = os.path.join(ROOT, "test_images", "original_pipeline_test", "Mob1_100_1_A")

MODEL_PATH = os.path.join(ROOT, "models", "extra_trees_classifier.pkl")
SELECTOR_PATH = os.path.join(ROOT, "models", "feature_selector.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "minmax_scaler.pkl")

PERSPECTIVE_WIDTH = 2200
PERSPECTIVE_HEIGHT = 800
CENTER_PERCENT = 70


def order_corners(points):
    points = points.reshape(4, 2)
    point_sums = points.sum(axis=1)
    point_differences = np.diff(points, axis=1)

    top_left = points[np.argmin(point_sums)]
    bottom_right = points[np.argmax(point_sums)]
    top_right = points[np.argmin(point_differences)]
    bottom_left = points[np.argmax(point_differences)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def detect_perspective_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_black = cv2.inRange(gray, 0, 80)
    contours, _ = cv2.findContours(
        mask_black,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError("No outer-frame contour was detected.")

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approximation = cv2.approxPolyDP(largest, epsilon, True)
    if len(approximation) != 4:
        raise RuntimeError(
            f"Outer contour approximated to {len(approximation)} corners, expected 4."
        )

    source_corners = order_corners(approximation)
    destination_corners = np.array(
        [
            [0, 0],
            [PERSPECTIVE_WIDTH, 0],
            [PERSPECTIVE_WIDTH, PERSPECTIVE_HEIGHT],
            [0, PERSPECTIVE_HEIGHT],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_corners, destination_corners)
    corrected_frame = cv2.warpPerspective(
        image,
        transform,
        (PERSPECTIVE_WIDTH, PERSPECTIVE_HEIGHT),
    )
    return corrected_frame, source_corners


def detect_boxes(corrected_frame):
    gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if height == 0:
            continue

        aspect = width / height
        if area > 5000 and 1.5 < aspect < 4:
            boxes.append((x, y, width, height, area, aspect))

    if len(boxes) < 4:
        raise RuntimeError(f"Only {len(boxes)} physical box candidates were detected.")

    boxes = sorted(boxes, key=lambda box: box[4], reverse=True)[:4]
    boxes = sorted(boxes, key=lambda box: box[1])
    boxes = sorted(boxes[:2], key=lambda box: box[0]) + sorted(
        boxes[2:],
        key=lambda box: box[0],
    )
    return boxes


def center_crop(image):
    height, width = image.shape[:2]
    crop_width = int(width * CENTER_PERCENT / 100)
    crop_height = int(height * CENTER_PERCENT / 100)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image[top:bottom, left:right]


def save_box_visualization(corrected_frame, boxes, output_path):
    annotated = corrected_frame.copy()
    for index, (x, y, width, height, _, _) in enumerate(boxes, start=1):
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


def predict_roi(roi, model, selector, scaler):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary_file:
            temporary_path = temporary_file.name

        if not cv2.imwrite(temporary_path, roi):
            raise RuntimeError("Could not save temporary ROI for feature extraction.")

        features = extract_features(temporary_path)
        if features is None:
            raise RuntimeError("Feature extraction returned no features.")

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
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    scaler = joblib.load(SCALER_PATH)

    corrected_frame, source_corners = detect_perspective_frame(image)
    boxes = detect_boxes(corrected_frame)
    crops = []

    visualization_path = os.path.join(
        OUTPUT_DIR,
        "perspective_corrected_physical_boxes.png",
    )
    save_box_visualization(corrected_frame, boxes, visualization_path)

    print("ORIGINAL DOCUMENTED PIPELINE EXPERIMENT")
    print(f"Input image: {IMAGE_PATH}")
    print(f"Input dimensions: {image.shape[1]} x {image.shape[0]} x {image.shape[2]}")
    print(f"Ordered source corners: {source_corners.astype(int).tolist()}")
    print(
        "Perspective-corrected frame dimensions: "
        f"{PERSPECTIVE_WIDTH} x {PERSPECTIVE_HEIGHT}"
    )
    print(f"Physical-box visualization: {visualization_path}")
    print("Physical boxes:")

    for index, (x, y, width, height, area, aspect) in enumerate(boxes, start=1):
        box_image = corrected_frame[y:y + height, x:x + width]
        crop = center_crop(box_image)
        crops.append(crop)
        crop_path = os.path.join(
            OUTPUT_DIR,
            f"roi_{index}_center_crop_70.png",
        )
        cv2.imwrite(crop_path, crop)
        print(
            f"  ROI {index}: box=({x}, {y}, {width}, {height}), "
            f"area={area:.1f}, aspect={aspect:.4f}, "
            f"final_crop={crop.shape[1]} x {crop.shape[0]}, "
            f"saved={crop_path}"
        )

    training_dir = os.path.join(ROOT, "test_images", "cropped_roi")
    print("Training ROI dimension comparison:")
    for index in range(1, 5):
        training_path = os.path.join(
            training_dir,
            f"Mob1_100_1_A_box{index}_center_crop.png",
        )
        training_image = cv2.imread(training_path)
        if training_image is None:
            print(f"  ROI {index}: unavailable: {training_path}")
            continue
        print(
            f"  ROI {index}: experimental={crops[index - 1].shape[1]} x "
            f"{crops[index - 1].shape[0]}, "
            f"training={training_image.shape[1]} x {training_image.shape[0]}, "
            f"path={training_path}"
        )

    print("Independent Extra Trees predictions:")
    for index, crop in enumerate(crops, start=1):
        prediction, confidence = predict_roi(crop, model, selector, scaler)
        print(
            f"  ROI {index} -> {prediction} mg/dL, "
            f"{confidence:.2f}% confidence"
        )


if __name__ == "__main__":
    main()