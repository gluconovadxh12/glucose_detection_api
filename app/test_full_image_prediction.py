import os
import tempfile

import cv2
import joblib
import pandas as pd

from features import extract_features
from roi_detector import detect_rois


IMAGE_PATH = "test_images/full/Mob1_100_1_A.png"
MODEL_PATH = "models/extra_trees_classifier.pkl"
SELECTOR_PATH = "models/feature_selector.pkl"
SCALER_PATH = "models/minmax_scaler.pkl"

print("==========================================")
print("FULL IMAGE ROI PREDICTION VALIDATION")
print("==========================================")
print("FULL IMAGE:")
print(IMAGE_PATH)

model = joblib.load(MODEL_PATH)
selector = joblib.load(SELECTOR_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model components loaded successfully.")

rois = detect_rois(IMAGE_PATH)
if rois is None:
    print("ERROR: ROI detection failed.")
    raise SystemExit(1)

print(f"ROIs detected: {len(rois)}")

predictions = []
confidences = []

for i, roi in enumerate(rois, start=1):
    print()
    print("------------------------------------------")
    print(f"Processing ROI {i}")
    print("------------------------------------------")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        cv2.imwrite(temp_path, roi)
        features = extract_features(temp_path)

        if features is None:
            print("Feature extraction failed.")
            continue

        X = pd.DataFrame([features])
        X_selected = selector.transform(X)
        X_scaled = scaler.transform(X_selected)

        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = probabilities.max() * 100

        predictions.append(int(prediction))
        confidences.append(float(confidence))

        print(f"ROI {i} → {int(prediction)} mg/dL, {confidence:.2f}% confidence")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if len(predictions) == 0:
    print("ERROR: No ROI predictions were produced.")
    raise SystemExit(1)

print()
print("==========================================")
print("PREDICTIONS")
print("==========================================")
for i, (prediction, confidence) in enumerate(zip(predictions, confidences), start=1):
    print(f"ROI {i} → {prediction} mg/dL, {confidence:.2f}% confidence")

print("==========================================")