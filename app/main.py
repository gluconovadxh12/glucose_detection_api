from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import joblib
import pandas as pd
import tempfile
import os
from pathlib import Path
from app.features import extract_features
from app.roi_detector import detect_rois


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Glucose Detection API",
    description="API for smartphone image-based glucose detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "extra_trees_classifier.pkl"
SELECTOR_PATH = BASE_DIR / "models" / "feature_selector.pkl"
SCALER_PATH = BASE_DIR / "models" / "minmax_scaler.pkl"


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

model = joblib.load(MODEL_PATH)
selector = joblib.load(SELECTOR_PATH)
scaler = joblib.load(SCALER_PATH)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Glucose Detection API is running",
        "version": "1.0.0"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_selector_loaded": selector is not None,
        "scaler_loaded": scaler is not None,
        "model_classes": len(model.classes_)
    }
# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate uploaded file
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )


    temp_path = None


    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image_data = await file.read()


        # ----------------------------------------------------
        # Create temporary image file
        # ----------------------------------------------------

        suffix = (
            os.path.splitext(file.filename or "")[1]
            or ".png"
        )


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(image_data)

            temp_path = temp_file.name


        # ----------------------------------------------------
        # Detect exactly four physical ROI images
        # ----------------------------------------------------

        try:
            rois = detect_rois(temp_path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to detect four ROIs: {exc}"
            ) from exc

        if rois is None or len(rois) != 4:
            raise HTTPException(
                status_code=400,
                detail="Unable to detect exactly four ROIs in the uploaded image."
            )

        results = []
        for roi_index, roi in enumerate(rois, start=1):
            roi_temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".png"
                ) as roi_temp_file:
                    roi_temp_path = roi_temp_file.name

                if not cv2.imwrite(roi_temp_path, roi):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to save ROI {roi_index} for processing."
                    )

                features = extract_features(roi_temp_path)
                if features is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to extract features from ROI {roi_index}."
                    )

                X = pd.DataFrame([features])
                X_selected = selector.transform(X)
                X_scaled = scaler.transform(X_selected)
                prediction = model.predict(X_scaled)[0]
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = float(probabilities.max() * 100)

                results.append({
                    "roi": roi_index,
                    "predicted_glucose": int(prediction),
                    "unit": "mg/dL",
                    "confidence": round(confidence, 2)
                })
            finally:
                if roi_temp_path and os.path.exists(roi_temp_path):
                    os.remove(roi_temp_path)

        return {
            "filename": file.filename,
            "rois_detected": len(rois),
            "results": results,
            "processing": {
                "features_extracted_per_roi": len(features),
                "features_selected": int(X_selected.shape[1])
            }
        }


    finally:

        # ----------------------------------------------------
        # Delete temporary image
        # ----------------------------------------------------

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)