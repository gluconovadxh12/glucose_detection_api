const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const chooseButton = document.getElementById("chooseButton");
const dropZone = document.getElementById("dropZone");
const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");
const removeButton = document.getElementById("removeButton");
const previewWrap = document.getElementById("previewWrap");
const imagePreview = document.getElementById("imagePreview");
const previewDimensions = document.getElementById("previewDimensions");
const detectButton = document.getElementById("detectButton");
const resultsPanel = document.getElementById("resultsPanel");
const resultGrid = document.getElementById("resultGrid");
const roiSummary = document.getElementById("roiSummary");
const statusBar = document.getElementById("statusBar");
const statusText = document.getElementById("statusText");

let selectedImage = null;
let previewUrl = null;

function setStatus(message, state = "ready") {
  statusBar.className = `status-bar status-${state}`;
  statusText.textContent = message;
}

function formatConfidence(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "--";
}

function clearPreviewUrl() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function resetInterface() {
  clearPreviewUrl();
  selectedImage = null;
  imageInput.value = "";
  selectedFile.hidden = true;
  previewWrap.hidden = true;
  resultsPanel.hidden = true;
  detectButton.disabled = true;
  resultGrid.replaceChildren();
  fileName.textContent = "No file selected";
  fileMeta.textContent = "Ready for analysis";
  previewDimensions.textContent = "";
  setStatus("Ready", "ready");
}

function displayImage(file) {
  if (!file || !file.type.startsWith("image/")) {
    resetInterface();
    setStatus("Please choose a valid image file.", "error");
    return;
  }

  clearPreviewUrl();
  selectedImage = file;
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileName.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB image selected`;
  selectedFile.hidden = false;
  previewWrap.hidden = false;
  resultsPanel.hidden = true;
  detectButton.disabled = false;
  setStatus("Ready", "ready");
}

function renderResults(payload) {
  const results = Array.isArray(payload.results) ? payload.results : [];
  resultGrid.replaceChildren();
  roiSummary.textContent = `${payload.rois_detected || results.length} ROIs Detected`;

  results.forEach((result) => {
    const card = document.createElement("article");
    card.className = "result-card";

    const roiLabel = document.createElement("div");
    roiLabel.className = "result-roi";
    roiLabel.textContent = `ROI ${result.roi}`;

    const value = document.createElement("div");
    value.className = "result-value";
    value.textContent = `${result.predicted_glucose} `;
    const unit = document.createElement("small");
    unit.textContent = result.unit || "mg/dL";
    value.appendChild(unit);

    const confidence = document.createElement("div");
    confidence.className = "confidence";
    const confidenceLabel = document.createElement("span");
    confidenceLabel.textContent = "Confidence";
    const confidenceValue = document.createElement("strong");
    confidenceValue.textContent = formatConfidence(result.confidence);
    confidence.append(confidenceLabel, confidenceValue);

    const track = document.createElement("div");
    track.className = "confidence-track";
    const fill = document.createElement("div");
    fill.className = "confidence-fill";
    const confidenceNumber = Math.max(0, Math.min(100, Number(result.confidence) || 0));
    fill.style.width = `${confidenceNumber}%`;
    track.appendChild(fill);

    card.append(roiLabel, value, confidence, track);
    resultGrid.appendChild(card);
  });

  resultsPanel.hidden = false;
}

async function detectGlucose() {
  if (!selectedImage) return;

  detectButton.disabled = true;
  setStatus("Analyzing image...", "loading");
  resultsPanel.hidden = true;

  const formData = new FormData();
  formData.append("file", selectedImage);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error("The prediction server returned an unreadable response.");
    }

    if (!response.ok) {
      throw new Error(payload.detail || "The prediction request failed.");
    }

    if (payload.rois_detected !== 4 || !Array.isArray(payload.results) || payload.results.length !== 4) {
      throw new Error("The server did not return exactly four ROI results.");
    }

    renderResults(payload);
    setStatus("Analysis completed", "success");
  } catch (error) {
    const message = error instanceof TypeError
      ? "Unable to connect to the glucose detection server. Please make sure the FastAPI server is running."
      : error.message;
    setStatus(message, "error");
  } finally {
    detectButton.disabled = !selectedImage;
  }
}

chooseButton.addEventListener("click", (event) => {
  event.preventDefault();
  imageInput.click();
});

imageInput.addEventListener("change", () => {
  displayImage(imageInput.files[0]);
});

removeButton.addEventListener("click", resetInterface);
detectButton.addEventListener("click", detectGlucose);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const file = event.dataTransfer.files[0];
  if (file) displayImage(file);
});

imagePreview.addEventListener("load", () => {
  previewDimensions.textContent = `${imagePreview.naturalWidth} × ${imagePreview.naturalHeight}px`;
});

resetInterface();
