import cv2
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_glcm_features(gray_32):
    """
    Calculate GLCM features using NumPy.

    Matches the original configuration:
    distance=1, angle=0, levels=32,
    symmetric=True, normed=True.
    """

    levels = 32

    # Horizontal pixel pairs: (row, col) and (row, col + 1)
    left = gray_32[:, :-1].astype(np.int32)
    right = gray_32[:, 1:].astype(np.int32)

    # Count gray-level pairs
    pair_indices = left * levels + right

    counts = np.bincount(
        pair_indices.ravel(),
        minlength=levels * levels
    ).reshape(levels, levels).astype(np.float64)

    # Make the matrix symmetric
    glcm = counts + counts.T

    # Normalize
    total = glcm.sum()

    if total == 0:
        return {
            "glcm_contrast": 0.0,
            "glcm_dissimilarity": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_energy": 0.0,
            "glcm_correlation": 1.0
        }

    glcm /= total

    # Gray-level coordinate matrices
    i, j = np.indices((levels, levels))
    difference = i - j

    # GLCM features
    contrast = np.sum(glcm * (difference ** 2))

    dissimilarity = np.sum(
        glcm * np.abs(difference)
    )

    homogeneity = np.sum(
        glcm / (1.0 + difference ** 2)
    )

    energy = np.sqrt(np.sum(glcm ** 2))

    # Correlation
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)

    std_i = np.sqrt(
        np.sum(((i - mean_i) ** 2) * glcm)
    )

    std_j = np.sqrt(
        np.sum(((j - mean_j) ** 2) * glcm)
    )

    if std_i > 0 and std_j > 0:
        correlation = np.sum(
            (i - mean_i) * (j - mean_j) * glcm
        ) / (std_i * std_j)
    else:
        correlation = 1.0

    return {
        "glcm_contrast": contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation
    }


def extract_features(image_path):

    # --------------------------------------------------------
    # Read ROI image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        return None

    # --------------------------------------------------------
    # Convert BGR → Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Convert pixel values to float
    # --------------------------------------------------------

    gray_float = gray.astype(np.float32)

    # --------------------------------------------------------
    # Normalize pixel intensity
    # --------------------------------------------------------

    normalized = gray_float / 255.0

    # Flatten pixels
    pixels = normalized.flatten()

    # --------------------------------------------------------
    # Basic statistical features
    # --------------------------------------------------------

    features = {}

    features["mean"] = np.mean(pixels)
    features["std"] = np.std(pixels)
    features["variance"] = np.var(pixels)

    features["min"] = np.min(pixels)
    features["max"] = np.max(pixels)

    features["median"] = np.median(pixels)

    features["q25"] = np.percentile(pixels, 25)
    features["q75"] = np.percentile(pixels, 75)

    features["skewness"] = skew(pixels)
    features["kurtosis"] = kurtosis(pixels)

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    features["energy"] = np.sum(pixels ** 2)

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist.flatten()

    probability = hist / np.sum(hist)

    probability = probability[
        probability > 0
    ]

    features["entropy"] = -np.sum(
        probability * np.log2(probability)
    )

    # --------------------------------------------------------
    # Smoothness
    # --------------------------------------------------------

    variance_gray = np.var(gray_float)

    features["smoothness"] = (
        1 - (1 / (1 + variance_gray))
    )

    # --------------------------------------------------------
    # Intensity histogram statistics
    # --------------------------------------------------------

    features["hist_mean"] = np.mean(hist)

    features["hist_std"] = np.std(hist)

    features["hist_energy"] = np.sum(
        (hist / np.sum(hist)) ** 2
    )

    # --------------------------------------------------------
    # Canny edge features
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        threshold1=50,
        threshold2=150
    )

    edge_pixels = np.sum(edges > 0)

    features["edge_count"] = edge_pixels

    features["edge_density"] = (
        edge_pixels / edges.size
    )

    # --------------------------------------------------------
    # Contour / shape features
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    features["contour_count"] = len(contours)

    if len(contours) > 0:

        areas = [
            cv2.contourArea(c)
            for c in contours
        ]

        perimeters = [
            cv2.arcLength(c, True)
            for c in contours
        ]

        features["area_mean"] = np.mean(areas)
        features["area_max"] = np.max(areas)

        features["perimeter_mean"] = np.mean(
            perimeters
        )

        features["perimeter_max"] = np.max(
            perimeters
        )

    else:

        features["area_mean"] = 0
        features["area_max"] = 0
        features["perimeter_mean"] = 0
        features["perimeter_max"] = 0

    # --------------------------------------------------------
    # GLCM texture features
    # --------------------------------------------------------

    gray_32 = (gray / 8).astype(np.uint8)

    glcm_features = calculate_glcm_features(gray_32)

    features.update(glcm_features)

    return features