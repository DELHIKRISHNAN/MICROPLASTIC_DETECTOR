"""
OpenCV Preprocessing Pipeline for Microplastic Detection
=========================================================
Handles image preprocessing, particle segmentation via adaptive thresholding
and contour detection, ROI extraction, and handcrafted feature computation.
"""

import cv2
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────
ROI_SIZE = (64, 64)           # CNN input size
MIN_CONTOUR_AREA = 50         # Minimum contour area (pixels) to consider
MAX_CONTOUR_AREA = 50000      # Maximum contour area to filter out background


def preprocess_image(img):
    """
    Apply the full OpenCV preprocessing pipeline to an input BGR image.

    Steps:
        1. Convert to grayscale
        2. Gaussian blur for noise reduction
        3. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
        4. Adaptive thresholding for particle segmentation
        5. Morphological operations to clean up the binary mask

    Parameters
    ----------
    img : np.ndarray
        Input BGR image (H, W, 3).

    Returns
    -------
    binary_mask : np.ndarray
        Binary mask (H, W) with particles as white (255) regions.
    gray : np.ndarray
        Grayscale version of the input.
    """
    # 1. Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur (5×5 kernel)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 4. Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # 5. Morphological closing + opening to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary, gray


def detect_contours(binary_mask):
    """
    Find external contours in the binary mask and filter by area.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask from preprocessing.

    Returns
    -------
    contours : list of np.ndarray
        Filtered contours.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
            filtered.append(cnt)

    return filtered


def extract_rois(img, contours):
    """
    Extract and resize Region-of-Interest crops for each contour.

    Parameters
    ----------
    img : np.ndarray
        Original BGR image.
    contours : list of np.ndarray
        Contours from detect_contours().

    Returns
    -------
    rois : list of np.ndarray
        List of resized (64, 64, 3) ROI images.
    bboxes : list of tuple
        Bounding boxes as (x, y, w, h) for each ROI.
    """
    h_img, w_img = img.shape[:2]
    rois = []
    bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Add padding (10% each side)
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Resize to CNN input size
        roi_resized = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_AREA)
        rois.append(roi_resized)
        bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return rois, bboxes


def compute_handcrafted_features(contour, gray_roi):
    """
    Compute handcrafted geometric and texture features for a single particle.

    Features (14-dim vector):
        - Area, Perimeter, Aspect Ratio, Circularity, Solidity, Extent
        - Hu Moments (7 values, log-transformed)
        - Mean Intensity

    Parameters
    ----------
    contour : np.ndarray
        Single contour.
    gray_roi : np.ndarray
        Grayscale ROI of the particle.

    Returns
    -------
    features : np.ndarray
        Feature vector of shape (14,).
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    # Aspect ratio
    aspect_ratio = float(w) / max(h, 1)

    # Circularity: 4π × Area / Perimeter²
    circularity = (4 * np.pi * area) / max(perimeter ** 2, 1e-6)

    # Solidity: Area / Convex Hull Area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1e-6)

    # Extent: Area / Bounding Rect Area
    rect_area = w * h
    extent = area / max(rect_area, 1e-6)

    # Hu Moments (7 invariant moments, log-transformed)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform to normalize scale
    hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # Mean intensity of the ROI
    mean_intensity = np.mean(gray_roi) / 255.0

    features = np.array([
        area / 10000.0,       # Normalize area
        perimeter / 1000.0,   # Normalize perimeter
        aspect_ratio,
        circularity,
        solidity,
        extent,
        *hu_log,
        mean_intensity
    ], dtype=np.float32)

    return features


def full_pipeline(img):
    """
    Run the complete OpenCV detection pipeline on an image.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.

    Returns
    -------
    results : list of dict
        Each dict contains:
            - 'roi': np.ndarray (64, 64, 3) — resized particle crop
            - 'bbox': tuple (x, y, w, h) — bounding box in original image
            - 'features': np.ndarray (14,) — handcrafted features
            - 'contour': np.ndarray — original contour points
    """
    binary_mask, gray = preprocess_image(img)
    contours = detect_contours(binary_mask)
    rois, bboxes = extract_rois(img, contours)

    results = []
    for i, (roi, bbox, cnt) in enumerate(zip(rois, bboxes, contours)):
        # Extract grayscale ROI for feature computation
        x, y, w, h = bbox
        gray_roi = gray[y:y + h, x:x + w]
        if gray_roi.size == 0:
            gray_roi = np.zeros((1, 1), dtype=np.uint8)

        features = compute_handcrafted_features(cnt, gray_roi)
        results.append({
            'roi': roi,
            'bbox': bbox,
            'features': features,
            'contour': cnt
        })

    return results
