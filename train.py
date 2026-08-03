"""
Training Script — Microplastic Detection Pipeline
===================================================
End-to-end training: dataset generation → CNN training → SVM training.

Pipeline:
    1. Generate synthetic dataset (4,000+ images)
    2. Train CNN classifier on particle ROIs
    3. Extract CNN features for all images
    4. Compute handcrafted OpenCV features
    5. Train SVM on combined features
    6. Evaluate → target 94%+ test accuracy
    7. Save all model artifacts
"""

import os
import sys
import glob
import numpy as np
import cv2
import time

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from generate_dataset import generate_dataset, CLASSES, IMG_SIZE
from pipeline.models import (
    build_cnn, train_cnn, train_svm,
    extract_combined_features, save_models, CLASS_NAMES
)
from pipeline.preprocessing import compute_handcrafted_features


def load_images_from_dir(data_dir):
    """
    Load all images and labels from a class-structured directory.

    Parameters
    ----------
    data_dir : str
        Path to directory with subdirectories per class.

    Returns
    -------
    images : np.ndarray
        Array of images (N, 64, 64, 3), normalized to [0, 1].
    labels : np.ndarray
        Integer class labels (N,).
    raw_images : list
        Original uint8 images (for feature extraction).
    """
    images = []
    labels = []
    raw_images = []

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(glob.glob(os.path.join(cls_dir, '*.png')))
        for fpath in files:
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            raw_images.append(img.copy())
            images.append(img.astype(np.float32) / 255.0)
            labels.append(cls_idx)

    return np.array(images), np.array(labels), raw_images


def compute_all_handcrafted_features(raw_images):
    """
    Compute handcrafted features for all images.

    For synthetic images, we create a contour from the non-background
    pixels and compute features from that.
    """
    all_features = []

    for img in raw_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to get particle mask
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use the largest contour
            cnt = max(contours, key=cv2.contourArea)
            features = compute_handcrafted_features(cnt, gray)
        else:
            # Fallback: zeros
            features = np.zeros(14, dtype=np.float32)

        all_features.append(features)

    return np.array(all_features)


def main():
    print("=" * 60)
    print("  MICROPLASTIC DETECTION — TRAINING PIPELINE")
    print("  OpenCV + CNN + SVM")
    print("=" * 60)

    dataset_dir = os.path.join(BASE_DIR, 'dataset')
    model_dir = os.path.join(BASE_DIR, 'models')

    # --- Step 1: Generate Dataset ------------------------------------
    print("\n" + "-" * 50)
    print("STEP 1: Generating Synthetic Dataset")
    print("-" * 50)

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    # Check if dataset already exists
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        existing = sum(len(glob.glob(os.path.join(train_dir, c, '*.png'))) for c in CLASSES)
        existing += sum(len(glob.glob(os.path.join(test_dir, c, '*.png'))) for c in CLASSES)
        if existing > 4000:
            print(f"[train] Dataset already exists ({existing} images). Skipping generation.")
        else:
            generate_dataset(dataset_dir)
    else:
        generate_dataset(dataset_dir)

    # --- Step 2: Load Data -------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 2: Loading Training & Test Data")
    print("-" * 50)

    X_train, y_train, raw_train = load_images_from_dir(train_dir)
    X_test, y_test, raw_test = load_images_from_dir(test_dir)

    print(f"[train] Training set:   {len(X_train)} images")
    print(f"[train] Test set:       {len(X_test)} images")
    print(f"[train] Classes:        {CLASSES}")
    print(f"[train] Image shape:    {X_train[0].shape}")

    # Class distribution
    for i, cls in enumerate(CLASSES):
        n_train = np.sum(y_train == i)
        n_test = np.sum(y_test == i)
        print(f"  [{cls:>10}]  train={n_train}  test={n_test}")

    # --- Step 3: Train CNN -------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 3: Training CNN Classifier")
    print("-" * 50)

    cnn_model, feature_extractor = build_cnn()
    cnn_model.summary()

    t0 = time.time()
    history = train_cnn(
        cnn_model, X_train, y_train, X_test, y_test,
        epochs=25, batch_size=32
    )
    cnn_time = time.time() - t0
    print(f"\n[train] CNN training completed in {cnn_time:.1f}s")

    # CNN standalone accuracy
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"[train] CNN Test Accuracy: {cnn_acc * 100:.1f}%")

    # --- Step 4: Compute Handcrafted Features ------------------------
    print("\n" + "-" * 50)
    print("STEP 4: Computing Handcrafted OpenCV Features")
    print("-" * 50)

    t0 = time.time()
    hc_train = compute_all_handcrafted_features(raw_train)
    hc_test = compute_all_handcrafted_features(raw_test)
    hc_time = time.time() - t0
    print(f"[train] Handcrafted features computed in {hc_time:.1f}s")
    print(f"[train] Feature vector size: {hc_train.shape[1]} dims")

    # --- Step 5: Extract Combined Features ---------------------------
    print("\n" + "-" * 50)
    print("STEP 5: Extracting Combined CNN + Handcrafted Features")
    print("-" * 50)

    combined_train = extract_combined_features(feature_extractor, X_train, hc_train)
    combined_test = extract_combined_features(feature_extractor, X_test, hc_test)
    print(f"[train] Combined feature vector: {combined_train.shape[1]} dims")
    print(f"        (CNN: 128 + Handcrafted: {hc_train.shape[1]})")

    # --- Step 6: Train SVM -------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 6: Training SVM Classifier")
    print("-" * 50)

    t0 = time.time()
    svm_model, scaler, svm_accuracy = train_svm(
        combined_train, y_train, combined_test, y_test
    )
    svm_time = time.time() - t0
    print(f"\n[train] SVM training completed in {svm_time:.1f}s")

    # --- Step 7: Save Models -----------------------------------------
    print("\n" + "-" * 50)
    print("STEP 7: Saving Model Artifacts")
    print("-" * 50)

    save_models(cnn_model, feature_extractor, svm_model, scaler, model_dir)

    # --- Final Report ------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Dataset:            {len(X_train) + len(X_test)} images ({len(CLASSES)} classes)")
    print(f"  CNN Test Accuracy:  {cnn_acc * 100:.1f}%")
    print(f"  SVM Test Accuracy:  {svm_accuracy * 100:.1f}%")
    print(f"  Target Accuracy:    94.0%")
    print(f"  Status:             {'✓ TARGET MET' if svm_accuracy >= 0.94 else '○ Below target'}")
    print(f"  Models saved to:    {model_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
