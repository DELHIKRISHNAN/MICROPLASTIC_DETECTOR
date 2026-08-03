"""
CNN Feature Extractor + SVM Classifier for Microplastic Detection
=================================================================
CNN extracts 128-dim learned features from particle ROIs.
SVM classifies particles using concatenated CNN + handcrafted features.
"""

import os
import numpy as np
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── Class Labels ─────────────────────────────────────────────────────────
CLASS_NAMES = ['fiber', 'film', 'foam', 'fragment', 'pellet']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (64, 64, 3)


# ── CNN Architecture ─────────────────────────────────────────────────────

def build_cnn(input_shape=IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Build a lightweight CNN for microplastic particle classification.

    Architecture:
        Conv2D(32) → BatchNorm → ReLU → MaxPool
        Conv2D(64) → BatchNorm → ReLU → MaxPool
        Conv2D(128) → BatchNorm → ReLU → MaxPool
        GlobalAveragePooling → Dense(128) [feature layer]
        Dropout(0.3) → Dense(num_classes, softmax)

    Parameters
    ----------
    input_shape : tuple
        Shape of input images (H, W, C).
    num_classes : int
        Number of output classes.

    Returns
    -------
    model : keras.Model
        Compiled CNN model.
    feature_extractor : keras.Model
        Model that outputs the 128-dim feature vector.
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Feature extraction head
    x = layers.GlobalAveragePooling2D()(x)
    feature_layer = layers.Dense(128, activation='relu', name='feature_layer')(x)
    x = layers.Dropout(0.3)(feature_layer)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    # Full classification model
    model = Model(inputs, outputs, name='MicroplasticCNN')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Feature extractor (outputs 128-dim vector)
    feature_extractor = Model(inputs, feature_layer, name='FeatureExtractor')

    return model, feature_extractor


def train_cnn(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """
    Train the CNN model with early stopping.

    Parameters
    ----------
    model : keras.Model
        CNN model from build_cnn().
    X_train, y_train : np.ndarray
        Training images and labels.
    X_val, y_val : np.ndarray
        Validation images and labels.
    epochs : int
        Maximum number of epochs.
    batch_size : int
        Batch size.

    Returns
    -------
    history : keras.callbacks.History
        Training history.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ── SVM Classifier ───────────────────────────────────────────────────────

def train_svm(features_train, labels_train, features_val, labels_val):
    """
    Train an SVM classifier on combined CNN + handcrafted features.

    Uses grid search to find optimal hyperparameters.

    Parameters
    ----------
    features_train : np.ndarray
        Training feature vectors (N, D).
    labels_train : np.ndarray
        Training labels (N,).
    features_val : np.ndarray
        Validation feature vectors.
    labels_val : np.ndarray
        Validation labels.

    Returns
    -------
    svm : SVC
        Trained SVM classifier.
    scaler : StandardScaler
        Fitted feature scaler.
    accuracy : float
        Validation accuracy.
    """
    # Standardize features
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)

    # Grid search over SVM hyperparameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf']
    }

    print("[train] Running SVM grid search...")
    svm = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    svm.fit(features_train_scaled, labels_train)

    best_svm = svm.best_estimator_
    print(f"[train] Best SVM params: {svm.best_params_}")

    # Evaluate on validation set
    val_preds = best_svm.predict(features_val_scaled)
    accuracy = accuracy_score(labels_val, val_preds)
    print(f"[train] SVM Validation Accuracy: {accuracy * 100:.1f}%")
    print("\n[train] Classification Report:")
    print(classification_report(labels_val, val_preds, target_names=CLASS_NAMES))
    print("[train] Confusion Matrix:")
    print(confusion_matrix(labels_val, val_preds))

    return best_svm, scaler, accuracy


def extract_combined_features(feature_extractor, rois, handcrafted_features):
    """
    Combine CNN features with handcrafted OpenCV features.

    Parameters
    ----------
    feature_extractor : keras.Model
        CNN feature extraction model (outputs 128-dim).
    rois : np.ndarray
        Array of ROI images (N, 64, 64, 3), normalized to [0, 1].
    handcrafted_features : np.ndarray
        Array of handcrafted features (N, 14).

    Returns
    -------
    combined : np.ndarray
        Combined feature vectors (N, 142).
    """
    # CNN features
    cnn_features = feature_extractor.predict(rois, verbose=0)

    # Concatenate CNN (128-dim) + handcrafted (14-dim) = 142-dim
    combined = np.concatenate([cnn_features, handcrafted_features], axis=1)

    return combined


# ── Model Loading / Saving ───────────────────────────────────────────────

def save_models(cnn_model, feature_extractor, svm_model, scaler, save_dir='models'):
    """Save all model artifacts to disk."""
    os.makedirs(save_dir, exist_ok=True)

    cnn_path = os.path.join(save_dir, 'cnn_model.h5')
    fe_path = os.path.join(save_dir, 'feature_extractor.h5')
    svm_path = os.path.join(save_dir, 'svm_model.pkl')
    scaler_path = os.path.join(save_dir, 'scaler.pkl')

    cnn_model.save(cnn_path)
    feature_extractor.save(fe_path)
    joblib.dump(svm_model, svm_path)
    joblib.dump(scaler, scaler_path)

    print(f"[save] CNN model      → {cnn_path}")
    print(f"[save] Feature ext.   → {fe_path}")
    print(f"[save] SVM model      → {svm_path}")
    print(f"[save] Scaler         → {scaler_path}")


def load_models(model_dir='models'):
    """
    Load all model artifacts from disk.

    Returns
    -------
    feature_extractor : keras.Model
    svm_model : SVC
    scaler : StandardScaler
    """
    fe_path = os.path.join(model_dir, 'feature_extractor.h5')
    svm_path = os.path.join(model_dir, 'svm_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    feature_extractor = keras.models.load_model(fe_path, compile=False)
    svm_model = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)

    print(f"[load] Feature extractor loaded from {fe_path}")
    print(f"[load] SVM model loaded from {svm_path}")
    print(f"[load] Scaler loaded from {scaler_path}")

    return feature_extractor, svm_model, scaler
