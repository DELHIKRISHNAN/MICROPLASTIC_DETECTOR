"""
Synthetic Microplastic Dataset Generator
=========================================
Generates 4,000+ labelled microplastic particle images across 5 classes:
    fiber, film, foam, fragment, pellet

Each image simulates a microscopy view of a fluorescently-stained
microplastic particle (Nile Red style) on a dark background.
"""

import os
import csv
import random
import numpy as np
import cv2


# ── Configuration ────────────────────────────────────────────────────────
CLASSES = ['fiber', 'film', 'foam', 'fragment', 'pellet']
IMAGES_PER_CLASS = 850       # 850 × 5 = 4,250 total images
IMG_SIZE = 64
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Fluorescence-style colors (bright on dark, mimicking Nile Red staining)
PARTICLE_COLORS = [
    (0, 255, 200),    # Cyan-green
    (0, 200, 255),    # Orange-yellow (BGR)
    (50, 255, 255),   # Yellow
    (0, 180, 255),    # Orange
    (100, 255, 200),  # Bright green
    (0, 255, 150),    # Green-cyan
    (80, 220, 255),   # Light yellow
    (0, 150, 255),    # Deep orange
]


def random_bg(size=IMG_SIZE):
    """Generate a dark noisy background simulating microscopy."""
    bg = np.random.randint(5, 30, (size, size, 3), dtype=np.uint8)
    # Add subtle Gaussian noise
    noise = np.random.normal(0, 5, (size, size, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 50).astype(np.uint8)
    return bg


def random_color():
    """Pick a random fluorescent color with some variation."""
    base = random.choice(PARTICLE_COLORS)
    variation = np.random.randint(-30, 30, 3)
    color = tuple(np.clip(np.array(base) + variation, 50, 255).astype(int).tolist())
    return color


def generate_fiber(img):
    """Generate a fiber-like particle (elongated, thin)."""
    h, w = img.shape[:2]
    color = random_color()

    # Fiber: a thin elongated shape
    cx, cy = w // 2 + random.randint(-8, 8), h // 2 + random.randint(-8, 8)
    length = random.randint(20, 50)
    thickness = random.randint(1, 4)
    angle = random.uniform(0, 180)

    # Create endpoints
    dx = int(length / 2 * np.cos(np.radians(angle)))
    dy = int(length / 2 * np.sin(np.radians(angle)))
    pt1 = (cx - dx, cy - dy)
    pt2 = (cx + dx, cy + dy)

    # Add slight curve by drawing a polyline with mid-point offset
    mid_offset_x = random.randint(-5, 5)
    mid_offset_y = random.randint(-5, 5)
    mid = (cx + mid_offset_x, cy + mid_offset_y)

    pts = np.array([pt1, mid, pt2], dtype=np.int32)
    cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    # Add glow effect
    blur_img = cv2.GaussianBlur(img, (5, 5), 2)
    mask = img > 30
    img[mask] = cv2.addWeighted(img, 0.7, blur_img, 0.3, 0)[mask]

    return img


def generate_film(img):
    """Generate a film-like particle (thin, irregular, sheet-like)."""
    h, w = img.shape[:2]
    color = random_color()

    # Film: irregular polygon with many vertices
    cx, cy = w // 2, h // 2
    num_points = random.randint(5, 9)
    points = []
    for i in range(num_points):
        angle = (2 * np.pi * i / num_points) + random.uniform(-0.4, 0.4)
        radius = random.randint(8, 25)
        px = int(cx + radius * np.cos(angle))
        py = int(cy + radius * np.sin(angle))
        points.append([px, py])

    pts = np.array(points, dtype=np.int32)

    # Semi-transparent fill
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    alpha = random.uniform(0.3, 0.6)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Thin edge
    cv2.polylines(img, [pts], True, color, 1, lineType=cv2.LINE_AA)

    return img


def generate_foam(img):
    """Generate a foam-like particle (irregular, porous texture)."""
    h, w = img.shape[:2]
    color = random_color()

    cx, cy = w // 2 + random.randint(-5, 5), h // 2 + random.randint(-5, 5)
    radius = random.randint(10, 22)

    # Outer irregular circle
    num_points = random.randint(10, 16)
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        r = radius + random.randint(-4, 4)
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        points.append([px, py])

    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)

    # Add porous holes (dark circles inside)
    for _ in range(random.randint(3, 8)):
        hole_x = cx + random.randint(-radius // 2, radius // 2)
        hole_y = cy + random.randint(-radius // 2, radius // 2)
        hole_r = random.randint(1, 4)
        dark = tuple(max(0, c - 150) for c in color)
        cv2.circle(img, (hole_x, hole_y), hole_r, dark, -1)

    # Glow
    blur_img = cv2.GaussianBlur(img, (7, 7), 3)
    mask = img > 30
    img[mask] = cv2.addWeighted(img, 0.6, blur_img, 0.4, 0)[mask]

    return img


def generate_fragment(img):
    """Generate a fragment-like particle (irregular edges, solid)."""
    h, w = img.shape[:2]
    color = random_color()

    cx, cy = w // 2 + random.randint(-5, 5), h // 2 + random.randint(-5, 5)

    # Irregular polygon
    num_points = random.randint(4, 8)
    points = []
    for i in range(num_points):
        angle = (2 * np.pi * i / num_points) + random.uniform(-0.5, 0.5)
        radius = random.randint(6, 20)
        px = int(cx + radius * np.cos(angle))
        py = int(cy + radius * np.sin(angle))
        points.append([px, py])

    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)

    # Jagged edge
    cv2.polylines(img, [pts], True, tuple(min(255, c + 50) for c in color), 1, cv2.LINE_AA)

    # Slight texture via noise in the region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    texture_noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
    for c_idx in range(3):
        channel = img[:, :, c_idx].astype(np.int16)
        channel += texture_noise[:, :, c_idx] * (mask > 0).astype(np.int16)
        img[:, :, c_idx] = np.clip(channel, 0, 255).astype(np.uint8)

    return img


def generate_pellet(img):
    """Generate a pellet-like particle (round/oval, smooth)."""
    h, w = img.shape[:2]
    color = random_color()

    cx, cy = w // 2 + random.randint(-5, 5), h // 2 + random.randint(-5, 5)
    radius_x = random.randint(8, 18)
    radius_y = random.randint(8, 18)
    angle = random.uniform(0, 360)

    cv2.ellipse(img, (cx, cy), (radius_x, radius_y), angle, 0, 360, color, -1, cv2.LINE_AA)

    # Smooth highlight (specular)
    highlight_color = tuple(min(255, c + 80) for c in color)
    cv2.ellipse(img, (cx - 2, cy - 2), (radius_x // 3, radius_y // 3),
                angle, 0, 360, highlight_color, -1, cv2.LINE_AA)

    # Glow
    blur_img = cv2.GaussianBlur(img, (7, 7), 3)
    mask = img > 30
    img[mask] = cv2.addWeighted(img, 0.65, blur_img, 0.35, 0)[mask]

    return img


# ── Generator mapping ────────────────────────────────────────────────────
GENERATORS = {
    'fiber': generate_fiber,
    'film': generate_film,
    'foam': generate_foam,
    'fragment': generate_fragment,
    'pellet': generate_pellet,
}


def apply_augmentation(img):
    """Apply random augmentation to an image."""
    # Random rotation
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                         borderMode=cv2.BORDER_REFLECT)

    # Random brightness adjustment
    brightness = random.uniform(0.7, 1.3)
    img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    # Random flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # Vertical

    # Occasional Gaussian blur
    if random.random() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate_dataset(output_dir='dataset', images_per_class=IMAGES_PER_CLASS):
    """
    Generate the full synthetic microplastic dataset.

    Parameters
    ----------
    output_dir : str
        Root directory for the dataset.
    images_per_class : int
        Number of images per class.

    Returns
    -------
    total_count : int
        Total number of images generated.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    # Create directories
    for cls in CLASSES:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    labels = []
    total_count = 0
    train_count = int(images_per_class * TRAIN_SPLIT)

    for cls in CLASSES:
        generator_fn = GENERATORS[cls]
        print(f"[dataset] Generating {images_per_class} images for class '{cls}'...")

        for i in range(images_per_class):
            img = random_bg()
            img = generator_fn(img)
            img = apply_augmentation(img)

            # Determine split
            if i < train_count:
                split = 'train'
                save_dir = train_dir
            else:
                split = 'test'
                save_dir = test_dir

            filename = f"{cls}_{i:04d}.png"
            filepath = os.path.join(save_dir, cls, filename)
            cv2.imwrite(filepath, img)

            labels.append({
                'filename': filename,
                'class': cls,
                'class_id': CLASSES.index(cls),
                'split': split,
                'path': filepath
            })
            total_count += 1

    # Write labels CSV
    csv_path = os.path.join(output_dir, 'labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'class', 'class_id', 'split', 'path'])
        writer.writeheader()
        writer.writerows(labels)

    print(f"\n[dataset] ✓ Generated {total_count} images across {len(CLASSES)} classes")
    print(f"[dataset]   Train: {train_count * len(CLASSES)} | Test: {(images_per_class - train_count) * len(CLASSES)}")
    print(f"[dataset]   Labels saved to {csv_path}")

    return total_count


if __name__ == '__main__':
    generate_dataset()
