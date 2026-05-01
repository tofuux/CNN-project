import cv2
import numpy as np


def _center_digit_in_square(roi: np.ndarray) -> np.ndarray:
    """
    Place the digit bounding-box content in the centre of a square canvas,
    matching how MNIST images are actually laid out.  Preserves aspect ratio
    so a thin "1" doesn't get squished into a fat blob.
    """
    h, w = roi.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = roi
    return canvas


def preprocess_image(path: str) -> np.ndarray:
    """
    Load a single-digit image from disk and return a (1, 28, 28, 1) float32
    array ready for model.predict().
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return preprocess_roi(img)


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Prepare an arbitrary greyscale ROI (digit crop) for prediction.

    Steps
    -----
    1. CLAHE  – equalise contrast so faint strokes become visible.
    2. Invert – ensure digit is WHITE on BLACK (MNIST convention).
       The inversion is conditional: we check which background is dominant.
    3. Centre  – place the digit in a square canvas (aspect-ratio safe).
    4. Resize  – to 28×28 with area interpolation (best for downscaling).
    5. Normalise – to [0, 1] float32.
    """
    # ── 1. Contrast enhancement ───────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    roi = clahe.apply(roi)

    # ── 2. Conditional inversion (white digit on black background) ────────
    # If the majority of pixels are bright, the background is light → invert.
    if np.mean(roi) > 127:
        roi = cv2.bitwise_not(roi)

    # ── 3. Aspect-ratio-safe centering ────────────────────────────────────
    roi = _center_digit_in_square(roi)

    # ── 4. Resize ─────────────────────────────────────────────────────────
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # ── 5. Normalise ──────────────────────────────────────────────────────
    roi = roi.astype("float32") / 255.0
    return roi.reshape(1, 28, 28, 1)
