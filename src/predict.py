import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

os.makedirs("outputs", exist_ok=True)

# ── Label map: EMNIST byclass order ──────────────────────────────────────────
# Classes 0-9 → digits, 10-35 → A-Z, 36-61 → a-z
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "outputs/best_model.keras"
IMAGE_PATH      = sys.argv[1] if len(sys.argv) > 1 else "test.png"
CONF_THRESHOLD  = 0.55
DEBUG           = True

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

# ── Load & preprocess image ───────────────────────────────────────────────────
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Upscale small images
min_dim = 400
scale   = max(1.0, min_dim / min(img_gray.shape))
if scale > 1.0:
    img_gray = cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img_bgr  = cv2.resize(img_bgr,  None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# Threshold
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
thresh  = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=15, C=8
)

# ── Helper: preprocess a single character ROI ─────────────────────────────────
def preprocess_char(roi):
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype("float32") / 255.0
    return roi.reshape(1, 28, 28, 1)

# ── Helper: predict a single character ROI ───────────────────────────────────
def predict_char(roi):
    processed = preprocess_char(roi)
    probs     = model.predict(processed, verbose=0)[0]
    idx       = int(np.argmax(probs))
    conf      = float(probs[idx])
    char      = LABELS[idx] if conf >= CONF_THRESHOLD else "?"
    return char, conf, idx

# ── Step 1: Find LINES (horizontal bands of text) ────────────────────────────
# Project pixel rows to find where text lives vertically
h_proj   = np.sum(thresh, axis=1)
in_line  = h_proj > thresh.shape[1] * 0.01   # rows with >1% ink

lines    = []
start    = None
for i, val in enumerate(in_line):
    if val and start is None:
        start = i
    elif not val and start is not None:
        lines.append((start, i))
        start = None
if start is not None:
    lines.append((start, thresh.shape[0]))

# ── Step 2: For each line, find WORDS, then CHARACTERS ───────────────────────
debug_img   = img_bgr.copy() if DEBUG else None
full_result = []

for (line_y1, line_y2) in lines:
    line_roi   = thresh[line_y1:line_y2, :]
    line_h     = line_y2 - line_y1

    # Find word blobs using horizontal dilation (joins nearby chars in a word)
    word_kernel = np.ones((1, max(10, line_h // 2)), np.uint8)
    word_dilated = cv2.dilate(line_roi, word_kernel, iterations=1)
    word_contours, _ = cv2.findContours(word_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_boxes = sorted([cv2.boundingRect(c) for c in word_contours], key=lambda b: b[0])

    line_words = []

    for (wx, wy, ww, wh) in word_boxes:
        if ww < 5 or wh < 5:
            continue

        word_roi = thresh[line_y1 + wy : line_y1 + wy + wh, wx : wx + ww]

        # Find individual characters inside this word using vertical dilation
        char_kernel  = np.ones((line_h, 1), np.uint8)
        char_dilated = cv2.dilate(word_roi, char_kernel, iterations=1)
        char_contours, _ = cv2.findContours(char_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = sorted([cv2.boundingRect(c) for c in char_contours], key=lambda b: b[0])

        word_str = ""

        for (cx, cy, cw, ch) in char_boxes:
            min_char_h = line_h * 0.3
            if cw < 4 or ch < min_char_h:
                continue

            pad  = max(3, int(min(cw, ch) * 0.1))
            cx1  = max(0, cx - pad)
            cy1  = max(0, cy - pad)
            cx2  = min(word_roi.shape[1], cx + cw + pad)
            cy2  = min(word_roi.shape[0], cy + ch + pad)

            char_roi          = word_roi[cy1:cy2, cx1:cx2]
            char_label, conf, idx = predict_char(char_roi)
            word_str          += char_label

            # Debug: draw character boxes
            if DEBUG:
                abs_x1 = wx + cx1
                abs_y1 = line_y1 + wy + cy1
                abs_x2 = wx + cx2
                abs_y2 = line_y1 + wy + cy2
                colour = (0, 200, 0) if conf >= CONF_THRESHOLD else (0, 0, 220)
                cv2.rectangle(debug_img, (abs_x1, abs_y1), (abs_x2, abs_y2), colour, 1)
                cv2.putText(debug_img, f"{char_label}({conf:.0%})",
                            (abs_x1, max(abs_y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        if word_str:
            line_words.append(word_str)

        # Debug: draw word boxes in blue
        if DEBUG:
            cv2.rectangle(debug_img,
                          (wx, line_y1 + wy),
                          (wx + ww, line_y1 + wy + wh),
                          (200, 100, 0), 2)

    if line_words:
        full_result.append(" ".join(line_words))

# ── Output ────────────────────────────────────────────────────────────────────
if DEBUG:
    cv2.imwrite("outputs/debug.png", debug_img)
    print("Debug image saved → outputs/debug.png")

final_text = "\n".join(full_result)
print(f"\nPredicted text:\n{final_text}")