import numpy as np
import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.makedirs("outputs", exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────
DATA_DIR = "archive"   # <-- your folder from screenshot
NUM_CLASSES = 62

print("Loading EMNIST CSV dataset...")

# ── LOAD DATA ────────────────────────────────────────────
train_df = pd.read_csv(f"{DATA_DIR}/emnist-byclass-train.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/emnist-byclass-test.csv")

# Labels
y_train = train_df.iloc[:, 0].values
y_test  = test_df.iloc[:, 0].values

# Pixels
x_train = train_df.iloc[:, 1:].values
x_test  = test_df.iloc[:, 1:].values

# Reshape to 28x28
x_train = x_train.reshape(-1, 28, 28)
x_test  = x_test.reshape(-1, 28, 28)

# FIX ROTATION (VERY IMPORTANT)
x_train = np.transpose(x_train, (0, 2, 1))
x_test  = np.transpose(x_test,  (0, 2, 1))

# Normalize + reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1,  28, 28, 1).astype("float32") / 255.0

# One-hot encode
y_train = to_categorical(y_train, NUM_CLASSES)
y_test  = to_categorical(y_test,  NUM_CLASSES)

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")

# ── DATA AUGMENTATION ────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05
)

datagen.fit(x_train)

# ── MODEL ────────────────────────────────────────────────
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# ── COMPILE ──────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── CALLBACKS ────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("outputs/best_model.keras", save_best_only=True, monitor='val_accuracy')
]

# ── TRAIN ────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS     = 10

model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# ── EVALUATE ─────────────────────────────────────────────
model.evaluate(x_test, y_test)

# ── SAVE ─────────────────────────────────────────────────
model.save("outputs/best_model.keras")

print("Model training complete and saved!")