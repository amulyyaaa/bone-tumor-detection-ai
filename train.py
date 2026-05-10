"""
Bone Tumor Detection - Model Training Script
Uses MobileNetV2 Transfer Learning for binary classification (Tumor / No Tumor)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import json

# ─── CONFIG ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
DATASET_DIR = "dataset"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "bone_tumor_model.h5")
HISTORY_PATH= os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── DATA GENERATORS ────────────────────────────────────────────────────────
print("\n[1/5] Loading and augmenting dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print(f"  Train samples : {train_generator.samples}")
print(f"  Val samples   : {val_generator.samples}")
print(f"  Classes       : {train_generator.class_indices}")

# Save class mapping
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

# ─── BUILD MODEL ────────────────────────────────────────────────────────────
print("\n[2/5] Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─── CALLBACKS ──────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

# ─── TRAIN ──────────────────────────────────────────────────────────────────
print("\n[3/5] Training model...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save history for later visualization
with open(HISTORY_PATH, "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

# ─── FINE-TUNING (PHASE 2) ──────────────────────────────────────────────────
print("\n[4/5] Fine-tuning top layers...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ───────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating model...")

val_generator.reset()
preds = (model.predict(val_generator) > 0.5).astype(int).flatten()
true_labels = val_generator.classes

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=["No Tumor", "Tumor"]))

cm = confusion_matrix(true_labels, preds)
print("Confusion Matrix:")
print(cm)

# ─── PLOT TRAINING CURVES ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"], label="Train Acc", color="#0D9488")
ax1.plot(history.history["val_accuracy"], label="Val Acc", color="#F97316", linestyle="--")
ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"], label="Train Loss", color="#0D9488")
ax2.plot(history.history["val_loss"], label="Val Loss", color="#F97316", linestyle="--")
ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
print(f"\nTraining curves saved → {MODEL_DIR}/training_curves.png")
print(f"Model saved          → {MODEL_PATH}")
print("\n✅  Training complete!")
