"""
train_models.py
Deep Learning Fruit Maturity Detection: Standard CNN vs Ordinal Regression CNN.

4 Ordered Maturity Classes:
0: unripe   (Level 0)
1: ripe     (Level 1)
2: overripe (Level 2)
3: rotten   (Level 3)

Ordinal Formulation:
K = 4 classes -> K - 1 = 3 binary heads:
Head 1: P(Y >= 1) - At least Ripe
Head 2: P(Y >= 2) - At least Overripe
Head 3: P(Y >= 3) - Rotten / Spoiled
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_absolute_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = Path(r"C:\projects\Fruit-Ripeness-Detection")
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 18
CLASSES = ["unripe", "ripe", "overripe", "rotten"]


def load_dataset_split(split_name: str):
    split_dir = DATASET_DIR / split_name
    images, labels, paths = [], [], []

    for class_idx, class_name in enumerate(CLASSES):
        class_folder = split_dir / class_name
        for img_file in sorted(class_folder.glob("*.jpg")):
            with Image.open(img_file) as img:
                img_rgb = img.convert("RGB").resize(IMAGE_SIZE)
                arr = np.array(img_rgb, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(class_idx)
                paths.append(str(img_file))

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Loaded {split_name} split: {len(X)} samples with shape {X.shape}")
    return X, y, paths


def to_ordinal_targets(y: np.ndarray) -> np.ndarray:
    """
    Transforms integer class labels {0, 1, 2, 3} into 3 cumulative binary targets:
    unripe   (0) -> [0, 0, 0]
    ripe     (1) -> [1, 0, 0]
    overripe (2) -> [1, 1, 0]
    rotten   (3) -> [1, 1, 1]
    """
    targets = np.zeros((len(y), 3), dtype=np.float32)
    targets[:, 0] = (y >= 1).astype(np.float32)
    targets[:, 1] = (y >= 2).astype(np.float32)
    targets[:, 2] = (y >= 3).astype(np.float32)
    return targets


def decode_ordinal_predictions(probs: np.ndarray):
    """
    Decodes sigmoid outputs [p1, p2, p3] into 4 class probabilities.
    Enforces monotonic probability consistency: p1 >= p2 >= p3.
    """
    p1 = probs[:, 0]
    p2 = probs[:, 1]
    p3 = probs[:, 2]

    # Enforce top-down cumulative monotonic consistency: p1 >= p2 >= p3
    p1_mono = p1
    p2_mono = np.minimum(p1_mono, p2)
    p3_mono = np.minimum(p2_mono, p3)

    p_unripe = np.clip(1.0 - p1_mono, 0.0, 1.0)
    p_ripe = np.clip(p1_mono - p2_mono, 0.0, 1.0)
    p_overripe = np.clip(p2_mono - p3_mono, 0.0, 1.0)
    p_rotten = np.clip(p3_mono, 0.0, 1.0)

    class_probs = np.stack([p_unripe, p_ripe, p_overripe, p_rotten], axis=-1)
    row_sums = class_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    class_probs = class_probs / row_sums

    pred_classes = np.argmax(class_probs, axis=1)
    return pred_classes, class_probs


def build_backbone():
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomContrast(0.1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_features")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_shared")(x)
    x = layers.Dropout(0.35, name="dropout_shared")(x)
    return inputs, x


def build_standard_cnn():
    inputs, features = build_backbone()
    outputs = layers.Dense(4, activation="softmax", name="ripeness_output")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Standard_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_ordinal_cnn():
    inputs, features = build_backbone()
    outputs = layers.Dense(3, activation="sigmoid", name="ordinal_thresholds")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Ordinal_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


def plot_and_save_visualizations(history_std, history_ord, y_test, std_preds, ord_preds):
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history_std.history["accuracy"], label="Standard CNN (Train Acc)", color="#2a9d8f", linewidth=2.2)
    plt.plot(history_std.history["val_accuracy"], label="Standard CNN (Val Acc)", color="#2a9d8f", linestyle="--", linewidth=2.0)
    plt.plot(history_ord.history["binary_accuracy"], label="Ordinal CNN (Train Binary Acc)", color="#e76f51", linewidth=2.2)
    plt.plot(history_ord.history["val_binary_accuracy"], label="Ordinal CNN (Val Binary Acc)", color="#e76f51", linestyle="--", linewidth=2.0)
    plt.title("Model Training & Validation Accuracy over Epochs (4-Class)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([0.4, 1.05])
    plt.legend(frameon=True, facecolor="white", shadow=True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "accuracy.png", dpi=300)
    plt.close()

    # 2. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history_std.history["loss"], label="Standard CNN (Train Loss)", color="#264653", linewidth=2.2)
    plt.plot(history_std.history["val_loss"], label="Standard CNN (Val Loss)", color="#264653", linestyle="--", linewidth=2.0)
    plt.plot(history_ord.history["loss"], label="Ordinal CNN (Train BCE Loss)", color="#f4a261", linewidth=2.2)
    plt.plot(history_ord.history["val_loss"], label="Ordinal CNN (Val BCE Loss)", color="#f4a261", linestyle="--", linewidth=2.0)
    plt.title("Training and Validation Loss Trajectories (4-Class)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(frameon=True, facecolor="white", shadow=True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "loss.png", dpi=300)
    plt.close()

    # 3. Side-by-Side 4x4 Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    cm_std = confusion_matrix(y_test, std_preds)
    cm_ord = confusion_matrix(y_test, ord_preds)

    sns.heatmap(cm_std, annot=True, fmt="d", cmap="Greens", ax=axes[0],
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,
                annot_kws={"size": 15, "weight": "bold"})
    axes[0].set_title("Standard CNN Confusion Matrix", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Class", fontsize=11)

    sns.heatmap(cm_ord, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,
                annot_kws={"size": 15, "weight": "bold"})
    axes[1].set_title("Ordinal Regression CNN Confusion Matrix", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Error Distance Distribution
    std_errors = np.abs(std_preds - y_test)
    ord_errors = np.abs(ord_preds - y_test)

    df_err = pd.DataFrame({
        "Error Distance": [0, 1, 2, 3] * 2,
        "Percentage (%)": [
            np.mean(std_errors == 0) * 100, np.mean(std_errors == 1) * 100, np.mean(std_errors == 2) * 100, np.mean(std_errors == 3) * 100,
            np.mean(ord_errors == 0) * 100, np.mean(ord_errors == 1) * 100, np.mean(ord_errors == 2) * 100, np.mean(ord_errors == 3) * 100
        ],
        "Model": ["Standard CNN"] * 4 + ["Ordinal CNN"] * 4
    })

    plt.figure(figsize=(10, 5))
    bar_plot = sns.barplot(data=df_err, x="Error Distance", y="Percentage (%)", hue="Model", palette=["#2a9d8f", "#e76f51"])
    plt.title("Error Distance Distribution (Ordinal Penalty Evaluation)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Absolute Error Distance (|Predicted - Actual|)", fontsize=11)
    plt.ylabel("Sample Percentage (%)", fontsize=11)
    plt.ylim([0, 105])
    for p in bar_plot.patches:
        height = p.get_height()
        if height > 0:
            bar_plot.annotate(f"{height:.1f}%",
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                              textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "error_distance_distribution.png", dpi=300)
    plt.close()


def train_and_evaluate():
    print("=" * 65)
    print("STARTING 4-CLASS MODEL TRAINING & COMPARATIVE BENCHMARK")
    print("=" * 65)

    X_train, y_train, _ = load_dataset_split("train")
    X_val, y_val, _ = load_dataset_split("validation")
    X_test, y_test, _ = load_dataset_split("test")

    y_train_ord = to_ordinal_targets(y_train)
    y_val_ord = to_ordinal_targets(y_val)

    # 1. Model A: Standard CNN
    print("\n[1/4] Training Standard CNN (4 Classes)...")
    std_model = build_standard_cnn()
    callbacks_std = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    ]
    history_std = std_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_std,
        verbose=1
    )
    std_model_path = MODELS_DIR / "standard_cnn.keras"
    std_model.save(std_model_path)
    print(f"[SUCCESS] Standard CNN saved to {std_model_path}")

    # 2. Model B: Ordinal Regression CNN
    print("\n[2/4] Training Ordinal Regression CNN (3 Binary Heads)...")
    ord_model = build_ordinal_cnn()
    callbacks_ord = [
        keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=6, restore_best_weights=True)
    ]
    history_ord = ord_model.fit(
        X_train, y_train_ord,
        validation_data=(X_val, y_val_ord),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_ord,
        verbose=1
    )
    ord_model_path = MODELS_DIR / "ordinal_cnn.keras"
    ord_model.save(ord_model_path)
    print(f"[SUCCESS] Ordinal CNN saved to {ord_model_path}")

    # 3. Evaluation on Test Dataset
    print("\n[3/4] Evaluating Both Models on Test Set...")
    std_raw_probs = std_model.predict(X_test, verbose=0)
    std_preds = np.argmax(std_raw_probs, axis=1)

    ord_raw_probs = ord_model.predict(X_test, verbose=0)
    ord_preds, ord_class_probs = decode_ordinal_predictions(ord_raw_probs)

    metrics = {
        "standard_cnn": {
            "accuracy": float(np.mean(std_preds == y_test)),
            "mae": float(mean_absolute_error(y_test, std_preds)),
            "precision_macro": float(precision_score(y_test, std_preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, std_preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, std_preds, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, std_preds).tolist(),
            "severe_error_rate_pct": float(np.mean(np.abs(std_preds - y_test) >= 2) * 100)
        },
        "ordinal_cnn": {
            "accuracy": float(np.mean(ord_preds == y_test)),
            "mae": float(mean_absolute_error(y_test, ord_preds)),
            "precision_macro": float(precision_score(y_test, ord_preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, ord_preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, ord_preds, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, ord_preds).tolist(),
            "severe_error_rate_pct": float(np.mean(np.abs(ord_preds - y_test) >= 2) * 100)
        }
    }

    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SUCCESS] Metrics saved to {metrics_path}")

    comp_df = pd.DataFrame([
        {"Metric": "Accuracy", "Standard CNN": metrics["standard_cnn"]["accuracy"] * 100, "Ordinal CNN": metrics["ordinal_cnn"]["accuracy"] * 100},
        {"Metric": "F1-Score (Macro)", "Standard CNN": metrics["standard_cnn"]["f1_macro"] * 100, "Ordinal CNN": metrics["ordinal_cnn"]["f1_macro"] * 100},
        {"Metric": "Precision (Macro)", "Standard CNN": metrics["standard_cnn"]["precision_macro"] * 100, "Ordinal CNN": metrics["ordinal_cnn"]["precision_macro"] * 100},
        {"Metric": "Recall (Macro)", "Standard CNN": metrics["standard_cnn"]["recall_macro"] * 100, "Ordinal CNN": metrics["ordinal_cnn"]["recall_macro"] * 100},
    ])

    plt.figure(figsize=(10, 5.5))
    x_indices = np.arange(len(comp_df))
    width = 0.35
    plt.bar(x_indices - width/2, comp_df["Standard CNN"], width, label="Standard CNN", color="#2a9d8f")
    plt.bar(x_indices + width/2, comp_df["Ordinal CNN"], width, label="Ordinal CNN", color="#e76f51")
    plt.xticks(x_indices, comp_df["Metric"], fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=12)
    plt.title("Comparative Performance Benchmark (4 Classes)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim([0, 110])
    plt.legend(frameon=True, facecolor="white", shadow=True)
    for i in x_indices:
        plt.text(i - width/2, comp_df["Standard CNN"].iloc[i] + 2, f"{comp_df['Standard CNN'].iloc[i]:.1f}%", ha='center', fontweight='bold', fontsize=9)
        plt.text(i + width/2, comp_df["Ordinal CNN"].iloc[i] + 2, f"{comp_df['Ordinal CNN'].iloc[i]:.1f}%", ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "comparison.png", dpi=300)
    plt.close()

    # 4. Plots
    print("\n[4/4] Generating Visualizations...")
    plot_and_save_visualizations(history_std, history_ord, y_test, std_preds, ord_preds)

    print("\n" + "=" * 65)
    print("FINAL 4-CLASS BENCHMARK RESULTS SUMMARY:")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Standard CNN':<16} | {'Ordinal CNN':<16}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {metrics['standard_cnn']['accuracy']*100:.2f}%{'':<10} | {metrics['ordinal_cnn']['accuracy']*100:.2f}%")
    print(f"{'Mean Absolute Error (MAE)':<25} | {metrics['standard_cnn']['mae']:.4f}{'':<10} | {metrics['ordinal_cnn']['mae']:.4f}")
    print(f"{'F1-Score (Macro)':<25} | {metrics['standard_cnn']['f1_macro']*100:.2f}%{'':<10} | {metrics['ordinal_cnn']['f1_macro']*100:.2f}%")
    print(f"{'Severe Error Rate (>=2)':<25} | {metrics['standard_cnn']['severe_error_rate_pct']:.2f}%{'':<10} | {metrics['ordinal_cnn']['severe_error_rate_pct']:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate()
