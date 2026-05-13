"""
MÓDULO 7 — Entrenamiento LSTM + comparativa vs CNN
Panel de domótica por voz — Raspberry Pi 4

Diseña, entrena y exporta una red LSTM que trata los frames MFCC
como una secuencia temporal, y genera una comparativa detallada
contra el modelo CNN del módulo 5.

Arquitectura LSTM:
    Input  (125, 39)          ← (T frames, C canales) = secuencia temporal
    → Bidirectional LSTM(64)  ← captura contexto pasado Y futuro
    → Dropout(0.4)
    → Dense(64) + ReLU
    → Dropout(0.3)
    → Dense(8)  + Softmax

CNN recibe (125, 39) también → comparativa justa con mismo input.

Uso:
    python train_lstm.py
    python train_lstm.py --splits splits --output models --epochs 60

Salida (en models/):
    lstm_model.keras
    lstm_model.tflite
    models/reports/
        lstm_training_curves.png
        lstm_confusion_matrix.png
        lstm_vs_cnn_comparison.png    ← comparativa visual completa
        lstm_metrics_report.txt
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

TRAIN_CFG = {
    "epochs"        : 60,
    "batch_size"    : 32,
    "learning_rate" : 1e-3,
    "lstm_units"    : 64,       # unidades por dirección (Bidirectional → 128 total)
    "dense_units"   : 64,
    "dropout_lstm"  : 0.4,
    "dropout_dense" : 0.3,
    "random_seed"   : 42,
    # Early stopping
    "es_patience"   : 12,
    "es_monitor"    : "val_accuracy",
    # Reduce LR
    "rlr_patience"  : 6,
    "rlr_factor"    : 0.5,
    "rlr_min_lr"    : 1e-6,
}

# ─────────────────────────────────────────────
# REPRODUCIBILIDAD
# ─────────────────────────────────────────────

def set_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

def load_splits(splits_dir):
    """
    Carga los arrays y transpone a (N, T, C) = (N, 125, 39).
    Esta orientación es la correcta para LSTM: cada paso temporal
    es un frame de 39 features MFCC+delta+delta².
    """
    files = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    data  = {}
    for name in files:
        path = os.path.join(splits_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No encontrado: {path}\n"
                "Ejecuta prepare_splits.py (módulo 4) primero."
            )
        data[name] = np.load(path)

    le_path = os.path.join(splits_dir, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # (N, C, T) → (N, T, C): T=125 pasos, C=39 features por paso
    for key in ["X_train", "X_val", "X_test"]:
        data[key] = np.transpose(data[key], (0, 2, 1))  # (N, 125, 39)

    return data, le

# ─────────────────────────────────────────────
# ARQUITECTURA LSTM
# ─────────────────────────────────────────────

def build_lstm(input_shape, n_classes, cfg):
    """
    Construye una LSTM bidireccional para clasificación de secuencias MFCC.

    ¿Por qué Bidirectional?
    ─────────────────────────────────────────────────────────────────────
    Una LSTM unidireccional procesa la secuencia de izquierda a derecha
    (frame 0 → frame 124). Para reconocimiento de voz esto es una
    limitación: el contexto de los últimos frames ayuda a interpretar
    los primeros (ej. la vocal final de "enciende" aclara la consonante
    inicial). Bidirectional LSTM(64) corre la secuencia en ambas
    direcciones y concatena los estados ocultos → 128 valores totales.

    ¿Por qué una sola capa LSTM y no dos apiladas?
    ─────────────────────────────────────────────────────────────────────
    Con datasets pequeños (<2000 muestras/clase), apilar LSTMs introduce
    demasiados parámetros y sobreajusta. Una capa Bi-LSTM(64) con
    Dropout(0.4) es suficiente y entrena en <20 min en CPU.

    input_shape: (T, C) = (125, 39)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    tf.random.set_seed(cfg["random_seed"])

    inp = layers.Input(shape=input_shape, name="mfcc_sequence")

    # ── LSTM bidireccional ────────────────────
    x = layers.Bidirectional(
        layers.LSTM(
            cfg["lstm_units"],
            return_sequences = False,    # solo necesitamos el estado final
            kernel_regularizer = regularizers.l2(1e-4),
            recurrent_regularizer = regularizers.l2(1e-4),
        ),
        name = "bi_lstm"
    )(inp)                               # salida: (N, 128)

    # ── Cabeza clasificadora ──────────────────
    x   = layers.Dropout(cfg["dropout_lstm"],  name="drop1")(x)
    x   = layers.Dense(cfg["dense_units"], activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="dense1")(x)
    x   = layers.Dropout(cfg["dropout_dense"], name="drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="BiLSTM_VoiceCommand")
    return model

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

def build_callbacks(models_dir, cfg):
    from tensorflow.keras import callbacks
    return [
        callbacks.EarlyStopping(
            monitor              = cfg["es_monitor"],
            patience             = cfg["es_patience"],
            restore_best_weights = True,
            verbose              = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = cfg["rlr_factor"],
            patience = cfg["rlr_patience"],
            min_lr   = cfg["rlr_min_lr"],
            verbose  = 1,
        ),
        callbacks.ModelCheckpoint(
            filepath       = os.path.join(models_dir, "lstm_best.keras"),
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 0,
        ),
    ]

# ─────────────────────────────────────────────
# EXPORTAR A TFLITE
# ─────────────────────────────────────────────

def export_tflite(model, output_path, X_train_sample=None):
    """
    Exporta el modelo Keras a TFLite con cuantización dinámica de pesos.
    Nota: la LSTM cuantizada INT8 completa a veces falla en versiones
    antiguas de TFLite; se usa cuantización dinámica por defecto y se
    intenta INT8 completa solo si X_train_sample está disponible.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if X_train_sample is not None:
        try:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback si alguna op no soporta INT8
            ]
            converter.inference_input_type  = tf.float32
            converter.inference_output_type = tf.float32

            def representative_data_gen():
                for i in range(min(200, len(X_train_sample))):
                    yield [X_train_sample[i:i+1].astype(np.float32)]

            converter.representative_dataset = representative_data_gen
            tflite_model = converter.convert()
            quant_type   = "INT8 (con fallback TFLITE_BUILTINS)"
        except Exception as e:
            print(f"  ⚠ INT8 falló ({e}), usando cuantización dinámica.")
            converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
            converter2.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter2.convert()
            quant_type   = "Dinámica de pesos"
    else:
        tflite_model = converter.convert()
        quant_type   = "Dinámica de pesos"

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  TFLite guardado : {output_path}")
    print(f"  Cuantización    : {quant_type}")
    print(f"  Tamaño          : {size_kb:.1f} KB")
    return size_kb

def benchmark_tflite(tflite_path, X_sample, n_runs=50):
    import tensorflow as tf
    import time

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    times = []
    for i in range(n_runs):
        sample = X_sample[i % len(X_sample) : i % len(X_sample) + 1].astype(np.float32)
        t0 = time.perf_counter()
        interpreter.set_tensor(inp_idx, sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms" : float(np.min(times)),
        "max_ms" : float(np.max(times)),
        "p95_ms" : float(np.percentile(times, 95)),
    }

# ─────────────────────────────────────────────
# VISUALIZACIONES LSTM
# ─────────────────────────────────────────────

def plot_training_curves(history, out_dir, prefix="lstm"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs    = range(1, len(history["accuracy"]) + 1)

    for ax, metric, ylabel in [
        (axes[0], ("accuracy",     "val_accuracy"), "Accuracy"),
        (axes[1], ("loss",         "val_loss"),      "Loss (Cross-Entropy)"),
    ]:
        train_key, val_key = metric
        ax.plot(epochs, history[train_key], color="#3C3489",
                label="Train", linewidth=1.8)
        ax.plot(epochs, history[val_key],   color="#1D9E75",
                label="Val",   linewidth=1.8)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — LSTM entrenamiento", fontweight="bold")
        ax.legend()
        if "accuracy" in train_key:
            ax.set_ylim([0, 1.05])
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


def plot_confusion_matrix(y_true, y_pred, classes, out_dir, prefix="lstm"):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, title, fmt in [
        (axes[0], cm,   "LSTM — Confusión absoluta",    "d"),
        (axes[1], norm, "LSTM — Confusión normalizada", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Purples",
                       vmin=0, vmax=None if fmt == "d" else 1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=42, ha="right", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicción", fontsize=10)
        ax.set_ylabel("Real",       fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                val   = f"{data[i,j]:{fmt}}"
                color = "white" if data[i, j] > thresh else "black"
                ax.text(j, i, val, ha="center", va="center",
                        fontsize=8, color=color)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

# ─────────────────────────────────────────────
# COMPARATIVA CNN vs LSTM
# ─────────────────────────────────────────────

def load_cnn_info(models_dir):
    """Intenta cargar los metadatos del CNN para la comparativa."""
    info_path = os.path.join(models_dir, "cnn_model_info.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path) as f:
        return json.load(f)


def load_cnn_predictions(models_dir, splits_dir):
    """
    Carga el modelo CNN y genera predicciones sobre el test set.
    Devuelve None si el modelo no existe (comparativa se omite parcialmente).
    """
    try:
        import tensorflow as tf
        cnn_path = os.path.join(models_dir, "cnn_model.keras")
        if not os.path.exists(cnn_path):
            return None, None
        cnn = tf.keras.models.load_model(cnn_path)

        X_test = np.transpose(np.load(os.path.join(splits_dir, "X_test.npy")), (0, 2, 1))
        y_test = np.load(os.path.join(splits_dir, "y_test.npy"))

        y_proba_cnn = cnn.predict(X_test, verbose=0)
        y_pred_cnn  = np.argmax(y_proba_cnn, axis=1)
        return y_pred_cnn, y_test
    except Exception as e:
        print(f"  ⚠ No se pudo cargar CNN para comparativa: {e}")
        return None, None


def plot_cnn_vs_lstm(y_test,
                     y_pred_cnn, y_pred_lstm,
                     history_lstm, cnn_info,
                     classes, latency_cnn, latency_lstm,
                     out_dir):
    """
    Panel de comparativa completo CNN vs LSTM con 4 subplots:
      1. Accuracy y F1 por modelo (barras)
      2. Precision/Recall/F1 por clase — ambos modelos
      3. Curvas de entrenamiento LSTM (val_accuracy)
      4. Latencia y tamaño del modelo
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    acc_cnn  = accuracy_score(y_test, y_pred_cnn)  if y_pred_cnn  is not None else 0
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    f1_cnn   = f1_score(y_test, y_pred_cnn,  average="macro") if y_pred_cnn is not None else 0
    f1_lstm  = f1_score(y_test, y_pred_lstm, average="macro")

    # ── Subplot 1: métricas globales ─────────
    ax1    = fig.add_subplot(gs[0, 0])
    models = ["CNN 1D", "Bi-LSTM"]
    accs   = [acc_cnn, acc_lstm]
    f1s    = [f1_cnn,  f1_lstm]
    x      = np.arange(2)
    width  = 0.35

    b1 = ax1.bar(x - width/2, accs, width, label="Accuracy",
                 color=["#3C3489", "#1D9E75"], alpha=0.85)
    b2 = ax1.bar(x + width/2, f1s,  width, label="F1 macro",
                 color=["#6B62C8", "#45C49A"], alpha=0.85)
    ax1.bar_label(b1, fmt="%.4f", fontsize=9, padding=3)
    ax1.bar_label(b2, fmt="%.4f", fontsize=9, padding=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylim([0, 1.12])
    ax1.set_ylabel("Valor")
    ax1.set_title("Accuracy y F1 macro — test set", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Winner badge
    winner = "CNN 1D" if acc_cnn >= acc_lstm else "Bi-LSTM"
    ax1.text(0.99, 0.04, f"Mejor: {winner}",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#E1F5EE", edgecolor="#1D9E75", alpha=0.9))

    # ── Subplot 2: F1 por clase ───────────────
    ax2    = fig.add_subplot(gs[0, 1])
    n_cls  = len(classes)
    x_cls  = np.arange(n_cls)
    w      = 0.38

    f1_cnn_per  = [f1_score(y_test == i, y_pred_cnn  == i) if y_pred_cnn  is not None else 0
                   for i in range(n_cls)]
    f1_lstm_per = [f1_score(y_test == i, y_pred_lstm == i) for i in range(n_cls)]

    ax2.bar(x_cls - w/2, f1_cnn_per,  w, color="#3C3489", alpha=0.8, label="CNN 1D")
    ax2.bar(x_cls + w/2, f1_lstm_per, w, color="#1D9E75", alpha=0.8, label="Bi-LSTM")
    ax2.set_xticks(x_cls)
    ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    ax2.set_ylim([0, 1.12])
    ax2.set_ylabel("F1-score")
    ax2.set_title("F1 por clase — CNN vs LSTM", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.axhline(0.9, color="#BA7517", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Subplot 3: curvas val_accuracy ────────
    ax3    = fig.add_subplot(gs[1, 0])
    epochs = range(1, len(history_lstm["val_accuracy"]) + 1)
    ax3.plot(epochs, history_lstm["accuracy"],
             color="#1D9E75", linewidth=1.8, label="LSTM train")
    ax3.plot(epochs, history_lstm["val_accuracy"],
             color="#1D9E75", linewidth=1.8, linestyle="--", label="LSTM val")

    # Si tenemos info del CNN, agregar su mejor val_accuracy como línea
    if cnn_info and "val_accuracy" in cnn_info:
        ax3.axhline(cnn_info["val_accuracy"], color="#3C3489",
                    linewidth=1.5, linestyle=":", alpha=0.8,
                    label=f"CNN val best ({cnn_info['val_accuracy']:.4f})")

    ax3.set_xlabel("Época")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim([0, 1.05])
    ax3.set_title("Curva de entrenamiento LSTM\nvs mejor CNN", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Subplot 4: latencia y tamaño ─────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Tabla comparativa
    rows = [
        ["Métrica",              "CNN 1D",                          "Bi-LSTM"],
        ["Accuracy (test)",      f"{acc_cnn:.4f}",                  f"{acc_lstm:.4f}"],
        ["F1 macro (test)",      f"{f1_cnn:.4f}",                   f"{f1_lstm:.4f}"],
        ["Parámetros",
         f"{cnn_info.get('total_params',0):,}" if cnn_info else "N/A",
         "ver abajo"],
        ["TFLite (KB)",
         f"{cnn_info.get('tflite_kb',0):.0f}" if cnn_info else "N/A",
         "ver abajo"],
        ["Latencia media*",
         f"{latency_cnn['mean_ms']:.1f} ms"  if latency_cnn  else "N/A",
         f"{latency_lstm['mean_ms']:.1f} ms" if latency_lstm else "N/A"],
        ["Latencia P95*",
         f"{latency_cnn['p95_ms']:.1f} ms"   if latency_cnn  else "N/A",
         f"{latency_lstm['p95_ms']:.1f} ms"  if latency_lstm else "N/A"],
        ["Épocas entrenadas",
         str(cnn_info.get("epochs_trained", "N/A")) if cnn_info else "N/A",
         str(len(history_lstm["accuracy"]))],
    ]
    colors_tbl = [["#EEEDFE"] * 3] + [["white"] * 3] * (len(rows) - 1)
    colors_tbl[0] = ["#3C3489"] * 3

    tbl = ax4.table(
        cellText  = rows,
        cellLoc   = "center",
        loc       = "center",
        cellColours = colors_tbl,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Header blanco
    for j in range(3):
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax4.set_title("Tabla comparativa CNN vs LSTM", fontweight="bold", pad=12)
    ax4.text(0.5, -0.02, "* medida en esta máquina; en RPi4 esperar ×3–5",
             ha="center", va="top", transform=ax4.transAxes,
             fontsize=8, color="gray", fontstyle="italic")

    fig.suptitle("Comparativa CNN 1D vs Bidirectional LSTM\nPanel de domótica por voz",
                 fontsize=14, fontweight="bold")

    path = os.path.join(out_dir, "lstm_vs_cnn_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

# ─────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────

def save_report(model, history, y_val, y_val_pred,
                y_test, y_test_pred,
                y_pred_cnn, classes,
                tflite_size_kb, latency_lstm, latency_cnn,
                cnn_info, out_dir):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE — ENTRENAMIENTO LSTM (MÓDULO 7)")
    log(sep)

    # ── Arquitectura ──────────────────────────
    log()
    log("ARQUITECTURA LSTM")
    log(sep2)
    model.summary(print_fn=lambda s: log("  " + s))

    # ── Hiperparámetros ───────────────────────
    log()
    log("HIPERPARÁMETROS")
    log(sep2)
    for k, v in TRAIN_CFG.items():
        log(f"  {k:<22}: {v}")

    # ── Entrenamiento ─────────────────────────
    n_epochs = len(history["accuracy"])
    best_val = max(history["val_accuracy"])
    best_ep  = history["val_accuracy"].index(best_val) + 1
    log()
    log("ENTRENAMIENTO")
    log(sep2)
    log(f"  Épocas ejecutadas  : {n_epochs}")
    log(f"  Mejor val_accuracy : {best_val:.4f}  (época {best_ep})")
    log(f"  Loss final (train) : {history['loss'][-1]:.4f}")
    log(f"  Loss final (val)   : {history['val_loss'][-1]:.4f}")

    # ── Métricas LSTM ─────────────────────────
    acc_lstm = accuracy_score(y_test, y_test_pred)
    f1_lstm  = f1_score(y_test, y_test_pred, average="macro")
    log()
    log("MÉTRICAS LSTM — TEST SET")
    log(sep2)
    log(f"  Accuracy  : {acc_lstm:.4f}")
    log(f"  F1 macro  : {f1_lstm:.4f}")
    log(f"  Precision : {precision_score(y_test, y_test_pred, average='macro'):.4f}")
    log(f"  Recall    : {recall_score(y_test, y_test_pred, average='macro'):.4f}")
    log()
    log(classification_report(y_test, y_test_pred,
                               target_names=classes, digits=4))

    # ── Comparativa CNN vs LSTM ───────────────
    log()
    log("COMPARATIVA CNN 1D vs Bi-LSTM")
    log(sep2)

    if y_pred_cnn is not None:
        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        f1_cnn  = f1_score(y_test, y_pred_cnn, average="macro")
    else:
        acc_cnn = cnn_info.get("test_accuracy", 0) if cnn_info else 0
        f1_cnn  = 0

    log(f"  {'Métrica':<22} {'CNN 1D':>10} {'Bi-LSTM':>10} {'Delta':>10}")
    log(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    log(f"  {'Accuracy (test)':<22} {acc_cnn:>10.4f} {acc_lstm:>10.4f} "
        f"{acc_lstm - acc_cnn:>+10.4f}")
    log(f"  {'F1 macro (test)':<22} {f1_cnn:>10.4f} {f1_lstm:>10.4f} "
        f"{f1_lstm - f1_cnn:>+10.4f}")

    if cnn_info:
        cnn_params = cnn_info.get("total_params", 0)
        lstm_params = model.count_params()
        log(f"  {'Parámetros':<22} {cnn_params:>10,} {lstm_params:>10,} "
            f"{'(+' if lstm_params > cnn_params else '('}"
            f"{abs(lstm_params - cnn_params):,})")
        log(f"  {'TFLite (KB)':<22} {cnn_info.get('tflite_kb', 0):>10.1f} "
            f"{tflite_size_kb:>10.1f}")

    if latency_cnn and latency_lstm:
        log(f"  {'Latencia media*':<22} {latency_cnn['mean_ms']:>9.1f}ms "
            f"{latency_lstm['mean_ms']:>9.1f}ms")
        log(f"  {'Latencia P95*':<22} {latency_cnn['p95_ms']:>9.1f}ms "
            f"{latency_lstm['p95_ms']:>9.1f}ms")
        log("  * en esta máquina; en RPi4 esperar ×3–5")

    # ── Recomendación ─────────────────────────
    log()
    log("RECOMENDACIÓN PARA EL PIPELINE FINAL")
    log(sep2)
    if y_pred_cnn is not None:
        if acc_cnn >= acc_lstm and (latency_cnn and latency_cnn["mean_ms"] <= latency_lstm.get("mean_ms", 999)):
            log("  → Usar CNN 1D en el pipeline final:")
            log("    mayor (o igual) accuracy Y menor latencia.")
            log("    LSTM disponible como alternativa o para comandos compuestos.")
        elif acc_lstm > acc_cnn:
            log("  → Bi-LSTM supera al CNN en accuracy.")
            log("    Evaluar si la diferencia de latencia es aceptable en RPi4.")
        else:
            log("  → CNN y LSTM tienen accuracy similar.")
            log("    Preferir CNN por menor latencia en RPi4.")
    else:
        log("  → No se pudo cargar CNN para comparativa automática.")
        log("    Revisar métricas manualmente en eval_full_report.txt (módulo 6).")

    log()
    log("PRÓXIMO PASO")
    log(sep2)
    log("  Ejecutar gpio_controller.py (módulo 8).")
    log()
    log(sep)

    report_path = os.path.join(out_dir, "lstm_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 7 — Entrenamiento LSTM")
    parser.add_argument("--splits",       default="splits",
                        help="Carpeta con los .npy (default: splits)")
    parser.add_argument("--output",       default="models",
                        help="Carpeta de salida (default: models)")
    parser.add_argument("--epochs",       type=int, default=TRAIN_CFG["epochs"])
    parser.add_argument("--batch",        type=int, default=TRAIN_CFG["batch_size"])
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--no-tflite",    action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    TRAIN_CFG["epochs"]     = args.epochs
    TRAIN_CFG["batch_size"] = args.batch

    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__} detectado.")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  GPUs: {len(gpus)}  "
              f"{'(acelerado)' if gpus else '(CPU — LSTM en CPU es más lento que CNN)'}")
    except ImportError:
        print("✖ TensorFlow no instalado. pip install tensorflow")
        return

    set_seeds(TRAIN_CFG["random_seed"])

    os.makedirs(args.output, exist_ok=True)
    reports_dir = os.path.join(args.output, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ── 1. Cargar datos ───────────────────────
    print(f"\nCargando splits desde: {os.path.abspath(args.splits)}")
    data, le = load_splits(args.splits)
    classes  = le.classes_
    n_classes = len(classes)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    input_shape = X_train.shape[1:]   # (125, 39)
    print(f"  Input shape : {input_shape}  ← (T=125 frames, C=39 features/frame)")
    print(f"  Clases      : {list(classes)}")
    print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

    # ── 2. Construir modelo ───────────────────
    print("\nConstruyendo Bi-LSTM...")
    model = build_lstm(input_shape, n_classes, TRAIN_CFG)
    model.summary()
    print(f"\nParámetros totales: {model.count_params():,}")

    # ── 3. Compilar ───────────────────────────
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=TRAIN_CFG["learning_rate"]),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    # ── 4. Entrenar ───────────────────────────
    print(f"\nEntrenando — máx. {TRAIN_CFG['epochs']} épocas, "
          f"batch {TRAIN_CFG['batch_size']}...")
    print("  (LSTM tarda ~2× más que CNN en CPU — considera Google Colab)\n")

    history_obj = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = TRAIN_CFG["epochs"],
        batch_size      = TRAIN_CFG["batch_size"],
        callbacks       = build_callbacks(args.output, TRAIN_CFG),
        verbose         = 1,
    )
    history = history_obj.history

    # ── 5. Predicciones ───────────────────────
    print("\nEvaluando...")
    y_val_pred  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(f"  Val  accuracy : {accuracy_score(y_val,  y_val_pred):.4f}")
    print(f"  Test accuracy : {accuracy_score(y_test, y_test_pred):.4f}")

    # ── 6. Guardar modelo ─────────────────────
    keras_path = os.path.join(args.output, "lstm_model.keras")
    model.save(keras_path)
    print(f"\nModelo Keras guardado: {keras_path}")

    # ── 7. TFLite ─────────────────────────────
    tflite_size_kb = 0
    latency_lstm   = None

    if not args.no_tflite:
        print("\nExportando a TFLite...")
        tflite_path    = os.path.join(args.output, "lstm_model.tflite")
        tflite_size_kb = export_tflite(model, tflite_path, X_train_sample=X_train)

        if not args.no_benchmark:
            print(f"  Benchmark ({50} inferencias)...")
            latency_lstm = benchmark_tflite(tflite_path, X_test)
            print(f"  Media: {latency_lstm['mean_ms']:.2f} ms  "
                  f"|  P95: {latency_lstm['p95_ms']:.2f} ms")

    # ── 8. Cargar CNN para comparativa ────────
    print("\nCargando CNN para comparativa...")
    cnn_info = load_cnn_info(args.output)

    y_pred_cnn, _ = load_cnn_predictions(args.output, args.splits)

    latency_cnn = None
    if cnn_info:
        latency_cnn = cnn_info.get("latency_ms")

    # Benchmark CNN TFLite si existe y no tenemos latencia
    if latency_cnn is None and not args.no_benchmark:
        cnn_tflite = os.path.join(args.output, "cnn_model.tflite")
        if os.path.exists(cnn_tflite):
            print("  Benchmark CNN TFLite para comparativa...")
            latency_cnn = benchmark_tflite(cnn_tflite, X_test)

    # ── 9. Guardar metadatos LSTM ─────────────
    lstm_info = {
        "model_type"    : "BiLSTM",
        "input_shape"   : list(input_shape),
        "n_classes"     : int(n_classes),
        "classes"       : list(classes),
        "total_params"  : int(model.count_params()),
        "val_accuracy"  : float(accuracy_score(y_val,  y_val_pred)),
        "test_accuracy" : float(accuracy_score(y_test, y_test_pred)),
        "epochs_trained": len(history["accuracy"]),
        "train_cfg"     : TRAIN_CFG,
        "tflite_kb"     : float(tflite_size_kb),
        "latency_ms"    : latency_lstm,
    }
    info_path = os.path.join(args.output, "lstm_model_info.json")
    with open(info_path, "w") as f:
        json.dump(lstm_info, f, indent=2)

    # ── 10. Reporte + gráficas ────────────────
    save_report(model, history,
                y_val, y_val_pred,
                y_test, y_test_pred,
                y_pred_cnn, classes,
                tflite_size_kb, latency_lstm, latency_cnn,
                cnn_info, reports_dir)

    if not args.no_plots:
        print("\nGenerando gráficas...")
        plot_training_curves(history,    reports_dir)
        plot_confusion_matrix(y_test, y_test_pred, classes, reports_dir)
        plot_cnn_vs_lstm(
            y_test,
            y_pred_cnn  if y_pred_cnn  is not None else np.zeros_like(y_test),
            y_test_pred,
            history, cnn_info,
            classes, latency_cnn, latency_lstm,
            reports_dir,
        )

    print("\n✔ Módulo 7 completado.")
    print(f"  LSTM test accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print("  Siguiente paso: gpio_controller.py (módulo 8)\n")


if __name__ == "__main__":
    main()