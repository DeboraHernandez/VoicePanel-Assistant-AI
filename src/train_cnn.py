"""
MÓDULO 5 — Entrenamiento CNN 1D
Panel de domótica por voz — Raspberry Pi 4

Diseña, entrena y exporta una CNN 1D sobre features MFCC
(39 canales × 125 frames) para clasificar 8 comandos de voz.

Arquitectura:
    Input (39, 125)
    → Conv1D(32,  k=3) + BN + ReLU + MaxPool(2)
    → Conv1D(64,  k=3) + BN + ReLU + MaxPool(2)
    → Conv1D(128, k=3) + BN + ReLU + GlobalAvgPool
    → Dropout(0.4)
    → Dense(128) + ReLU
    → Dropout(0.3)
    → Dense(8)   + Softmax

Uso:
    python train_cnn.py
    python train_cnn.py --splits splits --output models --epochs 60

Salida (en models/):
    cnn_model.keras          ← modelo completo para continuar entrenando
    cnn_model.tflite         ← modelo cuantizado para inferencia en RPi4
    models/reports/
        cnn_training_curves.png
        cnn_confusion_matrix.png
        cnn_metrics_report.txt
        cnn_architecture.txt
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silenciar logs de TF

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

TRAIN_CFG = {
    "epochs"        : 60,
    "batch_size"    : 32,
    "learning_rate" : 1e-3,
    "dropout_conv"  : 0.4,
    "dropout_dense" : 0.3,
    "dense_units"   : 128,
    "random_seed"   : 42,

    # Early stopping
    "es_patience"   : 12,
    "es_monitor"    : "val_accuracy",

    # Reduce LR on plateau
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
    Carga los 6 arrays .npy y el label_encoder desde splits_dir.
    Añade la dimensión de canal requerida por Conv1D: (N, T, C).
    """
    files = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    data  = {}
    for name in files:
        path = os.path.join(splits_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró {path}\n"
                "Ejecuta prepare_splits.py (módulo 4) primero."
            )
        data[name] = np.load(path)

    # Cargar label encoder
    le_path = os.path.join(splits_dir, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # X shape actual: (N, C, T) → Conv1D espera (N, T, C)
    for key in ["X_train", "X_val", "X_test"]:
        data[key] = np.transpose(data[key], (0, 2, 1))   # (N, 125, 39)

    return data, le


# ─────────────────────────────────────────────
# ARQUITECTURA CNN 1D
# ─────────────────────────────────────────────

def build_cnn(input_shape, n_classes, cfg):
    """
    Construye una CNN 1D con BatchNormalization, GlobalAveragePooling
    y Dropout. Justificación de cada decisión:

    • Conv1D sobre la dimensión temporal (125 frames):
      aprende patrones locales en el tiempo sin asumir periodicidad.
    • BatchNormalization después de cada Conv:
      estabiliza el entrenamiento y actúa como regularizador suave.
    • GlobalAveragePooling en lugar de Flatten:
      reduce drásticamente parámetros y ayuda a generalizar.
      Para una entrada (N, 125, 39) → Flatten daría 16,250 features;
      GAP da solo 128.
    • Dropout(0.4) antes del Dense: previene co-adaptación.
    • Softmax final: probabilidad para cada una de las 8 clases.

    input_shape: (T, C) = (125, 39)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    tf.random.set_seed(cfg["random_seed"])

    inp = layers.Input(shape=input_shape, name="mfcc_input")

    # ── Bloque 1 ─────────────────────────────
    x = layers.Conv1D(32, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── Bloque 2 ─────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── Bloque 3 ─────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Cabeza clasificadora ──────────────────
    x = layers.Dropout(cfg["dropout_conv"],  name="drop1")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense1")(x)
    x = layers.Dropout(cfg["dropout_dense"], name="drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CNN1D_VoiceCommand")
    return model


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

def build_callbacks(models_dir, cfg):
    from tensorflow.keras import callbacks

    cb_list = [
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
            filepath          = os.path.join(models_dir, "cnn_best.keras"),
            monitor           = "val_accuracy",
            save_best_only    = True,
            verbose           = 0,
        ),
    ]
    return cb_list


# ─────────────────────────────────────────────
# EXPORTAR A TFLITE
# ─────────────────────────────────────────────

def export_tflite(model, output_path, X_train_sample=None):
    """
    Exporta el modelo Keras a TFLite con cuantización dinámica de pesos.
    La cuantización reduce el tamaño ~4× y acelera la inferencia en ARM
    sin pérdida significativa de accuracy (generalmente <1%).

    Si se provee X_train_sample se aplica cuantización completa INT8
    (aún más rápida en RPi4 con NEON SIMD).
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if X_train_sample is not None:
        # Cuantización INT8 completa — requiere datos de calibración
        converter.optimizations      = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.float32   # entrada sigue siendo float
        converter.inference_output_type = tf.float32

        def representative_data_gen():
            for i in range(min(200, len(X_train_sample))):
                sample = X_train_sample[i:i+1].astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_data_gen
        quant_type = "INT8"
    else:
        # Cuantización dinámica de pesos (más simple, sin datos de calibración)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_type = "dynamic weights"

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  TFLite guardado: {output_path}")
    print(f"  Cuantización   : {quant_type}")
    print(f"  Tamaño         : {size_kb:.1f} KB")

    return size_kb


def benchmark_tflite(tflite_path, X_sample, n_runs=50):
    """
    Mide la latencia de inferencia del modelo TFLite en el hardware actual.
    En RPi4 los tiempos serán 3–5× más lentos que en una laptop x86.
    """
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
        times.append((time.perf_counter() - t0) * 1000)   # ms

    times = np.array(times)
    return {
        "mean_ms" : float(np.mean(times)),
        "min_ms"  : float(np.min(times)),
        "max_ms"  : float(np.max(times)),
        "p95_ms"  : float(np.percentile(times, 95)),
    }


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_training_curves(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["accuracy"]) + 1)

    # ── Accuracy ─────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["accuracy"],     color="#3C3489", label="Train", linewidth=1.8)
    ax.plot(epochs, history["val_accuracy"], color="#1D9E75", label="Val",   linewidth=1.8)
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy durante el entrenamiento", fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.spines[["top", "right"]].set_visible(False)

    # ── Loss ─────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["loss"],     color="#3C3489", label="Train", linewidth=1.8)
    ax.plot(epochs, history["val_loss"], color="#E24B4A", label="Val",   linewidth=1.8)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Loss durante el entrenamiento", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "cnn_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


def plot_confusion_matrix(y_true, y_pred, classes, out_dir):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalizada por fila

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,   "Matriz de confusión (absoluta)",   "d"),
        (axes[1], norm, "Matriz de confusión (normalizada)", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(title, fontweight="bold", fontsize=11)

        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = f"{data[i,j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(out_dir, "cnn_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────

def save_metrics_report(model, history, y_val, y_val_pred,
                        y_test, y_test_pred, classes,
                        tflite_size_kb, latency, out_dir):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE — ENTRENAMIENTO CNN 1D (MÓDULO 5)")
    log(sep)

    # ── Arquitectura ──────────────────────────
    log()
    log("ARQUITECTURA")
    log(sep2)
    model.summary(print_fn=lambda s: log("  " + s))

    # ── Hiperparámetros ───────────────────────
    log()
    log("HIPERPARÁMETROS")
    log(sep2)
    for k, v in TRAIN_CFG.items():
        log(f"  {k:<22}: {v}")

    # ── Épocas ────────────────────────────────
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

    # ── Métricas en val ───────────────────────
    log()
    log("MÉTRICAS EN VALIDACIÓN")
    log(sep2)
    log(f"  Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
    log(f"  F1 macro : {f1_score(y_val, y_val_pred, average='macro'):.4f}")
    log()
    log(classification_report(y_val, y_val_pred,
                               target_names=classes, digits=4))

    # ── Métricas en test ──────────────────────
    log()
    log("MÉTRICAS EN TEST (set reservado)")
    log(sep2)
    log(f"  Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    log(f"  F1 macro : {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    log()
    log(classification_report(y_test, y_test_pred,
                               target_names=classes, digits=4))

    # ── TFLite ────────────────────────────────
    log()
    log("MODELO TFLITE")
    log(sep2)
    log(f"  Tamaño             : {tflite_size_kb:.1f} KB")
    if latency:
        log(f"  Latencia media     : {latency['mean_ms']:.2f} ms")
        log(f"  Latencia P95       : {latency['p95_ms']:.2f} ms")
        log(f"  (medido en {50} inferencias en esta máquina)")
        log(f"  ⚠ En RPi4 esperar ~3-5× más lento (~{latency['mean_ms']*4:.0f} ms)")

    # ── Siguiente paso ────────────────────────
    log()
    log("PRÓXIMO PASO")
    log(sep2)
    log("  Ejecutar evaluate_cnn.py (módulo 6) para análisis detallado.")
    log("  Luego train_lstm.py (módulo 7).")
    log()
    log(sep)

    report_path = os.path.join(out_dir, "cnn_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 5 — Entrenamiento CNN 1D")
    parser.add_argument("--splits",  default="splits",
                        help="Carpeta con los arrays .npy (default: splits)")
    parser.add_argument("--output",  default="models",
                        help="Carpeta de salida para modelos (default: models)")
    parser.add_argument("--epochs",  type=int, default=TRAIN_CFG["epochs"],
                        help=f"Épocas máximas (default: {TRAIN_CFG['epochs']})")
    parser.add_argument("--batch",   type=int, default=TRAIN_CFG["batch_size"],
                        help=f"Batch size (default: {TRAIN_CFG['batch_size']})")
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--no-tflite",   action="store_true",
                        help="Omitir exportación TFLite")
    parser.add_argument("--no-benchmark",action="store_true",
                        help="Omitir benchmark de latencia TFLite")
    args = parser.parse_args()

    TRAIN_CFG["epochs"]     = args.epochs
    TRAIN_CFG["batch_size"] = args.batch

    # ── Verificar TF disponible ───────────────
    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__} detectado.")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  GPUs disponibles: {len(gpus)}  {'(entrenamiento acelerado)' if gpus else '(CPU — considera Google Colab si es lento)'}")
    except ImportError:
        print("✖ TensorFlow no está instalado.")
        print("  pip install tensorflow")
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
    print(f"  Input shape : {input_shape}")
    print(f"  Clases      : {list(classes)}")
    print(f"  Train       : {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

    # ── 2. Construir modelo ───────────────────
    print("\nConstruyendo modelo CNN 1D...")
    model = build_cnn(input_shape, n_classes, TRAIN_CFG)
    model.summary()

    total_params = model.count_params()
    print(f"\nParámetros totales: {total_params:,}")

    # ── 3. Compilar ───────────────────────────
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=TRAIN_CFG["learning_rate"]),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    # ── 4. Entrenar ───────────────────────────
    print(f"\nEntrenando — máx. {TRAIN_CFG['epochs']} épocas, "
          f"batch {TRAIN_CFG['batch_size']}...")
    print("  (Early stopping si val_accuracy no mejora en "
          f"{TRAIN_CFG['es_patience']} épocas)\n")

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
    print("\nEvaluando sobre val y test...")
    y_val_pred  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    val_acc  = accuracy_score(y_val,  y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Val  accuracy : {val_acc:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")

    # ── 6. Guardar modelo Keras ───────────────
    keras_path = os.path.join(args.output, "cnn_model.keras")
    model.save(keras_path)
    print(f"\nModelo Keras guardado: {keras_path}")

    # ── 7. Exportar TFLite ────────────────────
    tflite_size_kb = 0
    latency        = None

    if not args.no_tflite:
        print("\nExportando a TFLite...")
        tflite_path = os.path.join(args.output, "cnn_model.tflite")
        tflite_size_kb = export_tflite(model, tflite_path, X_train_sample=X_train)

        if not args.no_benchmark:
            print(f"  Benchmark de latencia ({50} inferencias)...")
            latency = benchmark_tflite(tflite_path, X_test)
            print(f"  Media: {latency['mean_ms']:.2f} ms  |  P95: {latency['p95_ms']:.2f} ms")

    # ── 8. Guardar metadatos del modelo ───────
    model_info = {
        "model_type"     : "CNN1D",
        "input_shape"    : list(input_shape),
        "n_classes"      : int(n_classes),
        "classes"        : list(classes),
        "total_params"   : int(total_params),
        "val_accuracy"   : float(val_acc),
        "test_accuracy"  : float(test_acc),
        "epochs_trained" : len(history["accuracy"]),
        "train_cfg"      : TRAIN_CFG,
        "tflite_kb"      : float(tflite_size_kb),
        "latency_ms"     : latency,
    }
    info_path = os.path.join(args.output, "cnn_model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"  Metadatos guardados: {info_path}")

    # ── 9. Reporte + gráficas ─────────────────
    save_metrics_report(model, history,
                        y_val, y_val_pred,
                        y_test, y_test_pred,
                        classes, tflite_size_kb, latency,
                        reports_dir)

    if not args.no_plots:
        print("\nGenerando gráficas...")
        plot_training_curves(history, reports_dir)
        plot_confusion_matrix(y_test, y_test_pred, classes, reports_dir)

    print("\n✔ Módulo 5 completado.")
    print(f"  Val accuracy  : {val_acc:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")
    print("  Siguiente paso: evaluate_cnn.py (módulo 6)\n")


if __name__ == "__main__":
    main()