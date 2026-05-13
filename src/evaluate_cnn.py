"""
MÓDULO 6 — Evaluación completa del modelo CNN 1D
Panel de domótica por voz — Raspberry Pi 4

Carga cnn_model.keras y el test set, genera todas las métricas
y gráficas necesarias para el documento del proyecto.

Uso:
    python evaluate_cnn.py
    python evaluate_cnn.py --model models/cnn_model.keras --splits splits

Salida (en models/reports/):
    eval_confusion_matrix.png
    eval_per_class_metrics.png
    eval_roc_curves.png
    eval_confidence_distribution.png
    eval_error_analysis.png
    eval_full_report.txt
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# CARGA DE DATOS Y MODELO
# ─────────────────────────────────────────────

def load_model_and_data(model_path, splits_dir):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("pip install tensorflow")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            "Ejecuta train_cnn.py (módulo 5) primero."
        )

    print(f"  Cargando modelo  : {model_path}")
    model = tf.keras.models.load_model(model_path)

    def load_npy(name):
        p = os.path.join(splits_dir, f"{name}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"No encontrado: {p}")
        return np.load(p)

    X_test = load_npy("X_test")
    y_test = load_npy("y_test")
    X_val  = load_npy("X_val")
    y_val  = load_npy("y_val")

    # (N, C, T) → (N, T, C) — igual que módulo 5
    X_test = np.transpose(X_test, (0, 2, 1))
    X_val  = np.transpose(X_val,  (0, 2, 1))

    le_path = os.path.join(splits_dir, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return model, X_test, y_test, X_val, y_val, le


# ─────────────────────────────────────────────
# PREDICCIONES
# ─────────────────────────────────────────────

def get_predictions(model, X):
    y_proba    = model.predict(X, verbose=0)
    y_pred     = np.argmax(y_proba, axis=1)
    confidence = np.max(y_proba, axis=1)
    return y_pred, y_proba, confidence


# ─────────────────────────────────────────────
# VISUALIZACIÓN 1 — Matriz de confusión
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes, out_dir, prefix="eval"):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, data, title, fmt in [
        (axes[0], cm,   "Matriz de confusión — valores absolutos",    "d"),
        (axes[1], norm, "Matriz de confusión — normalizada por fila", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
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
# VISUALIZACIÓN 2 — Métricas por clase
# ─────────────────────────────────────────────

def plot_per_class_metrics(y_true, y_pred, classes, out_dir, prefix="eval"):
    report  = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )
    metrics = ["precision", "recall", "f1-score"]
    x       = np.arange(len(classes))
    width   = 0.25
    colors  = ["#3C3489", "#1D9E75", "#E24B4A"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [report[cls][metric] for cls in classes]
        bars   = ax.bar(x + (i - 1) * width, values, width,
                        label=metric.capitalize(), color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Valor (0–1)")
    ax.set_ylim([0, 1.12])
    ax.set_title("Precision / Recall / F1 por clase (test set)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(0.9, color="#BA7517", linestyle="--",
               linewidth=0.8, alpha=0.6)

    acc = accuracy_score(y_true, y_pred)
    ax.text(0.99, 0.97, f"Accuracy global: {acc:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEDFE",
                      edgecolor="#3C3489", alpha=0.8))
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_per_class_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# VISUALIZACIÓN 3 — Curvas ROC
# ─────────────────────────────────────────────

def plot_roc_curves(y_true, y_proba, classes, out_dir, prefix="eval"):
    """
    Curva ROC one-vs-rest para cada clase.
    AUC cercano a 1.0 indica buena separabilidad de esa clase.
    """
    n_classes = len(classes)
    y_bin     = label_binarize(y_true, classes=range(n_classes))
    cols      = 4
    rows      = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes_flat = axes.flatten() if n_classes > 1 else [axes]
    colors    = [
        "#3C3489", "#1D9E75", "#E24B4A", "#BA7517",
        "#0C447C", "#72243E", "#27500A", "#444441",
    ]

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)

        ax = axes_flat[i]
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=colors[i % len(colors)])
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n_classes, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Curvas ROC por clase (one-vs-rest) — test set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# VISUALIZACIÓN 4 — Distribución de confianza
# ─────────────────────────────────────────────

def plot_confidence_distribution(y_true, y_pred, confidence,
                                 classes, out_dir, prefix="eval"):
    """
    Histograma de confianza (prob. máxima del softmax).
    Un buen modelo tiene alta confianza en aciertos y baja en errores.
    También calcula el umbral de rechazo óptimo para producción.
    """
    correct   = y_true == y_pred
    incorrect = ~correct

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Histograma global ────────────────────
    ax = axes[0]
    ax.hist(confidence[correct],   bins=20, alpha=0.75,
            color="#1D9E75", label=f"Correctas  ({correct.sum()})")
    ax.hist(confidence[incorrect], bins=20, alpha=0.75,
            color="#E24B4A", label=f"Incorrectas ({incorrect.sum()})")
    ax.set_xlabel("Confianza (prob. máxima del softmax)", fontsize=10)
    ax.set_ylabel("Número de muestras",                  fontsize=10)
    ax.set_title("Distribución de confianza\ncorrectas vs incorrectas",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Confianza media por clase ─────────────
    ax2 = axes[1]
    mean_c, mean_i = [], []
    for i in range(len(classes)):
        mask  = y_true == i
        right = confidence[mask &  correct]
        wrong = confidence[mask & incorrect]
        mean_c.append(right.mean() if len(right) > 0 else 0)
        mean_i.append(wrong.mean() if len(wrong) > 0 else 0)

    x     = np.arange(len(classes))
    width = 0.38
    ax2.bar(x - width/2, mean_c, width, color="#1D9E75", alpha=0.85, label="Correctas")
    ax2.bar(x + width/2, mean_i, width, color="#E24B4A", alpha=0.85, label="Incorrectas")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Confianza promedio")
    ax2.set_ylim([0, 1.1])
    ax2.set_title("Confianza media por clase", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_confidence_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# VISUALIZACIÓN 5 — Análisis de errores
# ─────────────────────────────────────────────

def plot_error_analysis(y_true, y_pred, classes, out_dir, prefix="eval"):
    """
    Mapa de calor de confusiones (sin la diagonal) y ranking
    de los pares de clases más confundidos.
    Sirve para decidir qué comandos re-grabar o aumentar más.
    """
    n      = len(classes)
    cm     = confusion_matrix(y_true, y_pred)
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Heatmap de errores ───────────────────
    ax = axes[0]
    im = ax.imshow(errors, cmap="Reds", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=42, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicho como →", fontsize=10)
    ax.set_ylabel("Real →",          fontsize=10)
    ax.set_title("Mapa de errores\n(diagonal eliminada)",
                 fontweight="bold", fontsize=11)
    for i in range(n):
        for j in range(n):
            if errors[i, j] > 0:
                color = "white" if errors[i, j] > errors.max() / 2 else "black"
                ax.text(j, i, str(errors[i, j]),
                        ha="center", va="center", fontsize=8, color=color)

    # ── Top pares confundidos ─────────────────
    ax2   = axes[1]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and errors[i, j] > 0:
                pairs.append((errors[i, j], f"{classes[i]}  →  {classes[j]}"))
    pairs.sort(reverse=True)
    top = pairs[:min(10, len(pairs))]

    if top:
        vals   = [p[0] for p in top]
        labels = [p[1] for p in top]
        c_list = ["#E24B4A" if v == max(vals) else "#FAECE7" for v in vals]
        bars   = ax2.barh(range(len(vals)), vals, color=c_list, edgecolor="none")
        ax2.set_yticks(range(len(vals)))
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel("Número de confusiones")
        ax2.set_title("Pares con más confusiones\n(candidatos a re-grabar o aumentar)",
                      fontweight="bold", fontsize=11)
        ax2.bar_label(bars, fontsize=9, padding=3)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "¡Sin errores en el test set!",
                 ha="center", va="center", fontsize=14,
                 transform=ax2.transAxes, color="#1D9E75")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_error_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE COMPLETO
# ─────────────────────────────────────────────

def save_full_report(y_val, y_val_pred, conf_val,
                     y_test, y_test_pred, conf_test,
                     y_proba_test, classes,
                     model_info, out_dir, prefix="eval"):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE COMPLETO — EVALUACIÓN CNN 1D (MÓDULO 6)")
    log(sep)

    # ── Info del modelo ───────────────────────
    if model_info:
        log()
        log("INFORMACIÓN DEL MODELO")
        log(sep2)
        log(f"  Tipo             : {model_info.get('model_type', 'CNN1D')}")
        log(f"  Input shape      : {model_info.get('input_shape')}")
        tp = model_info.get('total_params')
        log(f"  Parámetros       : {tp:,}" if tp else "  Parámetros       : N/A")
        log(f"  Épocas entrenado : {model_info.get('epochs_trained', 'N/A')}")
        log(f"  TFLite size      : {model_info.get('tflite_kb', 0):.1f} KB")

    # ── Métricas en validación ────────────────
    log()
    log("MÉTRICAS EN VALIDACIÓN")
    log(sep2)
    log(f"  Accuracy  : {accuracy_score(y_val, y_val_pred):.4f}")
    log(f"  F1 macro  : {f1_score(y_val, y_val_pred, average='macro'):.4f}")
    log(f"  Precision : {precision_score(y_val, y_val_pred, average='macro'):.4f}")
    log(f"  Recall    : {recall_score(y_val, y_val_pred, average='macro'):.4f}")
    n_correct_val = (y_val == y_val_pred).sum()
    if n_correct_val > 0:
        log(f"  Confianza promedio (correctas) : {conf_val[y_val == y_val_pred].mean():.4f}")
    log()
    log(classification_report(y_val, y_val_pred, target_names=classes, digits=4))

    # ── Métricas en test ──────────────────────
    log()
    log("MÉTRICAS EN TEST (set reservado — usar solo al final)")
    log(sep2)
    log(f"  Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
    log(f"  F1 macro  : {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    log(f"  Precision : {precision_score(y_test, y_test_pred, average='macro'):.4f}")
    log(f"  Recall    : {recall_score(y_test, y_test_pred, average='macro'):.4f}")
    n_correct  = (y_test == y_test_pred).sum()
    n_wrong    = (y_test != y_test_pred).sum()
    if n_correct > 0:
        log(f"  Confianza promedio (correctas)  : {conf_test[y_test == y_test_pred].mean():.4f}")
    if n_wrong > 0:
        log(f"  Confianza promedio (incorrectas): {conf_test[y_test != y_test_pred].mean():.4f}")
    else:
        log("  (sin errores en test)")
    log()
    log(classification_report(y_test, y_test_pred, target_names=classes, digits=4))

    # ── AUC por clase ─────────────────────────
    log()
    log("AUC ROC POR CLASE (test set, one-vs-rest)")
    log(sep2)
    n_classes = len(classes)
    y_bin     = label_binarize(y_test, classes=range(n_classes))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_test[:, i])
        roc_auc     = auc(fpr, tpr)
        bar         = "█" * int(roc_auc * 20)
        log(f"  {cls:<16} AUC = {roc_auc:.4f}  {bar}")

    # ── Análisis de errores ───────────────────
    log()
    log("ANÁLISIS DE ERRORES")
    log(sep2)
    cm_err = confusion_matrix(y_test, y_test_pred)
    errors = cm_err.copy()
    np.fill_diagonal(errors, 0)
    total_err = int(errors.sum())
    log(f"  Total errores en test : {total_err}")

    if total_err > 0:
        pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and errors[i, j] > 0:
                    pairs.append((int(errors[i, j]), classes[i], classes[j]))
        pairs.sort(reverse=True)
        log("  Top confusiones:")
        for cnt, real, pred in pairs[:5]:
            log(f"    '{real}' predicho como '{pred}' : {cnt} veces")
        log()
        log("  Recomendación: re-grabar más muestras o aplicar")
        log("  augmentation selectivo en estos pares de clases.")
    else:
        log("  ✔ Cero errores en el test set.")

    # ── Umbral de confianza ───────────────────
    log()
    log("UMBRAL DE CONFIANZA PARA PRODUCCIÓN")
    log(sep2)
    log("  (Predicciones con confianza < umbral se descartan en GPIO)")
    log()
    log(f"  {'Umbral':>7}  {'Aceptadas':>10}  {'% total':>7}  {'Acc aceptadas':>14}")
    log(f"  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*14}")
    for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
        mask     = conf_test >= thr
        accepted = mask.sum()
        if accepted == 0:
            continue
        acc_thr = accuracy_score(y_test[mask], y_test_pred[mask])
        pct     = 100 * accepted / len(y_test)
        log(f"  {thr:>7.2f}  {accepted:>10}  {pct:>6.1f}%  {acc_thr:>14.4f}")
    log()
    log("  → Umbral recomendado: 0.70–0.80")
    log("    Implementar en realtime_pipeline.py (módulo 9)")

    # ── Cierre ────────────────────────────────
    log()
    log("PRÓXIMO PASO")
    log(sep2)
    log("  Ejecutar train_lstm.py (módulo 7).")
    log()
    log(sep)

    report_path = os.path.join(out_dir, f"{prefix}_full_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 6 — Evaluación CNN 1D")
    parser.add_argument("--model",    default="models/cnn_model.keras",
                        help="Ruta al modelo .keras (default: models/cnn_model.keras)")
    parser.add_argument("--splits",   default="splits",
                        help="Carpeta con los .npy (default: splits)")
    parser.add_argument("--output",   default="models/reports",
                        help="Carpeta de salida (default: models/reports)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. Cargar modelo y datos ──────────────
    print(f"\nCargando modelo y datos...")
    model, X_test, y_test, X_val, y_val, le = load_model_and_data(
        args.model, args.splits
    )
    classes = le.classes_
    print(f"  Test  : {X_test.shape}  |  Val : {X_val.shape}")
    print(f"  Clases: {list(classes)}")

    # ── 2. Cargar info del modelo ─────────────
    model_info = {}
    info_path  = os.path.join(os.path.dirname(args.model), "cnn_model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)

    # ── 3. Predicciones ───────────────────────
    print("\nGenerando predicciones...")
    y_val_pred,  y_proba_val,  conf_val  = get_predictions(model, X_val)
    y_test_pred, y_proba_test, conf_test = get_predictions(model, X_test)

    val_acc  = accuracy_score(y_val,  y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1_test  = f1_score(y_test, y_test_pred, average="macro")
    print(f"  Val  accuracy : {val_acc:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Test F1 macro : {f1_test:.4f}")

    # ── 4. Reporte ────────────────────────────
    save_full_report(
        y_val, y_val_pred, conf_val,
        y_test, y_test_pred, conf_test,
        y_proba_test, classes, model_info,
        args.output,
    )

    # ── 5. Gráficas ───────────────────────────
    if not args.no_plots:
        print("\nGenerando gráficas...")
        plot_confusion_matrix(y_test, y_test_pred, classes, args.output)
        plot_per_class_metrics(y_test, y_test_pred, classes, args.output)
        plot_roc_curves(y_test, y_proba_test, classes, args.output)
        plot_confidence_distribution(
            y_test, y_test_pred, conf_test, classes, args.output
        )
        plot_error_analysis(y_test, y_test_pred, classes, args.output)

    print("\n✔ Módulo 6 completado.")
    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Test F1 macro : {f1_test:.4f}")
    print("  Siguiente paso: train_lstm.py (módulo 7)\n")


if __name__ == "__main__":
    main()