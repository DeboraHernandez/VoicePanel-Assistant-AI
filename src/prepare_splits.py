"""
MÓDULO 4 — Split train / val / test
Panel de domótica por voz — Raspberry Pi 4

Carga X_aug.npy / y_aug.npy (salida del módulo 3),
aplica un split estratificado 70 / 15 / 15 y guarda
los seis arrays listos para entrenar los modelos.

Uso:
    python prepare_splits.py
    python prepare_splits.py --features features_aug --output splits
    python prepare_splits.py --ratio 70 15 15

Salida (en splits/):
    X_train.npy   y_train.npy
    X_val.npy     y_val.npy
    X_test.npy    y_test.npy
    split_info.json            ← índices y metadatos reproducibles
    reports/
        split_report.txt
        split_distribution.png
        split_tsne.png         ← proyección 2-D del espacio de features
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DEFAULT_RATIO  = (0.70, 0.15, 0.15)   # train / val / test
RANDOM_STATE   = 42                    # semilla fija → reproducible

COMMANDS = [
    "enciende", "apaga", "ventilador",
    "abrir",    "cerrar", "musica",
    "detente",  "ruido_fondo",
]

# Colores consistentes para gráficas
CLASS_COLORS = [
    "#3C3489", "#1D9E75", "#E24B4A", "#BA7517",
    "#0C447C", "#72243E", "#27500A", "#444441",
]

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def load_features(features_dir):
    """Carga X_aug.npy, y_aug.npy y el label_encoder desde features_dir."""
    x_path  = os.path.join(features_dir, "X_aug.npy")
    y_path  = os.path.join(features_dir, "y_aug.npy")
    le_path = os.path.join(features_dir, "label_encoder.pkl")

    # Soporte para dataset sin augmentation (módulo 2 directo)
    if not os.path.exists(x_path):
        x_path = os.path.join(features_dir, "X.npy")
        y_path = os.path.join(features_dir, "y.npy")
        print("  ℹ  X_aug.npy no encontrado → usando X.npy (sin augmentation)")

    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"No se encontró X_aug.npy ni X.npy en: {features_dir}\n"
            "Ejecuta primero feature_extraction.py (módulo 2) o "
            "augment_dataset.py (módulo 3)."
        )

    X  = np.load(x_path)
    y  = np.load(y_path)

    le = None
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    else:
        # Reconstruir LabelEncoder desde los índices presentes
        le = LabelEncoder()
        le.fit(y)
        print("  ⚠  label_encoder.pkl no encontrado → reconstruido desde y")

    return X, y, le


def parse_ratio(ratio_tuple):
    """Valida y normaliza la tupla (train, val, test) a fracciones que sumen 1."""
    r = np.array(ratio_tuple, dtype=float)
    if r.sum() != 100 and abs(r.sum() - 1.0) > 0.01:
        raise ValueError(
            f"Los ratios {ratio_tuple} no suman 100 ni 1.0. "
            "Ejemplo válido: --ratio 70 15 15"
        )
    if r.sum() > 1.5:          # vienen como porcentajes (70 15 15)
        r = r / 100.0
    if abs(r.sum() - 1.0) > 0.01:
        raise ValueError("Los ratios deben sumar 1.0 (o 100 como porcentajes).")
    return tuple(r)


# ─────────────────────────────────────────────
# SPLIT ESTRATIFICADO
# ─────────────────────────────────────────────

def stratified_split(X, y, ratio, random_state=RANDOM_STATE):
    """
    Divide X, y en train / val / test de forma estratificada.

    Estratificado significa que cada partición mantiene la misma
    proporción de clases que el dataset completo, evitando que
    una clase quede subrepresentada en test por azar.

    Proceso en 2 pasos:
      1. Separar test  del resto  (tamaño = ratio[2])
      2. Separar val   del resto  (tamaño ajustado para que val/total = ratio[1])
    """
    r_train, r_val, r_test = ratio

    # Paso 1: separar test
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size    = r_test,
        random_state = random_state,
        stratify     = y,
    )

    # Paso 2: separar val del resto
    # val_size relativo al resto = r_val / (r_train + r_val)
    val_relative = r_val / (r_train + r_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size    = val_relative,
        random_state = random_state,
        stratify     = y_rest,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# VERIFICACIÓN DE CALIDAD
# ─────────────────────────────────────────────

def verify_splits(y_train, y_val, y_test, le):
    """
    Verifica que:
    - Todas las clases estén presentes en los 3 splits.
    - El desbalance dentro de cada split sea aceptable.
    Devuelve lista de advertencias.
    """
    warnings_list = []
    classes = np.unique(np.concatenate([y_train, y_val, y_test]))

    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        present = np.unique(y_split)
        missing = set(classes) - set(present)
        if missing:
            missing_names = le.inverse_transform(list(missing))
            warnings_list.append(
                f"⚠  Split '{split_name}' no contiene las clases: {list(missing_names)}"
            )

        counts = np.bincount(y_split, minlength=len(classes))
        if counts.min() == 0:
            continue
        imbalance = counts.max() / counts.min()
        if imbalance > 2.5:
            warnings_list.append(
                f"⚠  Split '{split_name}' tiene desbalance de {imbalance:.1f}x "
                f"(mín {counts.min()} muestras en alguna clase)"
            )

    return warnings_list


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_split_distribution(y_train, y_val, y_test, le, out_dir):
    """
    Gráfica de barras agrupadas: conteo por clase en cada split.
    """
    classes = le.classes_
    n_cls   = len(classes)
    x       = np.arange(n_cls)
    width   = 0.28

    def counts(y):
        c = np.bincount(y, minlength=n_cls)
        return c

    c_train = counts(y_train)
    c_val   = counts(y_val)
    c_test  = counts(y_test)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Subplot 1: conteos absolutos ─────────
    ax = axes[0]
    ax.bar(x - width, c_train, width, label="Train", color="#3C3489", alpha=0.85)
    ax.bar(x,         c_val,   width, label="Val",   color="#1D9E75", alpha=0.85)
    ax.bar(x + width, c_test,  width, label="Test",  color="#E24B4A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Número de muestras")
    ax.set_title("Distribución absoluta por split", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # ── Subplot 2: proporción relativa (%) ───
    ax2   = axes[1]
    total = c_train + c_val + c_test
    p_tr  = 100 * c_train / total
    p_val = 100 * c_val   / total
    p_te  = 100 * c_test  / total

    ax2.bar(x, p_tr,  color="#3C3489", alpha=0.85, label="Train")
    ax2.bar(x, p_val, bottom=p_tr,           color="#1D9E75", alpha=0.85, label="Val")
    ax2.bar(x, p_te,  bottom=p_tr + p_val,   color="#E24B4A", alpha=0.85, label="Test")
    ax2.axhline(70, color="#3C3489", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(85, color="#1D9E75", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("% del total por clase")
    ax2.set_title("Proporción relativa — verificar estratificación", fontweight="bold")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "split_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


def plot_tsne(X_train, y_train, le, out_dir, max_samples=1000):
    """
    Proyección t-SNE 2-D del conjunto de entrenamiento para verificar
    que las clases forman clusters separables (señal de que los features
    son discriminativos antes de entrenar el modelo).

    Usa PCA para reducir a 50 dims antes de t-SNE (más rápido y estable).
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("  ⚠  sklearn no disponible para t-SNE, omitiendo gráfica.")
        return

    # Aplanar (N, C, T) → (N, C*T)
    N    = X_train.shape[0]
    X_flat = X_train.reshape(N, -1).astype(np.float32)

    # Submuestrear si el dataset es grande
    if N > max_samples:
        idx    = np.random.default_rng(RANDOM_STATE).choice(N, max_samples, replace=False)
        X_flat = X_flat[idx]
        y_plot = y_train[idx]
    else:
        y_plot = y_train

    print(f"  t-SNE sobre {len(y_plot)} muestras del train set...")

    # PCA → 50 dims
    n_pca = min(50, X_flat.shape[1], X_flat.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=RANDOM_STATE).fit_transform(X_flat)

    # t-SNE → 2 dims
    tsne  = TSNE(n_components=2, random_state=RANDOM_STATE,
                 perplexity=min(30, len(y_plot) // 4),
                 n_iter=500, verbose=0)
    X_2d  = tsne.fit_transform(X_pca)

    # Gráfica
    classes = le.classes_
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, cls in enumerate(classes):
        mask = y_plot == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=cls,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            alpha=0.65, s=18, edgecolors="none",
        )

    ax.set_title(
        "Proyección t-SNE del train set (features MFCC)\n"
        "Clusters separados → features discriminativos ✔",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "split_tsne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────

def save_report(X_train, X_val, X_test,
                y_train, y_val, y_test,
                le, ratio, warnings_list, out_dir):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62
    classes  = le.classes_
    n_total  = len(y_train) + len(y_val) + len(y_test)

    log(sep)
    log("  REPORTE — SPLIT TRAIN/VAL/TEST (MÓDULO 4)")
    log(sep)

    log()
    log("CONFIGURACIÓN DEL SPLIT")
    log(sep2)
    log(f"  Ratio              : {ratio[0]*100:.0f} / {ratio[1]*100:.0f} / {ratio[2]*100:.0f}  (train/val/test)")
    log(f"  Estratificado      : Sí")
    log(f"  Random state       : {RANDOM_STATE}")
    log(f"  Total de muestras  : {n_total}")

    log()
    log("TAMAÑO DE CADA SPLIT")
    log(sep2)
    for name, y_s, X_s in [("Train", y_train, X_train),
                             ("Val  ", y_val,   X_val),
                             ("Test ", y_test,  X_test)]:
        pct = 100 * len(y_s) / n_total
        log(f"  {name} : {len(y_s):>5} muestras  ({pct:.1f}%)  shape X: {X_s.shape}")

    log()
    log("DISTRIBUCIÓN DE CLASES POR SPLIT")
    log(sep2)
    header = f"  {'Clase':<16} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    log(header)
    log(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    for i, cls in enumerate(classes):
        ct = int((y_train == i).sum())
        cv = int((y_val   == i).sum())
        ce = int((y_test  == i).sum())
        log(f"  {cls:<16} {ct:>6} {cv:>6} {ce:>6} {ct+cv+ce:>7}")

    log(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    log(f"  {'TOTAL':<16} {len(y_train):>6} {len(y_val):>6} {len(y_test):>6} {n_total:>7}")

    log()
    log("VERIFICACIÓN DE CALIDAD")
    log(sep2)
    if warnings_list:
        for w in warnings_list:
            log(f"  {w}")
    else:
        log("  ✔ Todas las clases presentes en los 3 splits")
        log("  ✔ Desbalance dentro de rangos aceptables")

    log()
    log("ARCHIVOS GUARDADOS")
    log(sep2)
    log("  X_train.npy  y_train.npy")
    log("  X_val.npy    y_val.npy")
    log("  X_test.npy   y_test.npy")
    log("  split_info.json")

    log()
    log("PRÓXIMO PASO")
    log(sep2)
    log("  Ejecutar train_cnn.py (módulo 5) usando la carpeta splits/")
    log()
    log(sep)

    report_path = os.path.join(out_dir, "split_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 4 — Splits train/val/test")
    parser.add_argument("--features", default="features_aug",
                        help="Carpeta con X_aug.npy y y_aug.npy (default: features_aug)")
    parser.add_argument("--output",   default="splits",
                        help="Carpeta de salida para los .npy (default: splits)")
    parser.add_argument("--ratio",    nargs=3, type=float, default=[70, 15, 15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Proporción train/val/test como porcentajes (default: 70 15 15)")
    parser.add_argument("--no-tsne",  action="store_true",
                        help="Omitir gráfica t-SNE (más rápido)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Omitir todas las gráficas")
    args = parser.parse_args()

    # ── Validar rutas ─────────────────────────
    if not os.path.isdir(args.features):
        print(f"✖ No se encontró la carpeta de features: {args.features}")
        print( "  Ejecuta primero augment_dataset.py (módulo 3).")
        return

    ratio = parse_ratio(args.ratio)

    os.makedirs(args.output, exist_ok=True)
    reports_dir = os.path.join(args.output, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ── 1. Cargar features ────────────────────
    print(f"\nCargando features desde: {os.path.abspath(args.features)}")
    X, y, le = load_features(args.features)
    print(f"  X shape  : {X.shape}")
    print(f"  y shape  : {y.shape}")
    print(f"  Clases   : {list(le.classes_)}")
    print(f"  Ratio    : {ratio[0]*100:.0f}/{ratio[1]*100:.0f}/{ratio[2]*100:.0f}")

    # ── 2. Split ──────────────────────────────
    print("\nAplicando split estratificado...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, ratio)

    print(f"  Train : {X_train.shape}  ({len(y_train)} muestras)")
    print(f"  Val   : {X_val.shape}    ({len(y_val)} muestras)")
    print(f"  Test  : {X_test.shape}   ({len(y_test)} muestras)")

    # ── 3. Verificar calidad ──────────────────
    warnings_list = verify_splits(y_train, y_val, y_test, le)
    if warnings_list:
        print("\nAdvertencias:")
        for w in warnings_list:
            print(f"  {w}")
    else:
        print("  ✔ Verificación de splits superada")

    # ── 4. Guardar arrays ─────────────────────
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val"  : X_val,   "y_val"  : y_val,
        "X_test" : X_test,  "y_test" : y_test,
    }
    for name, arr in splits.items():
        path = os.path.join(args.output, f"{name}.npy")
        np.save(path, arr)

    # Copiar label_encoder y feature_config al output
    import shutil
    for fname in ["label_encoder.pkl", "feature_config.json"]:
        src = os.path.join(args.features, fname)
        dst = os.path.join(args.output,   fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # split_info.json — metadatos reproducibles
    split_info = {
        "ratio"        : {"train": ratio[0], "val": ratio[1], "test": ratio[2]},
        "random_state" : RANDOM_STATE,
        "n_total"      : int(len(y)),
        "n_train"      : int(len(y_train)),
        "n_val"        : int(len(y_val)),
        "n_test"       : int(len(y_test)),
        "classes"      : list(le.classes_),
        "X_shape"      : list(X_train.shape[1:]),  # (C, T) por muestra
        "source_features": os.path.abspath(args.features),
    }
    info_path = os.path.join(args.output, "split_info.json")
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nArchivos guardados en: {os.path.abspath(args.output)}/")
    for name in splits:
        arr  = splits[name]
        path = os.path.join(args.output, f"{name}.npy")
        print(f"  {name}.npy → {arr.shape}  {arr.dtype}")
    print(f"  split_info.json")
    print(f"  label_encoder.pkl")
    print(f"  feature_config.json")

    # ── 5. Reporte + gráficas ─────────────────
    save_report(X_train, X_val, X_test,
                y_train, y_val, y_test,
                le, ratio, warnings_list, reports_dir)

    if not args.no_plots:
        print("\nGenerando gráficas...")
        plot_split_distribution(y_train, y_val, y_test, le, reports_dir)
        if not args.no_tsne:
            plot_tsne(X_train, y_train, le, reports_dir)
        else:
            print("  (t-SNE omitido con --no-tsne)")

    print("\n✔ Módulo 4 completado.")
    print("  Siguiente paso: train_cnn.py (módulo 5)\n")


if __name__ == "__main__":
    main()