"""
MÓDULO 2 — Extracción de features MFCC
Panel de domótica por voz — Raspberry Pi 4

Lee cada WAV del dataset, extrae MFCC + Delta + Delta-Delta,
aplica padding/truncado a longitud fija y guarda los arrays
listos para entrenar.

Uso:
    python feature_extraction.py
    python feature_extraction.py --dataset ruta/dataset --output ruta/features

Salida (en features/):
    X.npy              → shape (N, 39, 63)   float32
    y.npy              → shape (N,)           int32
    label_encoder.pkl  → LabelEncoder de sklearn
    feature_config.json→ parámetros usados (para reproducir en inferencia)
    reports/mfcc_report.txt
    reports/mfcc_samples.png
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wav_read
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURACIÓN — estos valores deben coincidir
# exactamente en feature_extraction.py,
# train_cnn.py y realtime_pipeline.py
# ─────────────────────────────────────────────

CFG = {
    # Audio
    "sample_rate"  : 16000,
    "duration_s"   : 2.0,

    # MFCC
    "n_mfcc"       : 13,        # coeficientes base
    "n_fft"        : 512,       # ~32 ms a 16 kHz
    "hop_length"   : 256,       # ~16 ms → solapamiento 50 %
    "win_length"   : 400,       # ~25 ms (estándar MFCC)
    "n_mels"       : 40,        # banco de filtros mel

    # Delta
    "use_delta"    : True,      # agrega delta  (+13 coef.)
    "use_delta2"   : True,      # agrega delta² (+13 coef.)
                                # total canales: 39

    # Normalización
    "normalize"    : True,      # mean=0, std=1 por muestra

    # Shape final
    # n_frames = ceil(duration_s * sample_rate / hop_length)
    #          = ceil(32000 / 256) = 125
    "n_frames"     : 125,
}

# canales totales según config
N_CHANNELS = CFG["n_mfcc"] * (1 + int(CFG["use_delta"]) + int(CFG["use_delta2"]))

COMMANDS = [
    "enciende", "apaga", "ventilador",
    "abrir",    "cerrar", "musica",
    "detente",  "ruido_fondo",
]

# ─────────────────────────────────────────────
# CARGA Y PREPROCESADO DE AUDIO
# ─────────────────────────────────────────────

def load_audio(path, target_sr, target_duration):
    """
    Carga un WAV con scipy (rápido, sin ffmpeg),
    convierte a float32 en [-1, 1], y ajusta la
    longitud exactamente a target_duration segundos
    mediante padding con ceros o truncado.
    """
    sr, audio = wav_read(path)

    # Estéreo → mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # int16 → float32 normalizado
    audio = audio.astype(np.float32) / 32768.0

    # Remuestrear si es necesario (no debería ocurrir con el script de grabación)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Longitud objetivo en muestras
    target_len = int(target_duration * target_sr)

    if len(audio) < target_len:
        # Padding con ceros al final
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        # Truncar centrado (descarta igual de ambos lados)
        excess = len(audio) - target_len
        start  = excess // 2
        audio  = audio[start : start + target_len]

    return audio


# ─────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES
# ─────────────────────────────────────────────

def extract_mfcc(audio, cfg):
    """
    Extrae MFCC + (opcionalmente) delta y delta² de un array float32.

    Devuelve un array de shape (n_channels, n_frames).
    """
    mfcc = librosa.feature.mfcc(
        y          = audio,
        sr         = cfg["sample_rate"],
        n_mfcc     = cfg["n_mfcc"],
        n_fft      = cfg["n_fft"],
        hop_length = cfg["hop_length"],
        win_length = cfg["win_length"],
        n_mels     = cfg["n_mels"],
    )                                      # shape: (n_mfcc, T)

    features = [mfcc]

    if cfg["use_delta"]:
        features.append(librosa.feature.delta(mfcc, order=1))

    if cfg["use_delta2"]:
        features.append(librosa.feature.delta(mfcc, order=2))

    feat = np.concatenate(features, axis=0)   # (n_channels, T)

    # ── Ajustar n_frames ─────────────────────
    target_frames = cfg["n_frames"]
    T = feat.shape[1]

    if T < target_frames:
        pad = target_frames - T
        feat = np.pad(feat, ((0, 0), (0, pad)), mode="constant")
    else:
        feat = feat[:, :target_frames]

    # ── Normalización por muestra (mean=0, std=1) ─
    if cfg["normalize"]:
        mean = feat.mean(axis=1, keepdims=True)
        std  = feat.std(axis=1, keepdims=True) + 1e-8
        feat = (feat - mean) / std

    return feat.astype(np.float32)           # (n_channels, n_frames)


# ─────────────────────────────────────────────
# PROCESAMIENTO DEL DATASET COMPLETO
# ─────────────────────────────────────────────

def process_dataset(dataset_path, cfg):
    """
    Recorre dataset_path/comando/*.wav y devuelve
    X (N, n_channels, n_frames), labels (N,) str,
    y una lista de archivos que fallaron.
    """
    X_list      = []
    labels_list = []
    failed      = []

    commands = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    print(f"\nClases detectadas ({len(commands)}): {commands}\n")

    for cmd in commands:
        cmd_path  = os.path.join(dataset_path, cmd)
        wav_files = sorted([
            f for f in os.listdir(cmd_path)
            if f.lower().endswith(".wav")
        ])

        if not wav_files:
            print(f"  ⚠  {cmd}: carpeta vacía, se omite.")
            continue

        ok_count = 0
        for fname in tqdm(wav_files, desc=f"  {cmd:<16}", unit="wav"):
            fpath = os.path.join(cmd_path, fname)
            try:
                audio = load_audio(fpath, cfg["sample_rate"], cfg["duration_s"])
                feat  = extract_mfcc(audio, cfg)
                X_list.append(feat)
                labels_list.append(cmd)
                ok_count += 1
            except Exception as e:
                failed.append({"file": fpath, "error": str(e)})

        print(f"       → {ok_count}/{len(wav_files)} muestras extraídas")

    X      = np.stack(X_list,  axis=0)              # (N, C, T)
    labels = np.array(labels_list, dtype=object)    # (N,)

    return X, labels, failed


# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────

def plot_mfcc_samples(X, y_str, label_encoder, out_dir):
    """
    Muestra el MFCC (canal 0 = base, sin delta) de una muestra por clase.
    """
    classes = label_encoder.classes_
    n       = len(classes)
    cols    = 4
    rows    = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, cls in enumerate(classes):
        mask    = y_str == cls
        indices = np.where(mask)[0]
        if len(indices) == 0:
            axes_flat[idx].set_visible(False)
            continue

        sample = X[indices[0]]              # (C, T)
        mfcc_only = sample[:13, :]          # solo los 13 coef. base

        ax = axes_flat[idx]
        im = ax.imshow(
            mfcc_only,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.set_xlabel("Frames", fontsize=9)
        ax.set_ylabel("Coef. MFCC", fontsize=9)
        plt.colorbar(im, ax=ax, format="%.1f", pad=0.02)

    # Ocultar ejes sobrantes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"MFCC base (13 coef.) — una muestra por clase\n"
        f"Shape completo con delta+delta²: {X.shape}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "mfcc_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


def plot_feature_channels(X, y_str, out_dir):
    """
    Para la primera muestra de la primera clase,
    muestra los 3 canales (MFCC, Δ, Δ²) apilados.
    """
    cls         = np.unique(y_str)[0]
    sample      = X[y_str == cls][0]        # (39, 125)
    n_mfcc      = 13
    chan_labels  = ["MFCC base", "Delta (Δ)", "Delta² (Δ²)"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    cmaps     = ["magma", "viridis", "cividis"]

    for i, (label, cmap) in enumerate(zip(chan_labels, cmaps)):
        chunk = sample[i * n_mfcc : (i + 1) * n_mfcc, :]
        ax    = axes[i]
        im    = ax.imshow(
            chunk, aspect="auto", origin="lower",
            cmap=cmap, interpolation="nearest",
        )
        ax.set_ylabel(label, fontsize=10)
        plt.colorbar(im, ax=ax, format="%.2f", pad=0.01)

    axes[-1].set_xlabel("Frames (tiempo →)", fontsize=10)
    fig.suptitle(
        f"Los 3 canales de features — comando: '{cls}'\n"
        f"Cada canal: (13, {sample.shape[1]})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_channels.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────

def save_report(X, y_int, y_str, label_encoder, failed, out_dir):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE — EXTRACCIÓN DE FEATURES (MÓDULO 2)")
    log(sep)
    log()
    log("PARÁMETROS UTILIZADOS")
    log(sep2)
    for k, v in CFG.items():
        log(f"  {k:<20}: {v}")

    log()
    log("SHAPE DE SALIDA")
    log(sep2)
    log(f"  X.npy   : {X.shape}  (muestras, canales, frames)")
    log(f"  y.npy   : {y_int.shape}")
    log(f"  dtype X : {X.dtype}")
    log(f"  dtype y : {y_int.dtype}")
    log(f"  Tamaño X en disco: ~{X.nbytes / 1e6:.1f} MB")

    log()
    log("DISTRIBUCIÓN DE CLASES")
    log(sep2)
    classes, counts = np.unique(y_str, return_counts=True)
    for cls, cnt in zip(classes, counts):
        bar = "█" * (cnt // 2)
        log(f"  {cls:<16} {cnt:>4}  {bar}")

    log()
    log("ESTADÍSTICAS DE FEATURES")
    log(sep2)
    log(f"  Valor mínimo global : {X.min():.4f}")
    log(f"  Valor máximo global : {X.max():.4f}")
    log(f"  Media global        : {X.mean():.4f}  (≈0 si normalización OK)")
    log(f"  Std global          : {X.std():.4f}   (≈1 si normalización OK)")

    if failed:
        log()
        log(f"ARCHIVOS QUE FALLARON ({len(failed)})")
        log(sep2)
        for f in failed:
            log(f"  {f['file']}: {f['error']}")
    else:
        log()
        log("ARCHIVOS QUE FALLARON : ninguno ✔")

    log()
    log("PRÓXIMO PASO")
    log(sep2)
    log("  Ejecutar augment_dataset.py (módulo 3) o")
    log("  prepare_splits.py (módulo 4) si ya tienes suficientes muestras.")
    log()
    log(sep)

    report_path = os.path.join(out_dir, "mfcc_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 2 — Extracción MFCC")
    parser.add_argument("--dataset", default="dataset",
                        help="Carpeta raíz del dataset (default: ./dataset)")
    parser.add_argument("--output",  default="features",
                        help="Carpeta de salida para .npy y .pkl (default: ./features)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Omitir generación de gráficas")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"✖ No se encontró: {args.dataset}")
        return

    os.makedirs(args.output,  exist_ok=True)
    reports_dir = os.path.join(args.output, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ── 1. Extraer features ───────────────────
    print("─" * 50)
    print(f"Dataset : {os.path.abspath(args.dataset)}")
    print(f"Salida  : {os.path.abspath(args.output)}")
    print(f"Shape por muestra: ({N_CHANNELS}, {CFG['n_frames']})")
    print("─" * 50)

    X, y_str, failed = process_dataset(args.dataset, CFG)

    # ── 2. Codificar etiquetas ────────────────
    le = LabelEncoder()
    y_int = le.fit_transform(y_str).astype(np.int32)

    print(f"\nClases codificadas: {list(le.classes_)}")
    print(f"  ↳ índices       : {list(range(len(le.classes_)))}")

    # ── 3. Guardar arrays ─────────────────────
    x_path  = os.path.join(args.output, "X.npy")
    y_path  = os.path.join(args.output, "y.npy")
    le_path = os.path.join(args.output, "label_encoder.pkl")
    cfg_path= os.path.join(args.output, "feature_config.json")

    np.save(x_path, X)
    np.save(y_path, y_int)

    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    with open(cfg_path, "w") as f:
        json.dump(CFG, f, indent=2)

    print(f"\nArchivos guardados:")
    print(f"  {x_path}   → {X.shape}  float32")
    print(f"  {y_path}   → {y_int.shape}  int32")
    print(f"  {le_path}")
    print(f"  {cfg_path}")

    # ── 4. Reporte + gráficas ─────────────────
    save_report(X, y_int, y_str, le, failed, reports_dir)

    if not args.no_plots:
        print("\nGenerando gráficas...")
        plot_mfcc_samples(X, y_str, le, reports_dir)
        plot_feature_channels(X, y_str, reports_dir)

    print("\n✔ Módulo 2 completado.")
    print("  Siguiente paso: augment_dataset.py (módulo 3)\n")


if __name__ == "__main__":
    main()