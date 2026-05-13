"""
MÓDULO 3 — Data Augmentation
Panel de domótica por voz — Raspberry Pi 4

Aplica 3 técnicas de augmentation sobre los WAV originales:
  1. Time shifting   — desplaza la señal en el tiempo
  2. Ruido gaussiano — inyecta ruido a SNR controlado
  3. Pitch shifting  — cambia el tono sin alterar la velocidad

Genera nuevos WAV en dataset_aug/ (misma estructura que dataset/)
y re-ejecuta la extracción MFCC sobre el dataset combinado,
guardando X_aug.npy / y_aug.npy listos para el módulo 4.

Uso:
    python augment_dataset.py
    python augment_dataset.py --dataset dataset --output dataset_aug --features features_aug

Salida:
    dataset_aug/          ← WAVs aumentados (misma estructura)
    features_aug/
        X_aug.npy         → (N_total, 39, 125)  float32
        y_aug.npy         → (N_total,)           int32
        label_encoder.pkl
        feature_config.json
        reports/
            augmentation_report.txt
            aug_class_distribution.png
            aug_waveform_comparison.png
            aug_mfcc_comparison.png
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from scipy.io.wavfile import read as wav_read, write as wav_write
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE AUGMENTATION
# ─────────────────────────────────────────────

AUG_CFG = {
    # Time shifting
    "shift_max_pct"   : 0.20,    # desplazamiento máximo ±20 % de la duración
    "shift_n"         : 2,       # cuántas versiones con shift por muestra

    # Ruido gaussiano
    "noise_snr_db"    : 15,      # SNR objetivo en dB (15 = ruido moderado)
    "noise_n"         : 2,       # versiones con ruido por muestra

    # Pitch shifting
    "pitch_semitones" : [-2, 2], # lista de semitonos a aplicar
    # (−2 y +2 = 2 versiones por muestra)
}

# ─────────────────────────────────────────────
# CONFIGURACIÓN MFCC  (debe ser idéntica al módulo 2)
# ─────────────────────────────────────────────

MFCC_CFG = {
    "sample_rate" : 16000,
    "duration_s"  : 2.0,
    "n_mfcc"      : 13,
    "n_fft"       : 512,
    "hop_length"  : 256,
    "win_length"  : 400,
    "n_mels"      : 40,
    "use_delta"   : True,
    "use_delta2"  : True,
    "normalize"   : True,
    "n_frames"    : 125,
}

N_CHANNELS = MFCC_CFG["n_mfcc"] * (
    1 + int(MFCC_CFG["use_delta"]) + int(MFCC_CFG["use_delta2"])
)

# Técnicas habilitadas (puedes desactivar alguna cambiando a False)
TECHNIQUES = {
    "time_shift" : True,
    "noise"      : True,
    "pitch_shift": True,
}

# ─────────────────────────────────────────────
# UTILIDADES DE AUDIO
# ─────────────────────────────────────────────

def load_wav_float(path, target_sr=16000):
    """Carga WAV como float32 en [-1, 1] a target_sr."""
    sr, audio = wav_read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32) / 32768.0
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def save_wav_float(path, audio, sr=16000):
    """Guarda float32 [-1, 1] como WAV int16."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16   = (audio_clipped * 32767).astype(np.int16)
    wav_write(path, sr, audio_int16)


def fix_length(audio, target_sr, target_duration):
    """Padding o truncado centrado a exactamente target_duration segundos."""
    target_len = int(target_duration * target_sr)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        excess = len(audio) - target_len
        start  = excess // 2
        audio  = audio[start : start + target_len]
    return audio


# ─────────────────────────────────────────────
# TÉCNICAS DE AUGMENTATION
# ─────────────────────────────────────────────

def time_shift(audio, sr, max_pct=0.20, seed=None):
    """
    Desplaza la señal aleatoriamente entre -max_pct y +max_pct
    de la duración total. El hueco se rellena con ceros.
    """
    rng        = np.random.default_rng(seed)
    shift_pct  = rng.uniform(-max_pct, max_pct)
    shift_samp = int(shift_pct * len(audio))

    if shift_samp > 0:
        # desplazar a la derecha
        shifted = np.concatenate([np.zeros(shift_samp, dtype=np.float32),
                                   audio[:-shift_samp]])
    elif shift_samp < 0:
        # desplazar a la izquierda
        shifted = np.concatenate([audio[-shift_samp:],
                                   np.zeros(-shift_samp, dtype=np.float32)])
    else:
        shifted = audio.copy()

    return shifted


def add_gaussian_noise(audio, snr_db=15, seed=None):
    """
    Inyecta ruido gaussiano blanco a un SNR objetivo expresado en dB.

    SNR (dB) = 10 · log10(P_señal / P_ruido)
    → P_ruido = P_señal / 10^(SNR/10)
    → sigma   = sqrt(P_ruido)
    """
    rng    = np.random.default_rng(seed)
    p_sig  = np.mean(audio ** 2) + 1e-10
    p_noise= p_sig / (10 ** (snr_db / 10))
    sigma  = np.sqrt(p_noise)
    noise  = rng.normal(0, sigma, size=len(audio)).astype(np.float32)
    return audio + noise


def pitch_shift(audio, sr, n_steps):
    """
    Cambia el tono n_steps semitonos sin alterar la velocidad.
    Usa el algoritmo de phase vocoder de librosa.
    n_steps > 0 → tono más agudo
    n_steps < 0 → tono más grave
    """
    return librosa.effects.pitch_shift(
        y        = audio,
        sr       = sr,
        n_steps  = float(n_steps),
    )


# ─────────────────────────────────────────────
# PIPELINE DE AUGMENTATION
# ─────────────────────────────────────────────

def augment_file(audio, sr, aug_cfg, seed_base=0):
    """
    Aplica todas las técnicas habilitadas sobre un audio y devuelve
    una lista de (audio_aumentado, nombre_tecnica).
    """
    augmented = []

    if TECHNIQUES["time_shift"]:
        for i in range(aug_cfg["shift_n"]):
            aug = time_shift(audio, sr,
                             max_pct=aug_cfg["shift_max_pct"],
                             seed=seed_base + i)
            aug = fix_length(aug, sr, MFCC_CFG["duration_s"])
            augmented.append((aug, f"shift{i+1}"))

    if TECHNIQUES["noise"]:
        for i in range(aug_cfg["noise_n"]):
            aug = add_gaussian_noise(audio,
                                     snr_db=aug_cfg["noise_snr_db"],
                                     seed=seed_base + 100 + i)
            aug = fix_length(aug, sr, MFCC_CFG["duration_s"])
            augmented.append((aug, f"noise{i+1}"))

    if TECHNIQUES["pitch_shift"]:
        for steps in aug_cfg["pitch_semitones"]:
            aug = pitch_shift(audio, sr, n_steps=steps)
            aug = fix_length(aug, sr, MFCC_CFG["duration_s"])
            tag = f"pitch{'p' if steps > 0 else 'm'}{abs(steps)}"
            augmented.append((aug, tag))

    return augmented


def generate_augmented_wavs(dataset_path, aug_output_path, aug_cfg):
    """
    Recorre dataset_path, genera WAVs aumentados en aug_output_path.
    Devuelve estadísticas: {comando: {original: N, augmented: M}}.
    """
    stats    = {}
    failed   = []

    commands = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    for cmd in commands:
        cmd_src = os.path.join(dataset_path, cmd)
        cmd_dst = os.path.join(aug_output_path, cmd)
        os.makedirs(cmd_dst, exist_ok=True)

        wav_files = sorted([
            f for f in os.listdir(cmd_src)
            if f.lower().endswith(".wav")
        ])

        aug_count = 0

        for idx, fname in enumerate(tqdm(wav_files, desc=f"  {cmd:<16}", unit="wav")):
            src_path = os.path.join(cmd_src, fname)
            try:
                audio, sr = load_wav_float(src_path, target_sr=MFCC_CFG["sample_rate"])
                audio     = fix_length(audio, sr, MFCC_CFG["duration_s"])

                augmented = augment_file(audio, sr, aug_cfg, seed_base=idx * 1000)

                for aug_audio, tag in augmented:
                    stem     = os.path.splitext(fname)[0]
                    out_name = f"{stem}_aug_{tag}.wav"
                    out_path = os.path.join(cmd_dst, out_name)
                    save_wav_float(out_path, aug_audio, sr=sr)
                    aug_count += 1

            except Exception as e:
                failed.append({"file": src_path, "error": str(e)})

        stats[cmd] = {
            "original" : len(wav_files),
            "augmented": aug_count,
            "total"    : len(wav_files) + aug_count,
        }
        print(f"       → {len(wav_files)} orig + {aug_count} aug = {stats[cmd]['total']} total")

    return stats, failed


# ─────────────────────────────────────────────
# EXTRACCIÓN MFCC (igual que módulo 2, inline)
# ─────────────────────────────────────────────

def extract_mfcc(audio, cfg):
    mfcc = librosa.feature.mfcc(
        y          = audio,
        sr         = cfg["sample_rate"],
        n_mfcc     = cfg["n_mfcc"],
        n_fft      = cfg["n_fft"],
        hop_length = cfg["hop_length"],
        win_length = cfg["win_length"],
        n_mels     = cfg["n_mels"],
    )
    features = [mfcc]
    if cfg["use_delta"]:
        features.append(librosa.feature.delta(mfcc, order=1))
    if cfg["use_delta2"]:
        features.append(librosa.feature.delta(mfcc, order=2))

    feat = np.concatenate(features, axis=0)

    T = feat.shape[1]
    if T < cfg["n_frames"]:
        feat = np.pad(feat, ((0, 0), (0, cfg["n_frames"] - T)), mode="constant")
    else:
        feat = feat[:, : cfg["n_frames"]]

    if cfg["normalize"]:
        mean = feat.mean(axis=1, keepdims=True)
        std  = feat.std(axis=1, keepdims=True) + 1e-8
        feat = (feat - mean) / std

    return feat.astype(np.float32)


def process_folder_to_npy(folder_path, cfg):
    """
    Extrae MFCC de todos los WAV en folder_path/comando/*.wav.
    Devuelve X (N, C, T), y_str (N,), failed [].
    """
    X_list, labels_list, failed = [], [], []

    commands = sorted([
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ])

    for cmd in commands:
        cmd_path  = os.path.join(folder_path, cmd)
        wav_files = sorted([f for f in os.listdir(cmd_path) if f.lower().endswith(".wav")])

        for fname in tqdm(wav_files, desc=f"  MFCC {cmd:<12}", unit="wav", leave=False):
            fpath = os.path.join(cmd_path, fname)
            try:
                audio, _ = load_wav_float(fpath, target_sr=cfg["sample_rate"])
                audio    = fix_length(audio, cfg["sample_rate"], cfg["duration_s"])
                feat     = extract_mfcc(audio, cfg)
                X_list.append(feat)
                labels_list.append(cmd)
            except Exception as e:
                failed.append({"file": fpath, "error": str(e)})

    return np.stack(X_list), np.array(labels_list, dtype=object), failed


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_class_distribution(stats, out_dir):
    cmds   = list(stats.keys())
    orig   = [stats[c]["original"]  for c in cmds]
    aug    = [stats[c]["augmented"] for c in cmds]
    x      = np.arange(len(cmds))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - width / 2, orig, width, label="Original",  color="#3C3489", alpha=0.85)
    b2 = ax.bar(x + width / 2, aug,  width, label="Aumentado", color="#1D9E75", alpha=0.85)
    ax.bar_label(b1, fontsize=9, padding=2)
    ax.bar_label(b2, fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(cmds, rotation=30, ha="right")
    ax.set_ylabel("Número de muestras")
    ax.set_title("Dataset original vs aumentado por clase", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "aug_class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


def plot_waveform_comparison(dataset_path, out_dir, sr=16000):
    """
    Para el primer WAV de la primera clase, muestra el original
    y las versiones aumentadas en subplots.
    """
    commands  = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    if not commands:
        return

    cmd      = commands[0]
    cmd_path = os.path.join(dataset_path, cmd)
    wavs     = sorted([f for f in os.listdir(cmd_path) if f.lower().endswith(".wav")])
    if not wavs:
        return

    try:
        audio, sr = load_wav_float(os.path.join(cmd_path, wavs[0]), target_sr=sr)
        audio     = fix_length(audio, sr, MFCC_CFG["duration_s"])
    except Exception:
        return

    augmented = augment_file(audio, sr, AUG_CFG, seed_base=42)
    all_pairs = [("Original", audio)] + [(tag, a) for a, tag in augmented]

    cols = 3
    rows = (len(all_pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 2.5), sharex=True)
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]
    t = np.linspace(0, len(audio) / sr, len(audio))

    for i, (label, sig) in enumerate(all_pairs):
        ax = axes_flat[i]
        ax.plot(t, sig, linewidth=0.5,
                color="#3C3489" if label == "Original" else "#1D9E75", alpha=0.9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("s", fontsize=8)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#aaa", linewidth=0.4)

    for j in range(len(all_pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Comparación de técnicas de augmentation — '{cmd}'",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "aug_waveform_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


def plot_mfcc_comparison(dataset_path, out_dir, sr=16000):
    """
    Muestra el MFCC del original y de cada augmentación para una muestra.
    """
    commands  = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    if not commands:
        return

    cmd      = commands[0]
    cmd_path = os.path.join(dataset_path, cmd)
    wavs     = sorted([f for f in os.listdir(cmd_path) if f.lower().endswith(".wav")])
    if not wavs:
        return

    try:
        audio, sr = load_wav_float(os.path.join(cmd_path, wavs[0]), target_sr=sr)
        audio     = fix_length(audio, sr, MFCC_CFG["duration_s"])
    except Exception:
        return

    augmented = augment_file(audio, sr, AUG_CFG, seed_base=42)
    all_pairs = [("Original", audio)] + [(tag, a) for a, tag in augmented]

    cols = 3
    rows = (len(all_pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3))
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]

    for i, (label, sig) in enumerate(all_pairs):
        feat = extract_mfcc(sig, MFCC_CFG)[:13, :]   # solo MFCC base
        ax   = axes_flat[i]
        im   = ax.imshow(feat, aspect="auto", origin="lower",
                         cmap="magma", interpolation="nearest")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Frames", fontsize=8)
        ax.set_ylabel("Coef.", fontsize=8)
        plt.colorbar(im, ax=ax, format="%.1f", pad=0.02)

    for j in range(len(all_pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"MFCC base tras cada augmentation — '{cmd}'",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "aug_mfcc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────

def save_report(stats, failed_gen, failed_mfcc, X_aug, y_aug, out_dir):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE — DATA AUGMENTATION (MÓDULO 3)")
    log(sep)

    log()
    log("TÉCNICAS APLICADAS")
    log(sep2)
    enabled = [k for k, v in TECHNIQUES.items() if v]
    for t in enabled:
        log(f"  ✔ {t}")
    for t in [k for k, v in TECHNIQUES.items() if not v]:
        log(f"  ✖ {t}  (desactivada)")

    log()
    log("PARÁMETROS DE AUGMENTATION")
    log(sep2)
    for k, v in AUG_CFG.items():
        log(f"  {k:<22}: {v}")

    log()
    log("DISTRIBUCIÓN RESULTANTE")
    log(sep2)
    log(f"  {'Clase':<16} {'Original':>8} {'Aumentado':>10} {'Total':>7}")
    log(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*7}")
    for cmd, s in stats.items():
        log(f"  {cmd:<16} {s['original']:>8} {s['augmented']:>10} {s['total']:>7}")
    log(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*7}")
    total_orig = sum(s["original"]  for s in stats.values())
    total_aug  = sum(s["augmented"] for s in stats.values())
    total_all  = sum(s["total"]     for s in stats.values())
    log(f"  {'TOTAL':<16} {total_orig:>8} {total_aug:>10} {total_all:>7}")

    log()
    log("ARRAYS FINALES")
    log(sep2)
    log(f"  X_aug.npy shape : {X_aug.shape}  float32")
    log(f"  y_aug.npy shape : {y_aug.shape}  int32")
    log(f"  Tamaño en disco : ~{X_aug.nbytes / 1e6:.1f} MB")

    if failed_gen:
        log()
        log(f"ERRORES EN GENERACIÓN ({len(failed_gen)})")
        log(sep2)
        for f in failed_gen:
            log(f"  {f['file']}: {f['error']}")

    if failed_mfcc:
        log()
        log(f"ERRORES EN EXTRACCIÓN MFCC ({len(failed_mfcc)})")
        log(sep2)
        for f in failed_mfcc:
            log(f"  {f['file']}: {f['error']}")

    log()
    log("PRÓXIMO PASO")
    log(sep2)
    log("  Ejecutar prepare_splits.py (módulo 4) usando features_aug/")
    log()
    log(sep)

    report_path = os.path.join(out_dir, "augmentation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 3 — Data Augmentation")
    parser.add_argument("--dataset",  default="dataset",
                        help="Dataset original (default: ./dataset)")
    parser.add_argument("--output",   default="dataset_aug",
                        help="Carpeta para WAVs aumentados (default: ./dataset_aug)")
    parser.add_argument("--features", default="features_aug",
                        help="Carpeta para .npy aumentados (default: ./features_aug)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Saltar generación de WAVs (solo re-extraer MFCC)")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"✖ No se encontró: {args.dataset}")
        return

    os.makedirs(args.output,   exist_ok=True)
    os.makedirs(args.features, exist_ok=True)
    reports_dir = os.path.join(args.features, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ── PASO 1: copiar originales al output ──────────
    # (para que el dataset_aug contenga original + aumentado)
    print("\n── Paso 1/3: Copiando originales a dataset_aug/")
    import shutil
    for cmd in os.listdir(args.dataset):
        src = os.path.join(args.dataset, cmd)
        dst = os.path.join(args.output, cmd)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                if f.lower().endswith(".wav"):
                    fsrc = os.path.join(src, f)
                    fdst = os.path.join(dst, f)
                    if not os.path.exists(fdst):
                        shutil.copy2(fsrc, fdst)
    print("  ✔ Originales copiados")

    # ── PASO 2: generar aumentados ────────────────────
    if not args.skip_gen:
        print("\n── Paso 2/3: Generando WAVs aumentados")
        n_tecnicas = sum(TECHNIQUES.values())
        n_versiones = (
            AUG_CFG["shift_n"] * int(TECHNIQUES["time_shift"]) +
            AUG_CFG["noise_n"] * int(TECHNIQUES["noise"]) +
            len(AUG_CFG["pitch_semitones"]) * int(TECHNIQUES["pitch_shift"])
        )
        print(f"  Técnicas activas: {n_tecnicas}  |  Versiones por muestra: {n_versiones}")
        stats, failed_gen = generate_augmented_wavs(args.dataset, args.output, AUG_CFG)
    else:
        print("\n── Paso 2/3: (generación de WAVs omitida con --skip-gen)")
        stats      = {}
        failed_gen = []

    # ── PASO 3: extraer MFCC del dataset combinado ───
    print("\n── Paso 3/3: Extrayendo MFCC del dataset combinado")
    X_aug, y_str, failed_mfcc = process_folder_to_npy(args.output, MFCC_CFG)

    le = LabelEncoder()
    y_aug = le.fit_transform(y_str).astype(np.int32)

    x_path  = os.path.join(args.features, "X_aug.npy")
    y_path  = os.path.join(args.features, "y_aug.npy")
    le_path = os.path.join(args.features, "label_encoder.pkl")
    cfg_path= os.path.join(args.features, "feature_config.json")

    np.save(x_path,  X_aug)
    np.save(y_path,  y_aug)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    with open(cfg_path, "w") as f:
        json.dump(MFCC_CFG, f, indent=2)

    print(f"\nArchivos guardados:")
    print(f"  {x_path}  → {X_aug.shape}")
    print(f"  {y_path}  → {y_aug.shape}")
    print(f"  {le_path}")
    print(f"  {cfg_path}")

    # ── Reporte + gráficas ────────────────────────────
    save_report(stats, failed_gen, failed_mfcc, X_aug, y_aug, reports_dir)

    if not args.no_plots:
        print("\nGenerando gráficas...")
        if stats:
            plot_class_distribution(stats, reports_dir)
        plot_waveform_comparison(args.dataset, reports_dir)
        plot_mfcc_comparison(args.dataset, reports_dir)

    print("\n✔ Módulo 3 completado.")
    print("  Siguiente paso: prepare_splits.py (módulo 4)\n")


if __name__ == "__main__":
    main()