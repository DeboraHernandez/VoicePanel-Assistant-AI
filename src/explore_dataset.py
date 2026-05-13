"""
MÓDULO 1 — Exploración y validación del dataset
Panel de domótica por voz — Raspberry Pi 4

Uso:
    python explore_dataset.py
    python explore_dataset.py --dataset ruta/al/dataset

Salida:
    - Reporte en consola
    - reports/dataset_report.txt
    - reports/class_distribution.png
    - reports/duration_distribution.png
    - reports/waveform_samples.png
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io.wavfile import read as wav_read
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

EXPECTED_COMMANDS = [
    "enciende", "apaga", "ventilador",
    "abrir",    "cerrar", "musica",
    "detente",  "ruido_fondo"
]

EXPECTED_SR      = 16000
EXPECTED_DURATION = 2.0      # segundos
DURATION_TOLERANCE = 0.15    # ±150 ms aceptable
MIN_AMPLITUDE    = 50        # muestras int16; por debajo = grabación vacía
REPORTS_DIR      = "reports"

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def load_wav(path):
    """Devuelve (sr, audio_int16) o lanza excepción con mensaje claro."""
    sr, audio = wav_read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]          # tomar canal 0 si es estéreo
    return sr, audio.astype(np.int16)


def amplitude_rms(audio):
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def is_silent(audio, threshold=MIN_AMPLITUDE):
    return amplitude_rms(audio) < threshold


# ─────────────────────────────────────────────
# ESCANEO DEL DATASET
# ─────────────────────────────────────────────

def scan_dataset(dataset_path):
    """
    Recorre dataset_path/comando/*.wav y devuelve una lista de dicts con
    toda la información por archivo, más una lista de errores encontrados.
    """
    records = []
    errors  = []

    for cmd in sorted(os.listdir(dataset_path)):
        cmd_path = os.path.join(dataset_path, cmd)
        if not os.path.isdir(cmd_path):
            continue

        wav_files = [f for f in os.listdir(cmd_path) if f.lower().endswith(".wav")]

        if not wav_files:
            errors.append(f"[VACÍO]   Carpeta sin archivos WAV: {cmd}/")
            continue

        for fname in sorted(wav_files):
            fpath = os.path.join(cmd_path, fname)
            rec = {
                "command"   : cmd,
                "file"      : fname,
                "path"      : fpath,
                "size_kb"   : os.path.getsize(fpath) / 1024,
                "sr"        : None,
                "n_samples" : None,
                "duration_s": None,
                "rms"       : None,
                "status"    : "OK",
                "issues"    : [],
            }

            # — Intento de lectura —
            try:
                sr, audio = load_wav(fpath)
                rec["sr"]         = sr
                rec["n_samples"]  = len(audio)
                rec["duration_s"] = len(audio) / sr
                rec["rms"]        = amplitude_rms(audio)

                # Validaciones
                if sr != EXPECTED_SR:
                    rec["issues"].append(f"SR={sr} (esperado {EXPECTED_SR})")

                dur_diff = abs(rec["duration_s"] - EXPECTED_DURATION)
                if dur_diff > DURATION_TOLERANCE:
                    rec["issues"].append(
                        f"Duración={rec['duration_s']:.2f}s "
                        f"(esperado {EXPECTED_DURATION}±{DURATION_TOLERANCE}s)"
                    )

                if is_silent(audio):
                    rec["issues"].append(f"RMS={rec['rms']:.1f} — posible grabación vacía")

                if rec["size_kb"] < 1.0:
                    rec["issues"].append(f"Archivo muy pequeño ({rec['size_kb']:.1f} KB)")

            except Exception as e:
                rec["status"] = "ERROR"
                rec["issues"].append(f"No se pudo leer: {e}")
                errors.append(f"[ERROR]   {fpath}: {e}")

            if rec["issues"]:
                rec["status"] = "WARN" if rec["status"] == "OK" else rec["status"]

            records.append(rec)

    return records, errors


# ─────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────

def analyse(records):
    df = pd.DataFrame(records)

    # ── Estadísticas por clase ──────────────────
    class_stats = (
        df.groupby("command")
        .agg(
            total        = ("file",       "count"),
            ok           = ("status",     lambda s: (s == "OK").sum()),
            warn         = ("status",     lambda s: (s == "WARN").sum()),
            error        = ("status",     lambda s: (s == "ERROR").sum()),
            dur_mean     = ("duration_s", "mean"),
            dur_std      = ("duration_s", "std"),
            rms_mean     = ("rms",        "mean"),
        )
        .reset_index()
    )

    # ── Archivos con problemas ──────────────────
    problems = df[df["status"] != "OK"].copy()

    # ── Clases faltantes ───────────────────────
    found_cmds   = set(df["command"].unique())
    missing_cmds = set(EXPECTED_COMMANDS) - found_cmds
    extra_cmds   = found_cmds - set(EXPECTED_COMMANDS)

    return df, class_stats, problems, missing_cmds, extra_cmds


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_class_distribution(df, out_dir):
    counts = df.groupby("command")["file"].count().sort_values(ascending=True)
    colors = ["#E24B4A" if c < 20 else "#1D9E75" for c in counts.values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none")
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Número de muestras", fontsize=11)
    ax.set_title("Distribución de clases en el dataset", fontsize=13, fontweight="bold")
    ax.axvline(counts.mean(), color="#BA7517", linestyle="--", linewidth=1.2,
               label=f"Promedio ({counts.mean():.0f})")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


def plot_duration_distribution(df, out_dir):
    df_ok = df[df["duration_s"].notna()]
    cmds  = df_ok["command"].unique()

    fig, ax = plt.subplots(figsize=(9, 4))
    data    = [df_ok[df_ok["command"] == c]["duration_s"].values for c in cmds]
    bp      = ax.boxplot(data, labels=cmds, patch_artist=True, vert=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("#9FE1CB")
        patch.set_alpha(0.7)

    ax.axhline(EXPECTED_DURATION, color="#E24B4A", linestyle="--",
               linewidth=1.2, label=f"Duración objetivo ({EXPECTED_DURATION}s)")
    ax.axhline(EXPECTED_DURATION + DURATION_TOLERANCE, color="#BA7517",
               linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(EXPECTED_DURATION - DURATION_TOLERANCE, color="#BA7517",
               linestyle=":", linewidth=1, alpha=0.7, label=f"Tolerancia ±{DURATION_TOLERANCE}s")
    ax.set_ylabel("Duración (s)", fontsize=11)
    ax.set_title("Distribución de duraciones por comando", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "duration_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado: {path}")


def plot_waveform_samples(dataset_path, df, out_dir):
    """Grafica una muestra aleatoria de cada clase lado a lado."""
    cmds = sorted(df["command"].unique())
    n    = len(cmds)
    cols = 4
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(14, rows * 2.8))
    gs  = gridspec.GridSpec(rows, cols, hspace=0.55, wspace=0.35)

    for idx, cmd in enumerate(cmds):
        subset = df[(df["command"] == cmd) & (df["status"] == "OK")]
        if subset.empty:
            continue

        row_s = subset.sample(1).iloc[0]
        try:
            sr, audio = load_wav(row_s["path"])
        except Exception:
            continue

        t  = np.linspace(0, len(audio) / sr, len(audio))
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.plot(t, audio, linewidth=0.4, color="#3C3489", alpha=0.85)
        ax.set_title(cmd, fontsize=11, fontweight="bold")
        ax.set_xlabel("s", fontsize=9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#888780", linewidth=0.5)

    fig.suptitle("Muestras de forma de onda por comando", fontsize=13, fontweight="bold", y=1.01)
    path = os.path.join(out_dir, "waveform_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE DE TEXTO
# ─────────────────────────────────────────────

def print_and_save_report(df, class_stats, problems, missing_cmds,
                          extra_cmds, scan_errors, out_dir):

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE DE EXPLORACIÓN DEL DATASET")
    log("  Panel de domótica por voz — Módulo 1")
    log(sep)

    # ── Resumen global ─────────────────────────
    total     = len(df)
    ok_count  = (df["status"] == "OK").sum()
    warn_count= (df["status"] == "WARN").sum()
    err_count = (df["status"] == "ERROR").sum()

    log()
    log("RESUMEN GLOBAL")
    log(sep2)
    log(f"  Total de archivos escaneados : {total}")
    log(f"  ✔  OK                        : {ok_count}")
    log(f"  ⚠  Con advertencias          : {warn_count}")
    log(f"  ✖  Con errores               : {err_count}")
    log(f"  Clases encontradas           : {df['command'].nunique()}")
    log(f"  Participantes únicos (aprox) : {df['file'].apply(lambda f: f.split('_')[1] if len(f.split('_'))>1 else '?').nunique()}")

    if df["duration_s"].notna().any():
        log(f"  Duración promedio            : {df['duration_s'].mean():.3f} s")
        log(f"  Duración mínima              : {df['duration_s'].min():.3f} s")
        log(f"  Duración máxima              : {df['duration_s'].max():.3f} s")

    # ── Clases faltantes / extra ───────────────
    log()
    log("CLASES")
    log(sep2)
    if missing_cmds:
        log(f"  ✖ Clases FALTANTES : {', '.join(sorted(missing_cmds))}")
    else:
        log("  ✔ Todas las clases esperadas están presentes")

    if extra_cmds:
        log(f"  ℹ Clases EXTRA (no esperadas) : {', '.join(sorted(extra_cmds))}")

    # ── Tabla por clase ────────────────────────
    log()
    log("DETALLE POR CLASE")
    log(sep2)
    log(f"  {'Comando':<16} {'Total':>6} {'OK':>5} {'Warn':>5} {'Error':>6} {'Dur_mean':>9} {'RMS_mean':>9}")
    log(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*9} {'-'*9}")
    for _, r in class_stats.iterrows():
        dur  = f"{r['dur_mean']:.3f}s" if pd.notna(r["dur_mean"]) else "  N/A  "
        rms  = f"{r['rms_mean']:.1f}"  if pd.notna(r["rms_mean"]) else "  N/A  "
        log(f"  {r['command']:<16} {r['total']:>6} {r['ok']:>5} {r['warn']:>5} {r['error']:>6} {dur:>9} {rms:>9}")

    # ── Balance de clases ─────────────────────
    counts = class_stats.set_index("command")["total"]
    imbalance = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    log()
    log("BALANCE DE CLASES")
    log(sep2)
    log(f"  Clase con más muestras  : {counts.idxmax()} ({counts.max()})")
    log(f"  Clase con menos muestras: {counts.idxmin()} ({counts.min()})")
    log(f"  Ratio desbalance        : {imbalance:.2f}x")
    if imbalance > 2.0:
        log("  ⚠  Desbalance notable — considerar data augmentation en clases pequeñas")
    else:
        log("  ✔ Balance aceptable")

    # ── Archivos con problemas ─────────────────
    if not problems.empty:
        log()
        log(f"ARCHIVOS CON PROBLEMAS ({len(problems)})")
        log(sep2)
        for _, r in problems.iterrows():
            log(f"  [{r['status']}] {r['command']}/{r['file']}")
            for issue in r["issues"]:
                log(f"         → {issue}")
    else:
        log()
        log("ARCHIVOS CON PROBLEMAS")
        log(sep2)
        log("  ✔ Ninguno — dataset limpio")

    # ── Errores de escaneo ────────────────────
    if scan_errors:
        log()
        log(f"ERRORES DE ESCANEO ({len(scan_errors)})")
        log(sep2)
        for e in scan_errors:
            log(f"  {e}")

    # ── Recomendaciones ───────────────────────
    log()
    log("RECOMENDACIONES PARA EL SIGUIENTE MÓDULO")
    log(sep2)

    if err_count > 0:
        log("  1. Eliminar o re-grabar archivos con ERROR antes de extraer features.")
    if warn_count > 0:
        log("  2. Revisar archivos con WARN — especialmente los de RMS bajo (silencio).")
    if imbalance > 1.5:
        log("  3. Aplicar data augmentation en módulo 3 priorizando clases pequeñas.")
    if ok_count >= total * 0.95:
        log("  ✔ Dataset en buenas condiciones para proceder al módulo 2 (feature extraction).")

    log()
    log(sep)
    log("  Fin del reporte")
    log(sep)

    # Guardar txt
    report_path = os.path.join(out_dir, "dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")

    return lines


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 1 — Exploración del dataset")
    parser.add_argument("--dataset", default="dataset",
                        help="Ruta a la carpeta dataset (default: ./dataset)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Omitir generación de gráficas")
    args = parser.parse_args()

    dataset_path = args.dataset

    # Verificar existencia
    if not os.path.isdir(dataset_path):
        print(f"✖ No se encontró la carpeta: {dataset_path}")
        print(  "  Verifica que el dataset está en la ruta correcta.")
        return

    # Crear directorio de reportes
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"\nEscaneando dataset en: {os.path.abspath(dataset_path)}")
    print("─" * 50)

    # 1. Escanear
    records, scan_errors = scan_dataset(dataset_path)

    if not records:
        print("✖ No se encontraron archivos WAV. Verifica la estructura del dataset.")
        return

    # 2. Analizar
    df, class_stats, problems, missing_cmds, extra_cmds = analyse(records)

    # 3. Reporte en consola + txt
    print()
    print_and_save_report(df, class_stats, problems, missing_cmds,
                          extra_cmds, scan_errors, REPORTS_DIR)

    # 4. Gráficas
    if not args.no_plots:
        print("\nGenerando gráficas...")
        plot_class_distribution(df, REPORTS_DIR)
        plot_duration_distribution(df, REPORTS_DIR)
        plot_waveform_samples(dataset_path, df, REPORTS_DIR)

    # 5. CSV de problemas (útil para el equipo)
    if not problems.empty:
        prob_path = os.path.join(REPORTS_DIR, "problematic_files.csv")
        problems[["command", "file", "status", "issues"]].to_csv(prob_path, index=False)
        print(f"  CSV de problemas guardado en: {prob_path}")

    print("\n✔ Módulo 1 completado. Siguiente paso: feature_extraction.py (módulo 2)\n")


if __name__ == "__main__":
    main()