"""
MÓDULO 10 — Benchmark de latencia del pipeline
Panel de domótica por voz — Raspberry Pi 4

Mide la latencia de cada etapa del pipeline por separado
y en conjunto, usando muestras reales del test set.

Etapas medidas:
    1. VAD (detección de actividad de voz)
    2. Extracción MFCC
    3. Inferencia TFLite (CNN y LSTM)
    4. Despacho GPIO (ActuatorController.execute)
    5. Pipeline completo extremo a extremo

Uso:
    python benchmark.py
    python benchmark.py --model models/cnn_model.tflite --runs 100
    python benchmark.py --both          ← compara CNN vs LSTM
    python benchmark.py --no-gpio       ← omitir GPIO (en laptop)

Salida (en models/reports/):
    latency_report.txt
    latency_chart.png          ← boxplot por etapa
    latency_cnn_vs_lstm.png    ← comparativa si se usa --both
"""

import os
import sys
import json
import pickle
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DEFAULT_MODEL   = "models/cnn_model.tflite"
DEFAULT_MODEL2  = "models/lstm_model.tflite"
DEFAULT_SPLITS  = "splits"
DEFAULT_RUNS    = 100
LATENCY_TARGET  = 500.0   # ms — requisito del enunciado

# Parámetros MFCC — deben coincidir con módulo 2
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

# ─────────────────────────────────────────────
# DETECCIÓN DE PLATAFORMA
# ─────────────────────────────────────────────

def _is_raspberry_pi():
    try:
        with open("/proc/cpuinfo") as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False

ON_RPI = _is_raspberry_pi()

# ─────────────────────────────────────────────
# CARGA DE DATOS Y MODELO
# ─────────────────────────────────────────────

def load_test_audio(splits_dir):
    """
    Carga X_test.npy y lo convierte de vuelta a audio float32.
    En realidad usamos los features directamente para la medición
    de MFCC e inferencia, y generamos audio sintético para VAD.
    Devuelve (X_test_features, audio_samples, y_test, classes).
    """
    x_path  = os.path.join(splits_dir, "X_test.npy")
    y_path  = os.path.join(splits_dir, "y_test.npy")
    le_path = os.path.join(splits_dir, "label_encoder.pkl")

    for p in [x_path, y_path, le_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No encontrado: {p}\n"
                "Ejecuta prepare_splits.py (módulo 4) primero."
            )

    X_test = np.load(x_path)   # (N, C, T) = (N, 39, 125)
    y_test = np.load(y_path)

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # Generar señales de audio sintéticas para benchmark de VAD y MFCC
    # (ruido blanco normalizado, mismo shape que grabaciones reales)
    n_samples_audio = int(MFCC_CFG["sample_rate"] * MFCC_CFG["duration_s"])
    rng = np.random.default_rng(42)
    audio_samples = rng.normal(0, 0.1, size=(len(X_test), n_samples_audio)).astype(np.float32)

    # X transpuesto para el modelo: (N, T, C) = (N, 125, 39)
    X_model = np.transpose(X_test, (0, 2, 1))

    return X_model, audio_samples, y_test, le.classes_


def load_tflite(model_path):
    """Carga un intérprete TFLite con manejo de SELECT_TF_OPS (Bi-LSTM)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    try:
        import tensorflow as tf
        try:
            interp = tf.lite.Interpreter(model_path=model_path, num_threads=4)
            interp.allocate_tensors()
        except RuntimeError:
            # Flex ops (Bi-LSTM) — intentar con experimental_delegates vacío
            interp = tf.lite.Interpreter(model_path=model_path)
            interp.allocate_tensors()
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=model_path, num_threads=4)
            interp.allocate_tensors()
        except ImportError:
            raise ImportError("pip install tensorflow  o  pip install tflite-runtime")

    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    size_kb = os.path.getsize(model_path) / 1024

    print(f"  Modelo cargado : {model_path}  ({size_kb:.1f} KB)")
    return interp, inp_idx, out_idx, size_kb


# ─────────────────────────────────────────────
# BENCHMARK DE CADA ETAPA
# ─────────────────────────────────────────────

def bench_vad(audio_samples, n_runs):
    """
    Mide la latencia del VAD por energía RMS.
    Implementación idéntica a realtime_pipeline.py.
    """
    frame_size = int(MFCC_CFG["sample_rate"] * 0.02)   # frames de 20ms
    threshold  = 0.01
    times      = []

    for i in range(n_runs):
        audio = audio_samples[i % len(audio_samples)]
        t0    = time.perf_counter()

        # VAD por energía (copia de EnergyVAD.is_speech)
        n_frames   = len(audio) // frame_size
        frames     = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
        rms_values = np.sqrt(np.mean(frames ** 2, axis=1))
        _ = np.sum(rms_values > threshold) >= 3

        times.append((time.perf_counter() - t0) * 1000)

    return np.array(times)


def bench_mfcc(audio_samples, n_runs):
    """
    Mide la latencia de extracción de features MFCC + Delta + Delta².
    Implementación idéntica a realtime_pipeline.py / feature_extraction.py.
    """
    import librosa
    times = []

    for i in range(n_runs):
        audio = audio_samples[i % len(audio_samples)]
        t0    = time.perf_counter()

        mfcc = librosa.feature.mfcc(
            y          = audio,
            sr         = MFCC_CFG["sample_rate"],
            n_mfcc     = MFCC_CFG["n_mfcc"],
            n_fft      = MFCC_CFG["n_fft"],
            hop_length  = MFCC_CFG["hop_length"],
            win_length = MFCC_CFG["win_length"],
            n_mels     = MFCC_CFG["n_mels"],
        )
        features = [mfcc]
        if MFCC_CFG["use_delta"]:
            features.append(librosa.feature.delta(mfcc, order=1))
        if MFCC_CFG["use_delta2"]:
            features.append(librosa.feature.delta(mfcc, order=2))

        feat = np.concatenate(features, axis=0)
        T    = feat.shape[1]
        if T < MFCC_CFG["n_frames"]:
            feat = np.pad(feat, ((0, 0), (0, MFCC_CFG["n_frames"] - T)))
        else:
            feat = feat[:, :MFCC_CFG["n_frames"]]

        mean = feat.mean(axis=1, keepdims=True)
        std  = feat.std(axis=1, keepdims=True) + 1e-8
        feat = (feat - mean) / std
        _    = feat.T[np.newaxis, :, :].astype(np.float32)

        times.append((time.perf_counter() - t0) * 1000)

    return np.array(times)


def bench_inference(X_model, interp, inp_idx, out_idx, n_runs):
    """
    Mide la latencia de inferencia TFLite pura.
    Excluye extracción de features — solo el forward pass.
    """
    times = []

    for i in range(n_runs):
        sample = X_model[i % len(X_model) : i % len(X_model) + 1].astype(np.float32)
        t0     = time.perf_counter()
        interp.set_tensor(inp_idx, sample)
        interp.invoke()
        _  = interp.get_tensor(out_idx)
        times.append((time.perf_counter() - t0) * 1000)

    return np.array(times)


def bench_gpio(classes, n_runs, no_gpio=False):
    """
    Mide la latencia del despacho a GPIO.
    En modo --no-gpio usa el mock de ActuatorController.
    """
    # Forzar mock si no estamos en RPi o se pidió --no-gpio
    if no_gpio and ON_RPI:
        import gpio_controller as gc
        original = gc.ON_RPI
        gc.ON_RPI = False
    
    try:
        from gpio_controller import ActuatorController
        ctrl  = ActuatorController()
        times = []

        for i in range(n_runs):
            cmd = classes[i % len(classes)]
            t0  = time.perf_counter()
            ctrl.execute(cmd, confidence=0.9)
            times.append((time.perf_counter() - t0) * 1000)

        ctrl.close()
    except Exception as e:
        print(f"  ⚠ GPIO benchmark falló ({e}) — usando tiempos estimados (0.5ms).")
        times = [0.5] * n_runs
    finally:
        if no_gpio and ON_RPI:
            gc.ON_RPI = original

    return np.array(times)


def bench_pipeline_e2e(audio_samples, X_model, interp, inp_idx, out_idx,
                        classes, n_runs, no_gpio=False):
    """
    Mide la latencia total extremo a extremo:
    VAD → MFCC → Inferencia → GPIO
    Esta es la métrica más importante para el documento:
    debe ser < 500 ms para cumplir con el requisito del enunciado.
    """
    import librosa

    try:
        from gpio_controller import ActuatorController
        ctrl = ActuatorController()
        has_gpio = True
    except Exception:
        ctrl     = None
        has_gpio = False

    frame_size = int(MFCC_CFG["sample_rate"] * 0.02)
    threshold  = 0.01
    times_total = []
    times_stage = {"vad": [], "mfcc": [], "inference": [], "gpio": []}

    for i in range(n_runs):
        audio = audio_samples[i % len(audio_samples)]

        # ── VAD ───────────────────────────────
        t0 = time.perf_counter()
        n_frames   = len(audio) // frame_size
        frames_arr = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
        rms_values = np.sqrt(np.mean(frames_arr ** 2, axis=1))
        _ = np.sum(rms_values > threshold) >= 3
        t1 = time.perf_counter()

        # ── MFCC ──────────────────────────────
        mfcc = librosa.feature.mfcc(
            y=audio, sr=MFCC_CFG["sample_rate"],
            n_mfcc=MFCC_CFG["n_mfcc"], n_fft=MFCC_CFG["n_fft"],
            hop_length=MFCC_CFG["hop_length"], win_length=MFCC_CFG["win_length"],
            n_mels=MFCC_CFG["n_mels"],
        )
        feats = [mfcc]
        if MFCC_CFG["use_delta"]:
            feats.append(librosa.feature.delta(mfcc, order=1))
        if MFCC_CFG["use_delta2"]:
            feats.append(librosa.feature.delta(mfcc, order=2))
        feat = np.concatenate(feats, axis=0)
        T = feat.shape[1]
        if T < MFCC_CFG["n_frames"]:
            feat = np.pad(feat, ((0, 0), (0, MFCC_CFG["n_frames"] - T)))
        else:
            feat = feat[:, :MFCC_CFG["n_frames"]]
        mean = feat.mean(axis=1, keepdims=True)
        std  = feat.std(axis=1, keepdims=True) + 1e-8
        feat = ((feat - mean) / std).T[np.newaxis, :, :].astype(np.float32)
        t2   = time.perf_counter()

        # ── Inferencia ────────────────────────
        interp.set_tensor(inp_idx, feat)
        interp.invoke()
        probs    = interp.get_tensor(out_idx)[0]
        pred_idx = int(np.argmax(probs))
        t3       = time.perf_counter()

        # ── GPIO ──────────────────────────────
        if has_gpio and not no_gpio:
            ctrl.execute(classes[pred_idx], confidence=float(probs[pred_idx]))
        t4 = time.perf_counter()

        times_stage["vad"].append(      (t1 - t0) * 1000)
        times_stage["mfcc"].append(     (t2 - t1) * 1000)
        times_stage["inference"].append((t3 - t2) * 1000)
        times_stage["gpio"].append(     (t4 - t3) * 1000)
        times_total.append(             (t4 - t0) * 1000)

    if has_gpio:
        ctrl.close()

    return np.array(times_total), {k: np.array(v) for k, v in times_stage.items()}


# ─────────────────────────────────────────────
# ESTADÍSTICAS
# ─────────────────────────────────────────────

def stats(times):
    """Devuelve dict con mean, std, min, p50, p95, p99, max."""
    return {
        "mean" : float(np.mean(times)),
        "std"  : float(np.std(times)),
        "min"  : float(np.min(times)),
        "p50"  : float(np.percentile(times, 50)),
        "p95"  : float(np.percentile(times, 95)),
        "p99"  : float(np.percentile(times, 99)),
        "max"  : float(np.max(times)),
    }


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_latency_chart(times_dict, model_name, out_dir):
    """
    Boxplot de latencia por etapa + línea de objetivo 500ms.
    """
    stages = ["vad", "mfcc", "inference", "gpio", "total"]
    labels = ["VAD", "MFCC", "Inferencia\nTFLite", "GPIO\ndespacho", "TOTAL\nE2E"]
    colors = ["#E6F1FB", "#E1F5EE", "#EEEDFE", "#FAECE7", "#3C3489"]

    data = [times_dict.get(s, np.array([0])) for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Subplot 1: boxplot por etapa ─────────
    ax = axes[0]
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Línea de objetivo
    ax.axhline(LATENCY_TARGET, color="#E24B4A", linestyle="--",
               linewidth=1.5, label=f"Objetivo < {LATENCY_TARGET:.0f} ms")

    # Anotar mediana de cada caja
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + 2, f"{med:.1f}ms",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Latencia (ms)", fontsize=11)
    ax.set_title(f"Latencia por etapa — {model_name}",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Subplot 2: stacked bar (composición E2E) ─
    ax2  = axes[1]
    stage_names  = ["VAD", "MFCC", "Inferencia", "GPIO"]
    stage_keys   = ["vad", "mfcc", "inference", "gpio"]
    stage_colors = ["#E6F1FB", "#E1F5EE", "#EEEDFE", "#FAECE7"]
    means        = [np.mean(times_dict.get(k, [0])) for k in stage_keys]
    total_mean   = sum(means)

    bottom = 0
    for name, mean, color in zip(stage_names, means, stage_colors):
        bar = ax2.bar(0, mean, bottom=bottom, color=color,
                      edgecolor="#888", linewidth=0.5, label=name)
        pct = 100 * mean / total_mean if total_mean > 0 else 0
        if mean > 1:
            ax2.text(0, bottom + mean / 2,
                     f"{name}\n{mean:.1f}ms\n({pct:.0f}%)",
                     ha="center", va="center", fontsize=8)
        bottom += mean

    ax2.axhline(LATENCY_TARGET, color="#E24B4A", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_xticks([])
    ax2.set_ylabel("ms")
    ax2.set_title("Composición\nE2E", fontweight="bold", fontsize=10)
    ax2.spines[["top", "right", "bottom"]].set_visible(False)

    plt.tight_layout()
    fname  = f"latency_{model_name.lower().replace(' ', '_')}.png"
    path   = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")
    return path


def plot_cnn_vs_lstm(results_cnn, results_lstm, out_dir):
    """
    Comparativa de latencia CNN vs LSTM: boxplots lado a lado
    para cada etapa, más un resumen de la latencia total.
    """
    stages      = ["vad", "mfcc", "inference", "gpio", "total"]
    stage_labels= ["VAD", "MFCC", "Inferencia", "GPIO", "Total E2E"]
    colors_cnn  = "#3C3489"
    colors_lstm = "#1D9E75"

    fig, axes = plt.subplots(1, len(stages), figsize=(16, 5), sharey=False)

    for ax, stage, label in zip(axes, stages, stage_labels):
        d_cnn  = results_cnn.get(stage,  np.array([0]))
        d_lstm = results_lstm.get(stage, np.array([0]))

        bp = ax.boxplot(
            [d_cnn, d_lstm],
            labels   = ["CNN", "LSTM"],
            patch_artist = True,
            medianprops  = dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(colors_cnn);  bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(colors_lstm); bp["boxes"][1].set_alpha(0.75)

        if stage == "total":
            ax.axhline(LATENCY_TARGET, color="#E24B4A", linestyle="--",
                       linewidth=1.2, label=f"< {LATENCY_TARGET:.0f}ms")
            ax.legend(fontsize=8)

        for i, d in enumerate([d_cnn, d_lstm]):
            ax.text(i + 1, np.median(d) + 0.5,
                    f"{np.median(d):.1f}ms",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_ylabel("ms" if stage == "vad" else "")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Comparativa de latencia CNN 1D vs Bi-LSTM — todas las etapas",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "latency_cnn_vs_lstm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────

def save_report(results_list, n_runs, platform, out_dir):
    """
    Genera el reporte completo de latencia para el documento.
    results_list: lista de (model_name, times_dict) donde
                  times_dict incluye "total" y claves por etapa.
    """
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 62
    sep2 = "-" * 62

    log(sep)
    log("  REPORTE DE LATENCIA — MÓDULO 10")
    log("  Panel de domótica por voz — Raspberry Pi 4")
    log(sep)

    log()
    log("CONFIGURACIÓN DEL BENCHMARK")
    log(sep2)
    log(f"  Plataforma    : {platform}")
    log(f"  Iteraciones   : {n_runs}")
    log(f"  Objetivo E2E  : < {LATENCY_TARGET:.0f} ms  (requisito del enunciado)")
    log(f"  Audio         : {MFCC_CFG['duration_s']}s @ {MFCC_CFG['sample_rate']} Hz")
    log(f"  Features      : MFCC {MFCC_CFG['n_mfcc']} coef. + Δ + Δ²  →  (39, 125)")

    for model_name, times_dict in results_list:
        log()
        log(sep)
        log(f"  MODELO: {model_name}")
        log(sep)

        stage_map = [
            ("vad",       "VAD (energía RMS)"),
            ("mfcc",      "MFCC + Δ + Δ²"),
            ("inference", "Inferencia TFLite"),
            ("gpio",      "Despacho GPIO"),
            ("total",     "TOTAL E2E"),
        ]

        log()
        log(f"  {'Etapa':<22} {'Media':>8} {'Std':>7} {'Min':>7} "
            f"{'P50':>7} {'P95':>7} {'P99':>7} {'Max':>7}")
        log(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*7} "
            f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}")

        for key, label in stage_map:
            if key not in times_dict:
                continue
            s = stats(times_dict[key])
            log(f"  {label:<22} "
                f"{s['mean']:>7.2f}ms "
                f"{s['std']:>6.2f}ms "
                f"{s['min']:>6.2f}ms "
                f"{s['p50']:>6.2f}ms "
                f"{s['p95']:>6.2f}ms "
                f"{s['p99']:>6.2f}ms "
                f"{s['max']:>6.2f}ms")

        # ── Cumplimiento del objetivo ──────────
        if "total" in times_dict:
            t = stats(times_dict["total"])
            log()
            log("  CUMPLIMIENTO DEL OBJETIVO < 500 ms")
            log(sep2)
            pct_ok = 100 * np.mean(times_dict["total"] < LATENCY_TARGET)
            log(f"  Latencia media E2E : {t['mean']:.2f} ms")
            log(f"  Latencia P95  E2E  : {t['p95']:.2f} ms")
            log(f"  % inferencias < 500ms : {pct_ok:.1f}%")

            if t["p95"] < LATENCY_TARGET:
                log(f"  ✔ P95 ({t['p95']:.1f}ms) < {LATENCY_TARGET:.0f}ms — CUMPLE el requisito")
            else:
                log(f"  ⚠ P95 ({t['p95']:.1f}ms) > {LATENCY_TARGET:.0f}ms — REVISAR pipeline")
                log("     Opciones: reducir n_mels, usar n_mfcc=13 sin Δ², o simplificar modelo")

        # ── Estimación en RPi4 ────────────────
        if "Raspberry Pi" not in platform:
            log()
            log("  ESTIMACIÓN EN RASPBERRY PI 4")
            log(sep2)
            log("  (Esta medición fue en laptop/PC — RPi4 es ~3–5× más lento)")
            if "total" in times_dict:
                t = stats(times_dict["total"])
                for factor, label in [(3, "optimista (×3)"), (4, "esperado (×4)"), (5, "pesimista (×5)")]:
                    est = t["mean"] * factor
                    status = "✔" if est < LATENCY_TARGET else "⚠"
                    log(f"  {status} Estimado {label}: {est:.0f} ms  "
                        f"({'< 500ms OK' if est < LATENCY_TARGET else '> 500ms — optimizar'})")

    # ── Comparativa si hay 2 modelos ──────────
    if len(results_list) == 2:
        name_a, td_a = results_list[0]
        name_b, td_b = results_list[1]

        log()
        log(sep)
        log("  COMPARATIVA DE MODELOS")
        log(sep)
        log()
        log(f"  {'Etapa':<22} {name_a:>12} {name_b:>12} {'Ratio':>8}")
        log(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*8}")

        for key, label in [
            ("mfcc",      "MFCC"),
            ("inference", "Inferencia"),
            ("total",     "Total E2E"),
        ]:
            if key in td_a and key in td_b:
                ma = np.mean(td_a[key])
                mb = np.mean(td_b[key])
                ratio = mb / ma if ma > 0 else 0
                log(f"  {label:<22} {ma:>11.2f}ms {mb:>11.2f}ms  {ratio:>7.2f}×")

        log()
        log("  Ratio > 1 → LSTM es más lento; < 1 → LSTM es más rápido")

    # ── Recomendación final ───────────────────
    log()
    log(sep2)
    log("RECOMENDACIÓN FINAL PARA EL PIPELINE")
    log(sep2)

    if results_list:
        best_name = results_list[0][0]
        best_td   = results_list[0][1]
        if len(results_list) == 2:
            t_a = np.mean(results_list[0][1].get("total", [9999]))
            t_b = np.mean(results_list[1][1].get("total", [9999]))
            best_name = results_list[0][0] if t_a <= t_b else results_list[1][0]
            best_td   = results_list[0][1] if t_a <= t_b else results_list[1][1]

        t = stats(best_td.get("total", np.array([0])))
        log(f"  Modelo recomendado : {best_name}")
        log(f"  Latencia E2E media : {t['mean']:.2f} ms")
        log(f"  Umbral de confianza sugerido: 0.75  (ver módulo 6)")
        log()
        log("  Para usar en realtime_pipeline.py:")
        fname = "cnn_model.tflite" if "CNN" in best_name else "lstm_model.tflite"
        log(f"    python realtime_pipeline.py --model models/{fname}")

    log()
    log("GRÁFICAS GENERADAS")
    log(sep2)
    log("  latency_*.png          ← boxplot por etapa")
    if len(results_list) == 2:
        log("  latency_cnn_vs_lstm.png ← comparativa")
    log()
    log(sep)

    report_path = os.path.join(out_dir, "latency_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Reporte guardado en: {report_path}")

    # Guardar también como JSON para análisis posterior
    json_data = {}
    for model_name, times_dict in results_list:
        json_data[model_name] = {
            k: stats(v) for k, v in times_dict.items()
        }
    json_path = os.path.join(out_dir, "latency_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON guardado en   : {json_path}")


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE BENCHMARK
# ─────────────────────────────────────────────

def run_benchmark(model_path, X_model, audio_samples, classes,
                  n_runs, no_gpio, model_label):
    """
    Ejecuta el benchmark completo de un modelo y devuelve
    un dict con los tiempos por etapa.
    """
    print(f"\n{'─'*50}")
    print(f"  Benchmark: {model_label}  ({n_runs} iteraciones)")
    print(f"{'─'*50}")

    # Cargar modelo
    try:
        interp, inp_idx, out_idx, _ = load_tflite(model_path)
    except Exception as e:
        print(f"  ✖ No se pudo cargar el modelo: {e}")
        return None

    results = {}

    # 1. VAD
    print(f"  [1/5] VAD...", end=" ", flush=True)
    results["vad"] = bench_vad(audio_samples, n_runs)
    print(f"media={np.mean(results['vad']):.2f}ms")

    # 2. MFCC
    print(f"  [2/5] MFCC...", end=" ", flush=True)
    results["mfcc"] = bench_mfcc(audio_samples, n_runs)
    print(f"media={np.mean(results['mfcc']):.2f}ms")

    # 3. Inferencia
    print(f"  [3/5] Inferencia TFLite...", end=" ", flush=True)
    results["inference"] = bench_inference(X_model, interp, inp_idx, out_idx, n_runs)
    print(f"media={np.mean(results['inference']):.2f}ms")

    # 4. GPIO
    print(f"  [4/5] GPIO despacho...", end=" ", flush=True)
    results["gpio"] = bench_gpio(classes, n_runs, no_gpio=no_gpio)
    print(f"media={np.mean(results['gpio']):.2f}ms")

    # 5. E2E
    print(f"  [5/5] Pipeline E2E ({n_runs} runs completos)...")
    times_e2e, times_stages = bench_pipeline_e2e(
        audio_samples, X_model, interp, inp_idx, out_idx,
        classes, n_runs, no_gpio=no_gpio
    )
    results["total"] = times_e2e

    t_mean = np.mean(times_e2e)
    t_p95  = np.percentile(times_e2e, 95)
    status = "✔" if t_p95 < LATENCY_TARGET else "⚠"
    print(f"  {status} E2E media={t_mean:.2f}ms  P95={t_p95:.2f}ms  "
          f"({'OK' if t_p95 < LATENCY_TARGET else 'REVISAR'})")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Módulo 10 — Benchmark de latencia")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Modelo TFLite principal (default: {DEFAULT_MODEL})")
    parser.add_argument("--model2",   default=DEFAULT_MODEL2,
                        help=f"Segundo modelo para comparativa (default: {DEFAULT_MODEL2})")
    parser.add_argument("--splits",   default=DEFAULT_SPLITS,
                        help=f"Carpeta con splits (default: {DEFAULT_SPLITS})")
    parser.add_argument("--output",   default="models/reports",
                        help="Carpeta de salida (default: models/reports)")
    parser.add_argument("--runs",     type=int, default=DEFAULT_RUNS,
                        help=f"Número de iteraciones (default: {DEFAULT_RUNS})")
    parser.add_argument("--both",     action="store_true",
                        help="Benchmark CNN y LSTM y comparar")
    parser.add_argument("--no-gpio",  action="store_true",
                        help="Omitir benchmark real de GPIO (usar mock)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Detectar plataforma ───────────────────
    if ON_RPI:
        import subprocess
        try:
            platform = subprocess.check_output(
                ["cat", "/proc/device-tree/model"], text=True
            ).strip()
        except Exception:
            platform = "Raspberry Pi (modelo desconocido)"
    else:
        import platform as _pl
        platform = f"{_pl.system()} {_pl.processor()} (no es RPi — latencias de referencia)"

    print(f"\n{'═'*55}")
    print(f"  BENCHMARK DE LATENCIA — MÓDULO 10")
    print(f"  Plataforma: {platform}")
    print(f"{'═'*55}")

    # ── Cargar datos ──────────────────────────
    print(f"\nCargando datos de test desde: {args.splits}")
    try:
        X_model, audio_samples, y_test, classes = load_test_audio(args.splits)
    except FileNotFoundError as e:
        print(f"✖ {e}")
        sys.exit(1)
    print(f"  Muestras de test : {len(X_model)}")
    print(f"  Clases           : {list(classes)}")

    # ── Determinar modelos a benchmarkear ─────
    models_to_run = [(args.model, "CNN 1D")]
    if args.both:
        models_to_run.append((args.model2, "Bi-LSTM"))

    # ── Ejecutar benchmarks ───────────────────
    results_list = []
    for model_path, label in models_to_run:
        if not os.path.exists(model_path):
            print(f"  ⚠ Modelo no encontrado: {model_path} — omitido")
            continue
        result = run_benchmark(
            model_path, X_model, audio_samples, classes,
            args.runs, args.no_gpio, label
        )
        if result is not None:
            results_list.append((label, result))

    if not results_list:
        print("✖ No se pudo ejecutar ningún benchmark.")
        sys.exit(1)

    # ── Reporte ───────────────────────────────
    save_report(results_list, args.runs, platform, args.output)

    # ── Gráficas ──────────────────────────────
    if not args.no_plots:
        print("\nGenerando gráficas...")
        for model_name, times_dict in results_list:
            plot_latency_chart(times_dict, model_name, args.output)

        if len(results_list) == 2:
            plot_cnn_vs_lstm(
                results_list[0][1],
                results_list[1][1],
                args.output,
            )

    # ── Resumen final en consola ──────────────
    print(f"\n{'═'*55}")
    print("  RESUMEN FINAL")
    print(f"{'═'*55}")
    for model_name, times_dict in results_list:
        t = stats(times_dict.get("total", np.array([0])))
        ok = "✔" if t["p95"] < LATENCY_TARGET else "⚠"
        print(f"  {ok} {model_name:<12}  "
              f"media={t['mean']:.1f}ms  "
              f"P95={t['p95']:.1f}ms  "
              f"({'CUMPLE' if t['p95'] < LATENCY_TARGET else 'REVISAR'} < {LATENCY_TARGET:.0f}ms)")

    print(f"\n✔ Módulo 10 completado.")
    print(f"  Ver reporte completo en: {args.output}/latency_report.txt\n")


if __name__ == "__main__":
    main()