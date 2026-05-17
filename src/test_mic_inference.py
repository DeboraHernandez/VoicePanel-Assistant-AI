"""
test_mic_inference.py — Test rápido de inferencia por micrófono

Graba 2 segundos de audio, extrae MFCC y corre inferencia.
Soporta modelo Keras (.keras) o SavedModel (directorio).

Uso:
    # Con SavedModel (por defecto):
    python test_mic_inference.py

    # Especificar modelo explícitamente:
    python test_mic_inference.py --model models/cnn_savedmodel
    python test_mic_inference.py --model models/cnn_model.keras

    # Opciones extra:
    python test_mic_inference.py --splits splits --threshold 0.6
    python test_mic_inference.py --list-devices
    python test_mic_inference.py --device 2

Requisitos:
    pip install tensorflow librosa numpy sounddevice
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN — debe coincidir con módulo 2
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

SAMPLE_RATE = MFCC_CFG["sample_rate"]
DURATION_S  = MFCC_CFG["duration_s"]
N_SAMPLES   = int(SAMPLE_RATE * DURATION_S)   # 32 000 muestras

DEFAULT_MODEL     = "models/cnn_savedmodel"
DEFAULT_SPLITS    = "splits"
DEFAULT_THRESHOLD = 0.70


# ─────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES
# ─────────────────────────────────────────────

def extract_features(audio: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Extrae MFCC + Delta + Delta² del audio grabado.
    Devuelve tensor (1, 125, 39) float32 listo para inferencia.
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y          = audio,
        sr         = cfg["sample_rate"],
        n_mfcc     = cfg["n_mfcc"],
        n_fft      = cfg["n_fft"],
        hop_length = cfg["hop_length"],
        win_length = cfg["win_length"],
        n_mels     = cfg["n_mels"],
    )                                        # (13, T)

    features = [mfcc]
    if cfg["use_delta"]:
        features.append(librosa.feature.delta(mfcc, order=1))
    if cfg["use_delta2"]:
        features.append(librosa.feature.delta(mfcc, order=2))

    feat = np.concatenate(features, axis=0)  # (39, T)

    # Padding / truncado a n_frames fijo
    T = feat.shape[1]
    if T < cfg["n_frames"]:
        feat = np.pad(feat, ((0, 0), (0, cfg["n_frames"] - T)), mode="constant")
    else:
        feat = feat[:, :cfg["n_frames"]]

    # Normalización por muestra (media 0, std 1)
    if cfg["normalize"]:
        mean = feat.mean(axis=1, keepdims=True)
        std  = feat.std(axis=1, keepdims=True) + 1e-8
        feat = (feat - mean) / std

    # (39, 125) → (125, 39) → (1, 125, 39)
    return feat.T[np.newaxis, :, :].astype(np.float32)


# ─────────────────────────────────────────────
# CARGA DE MODELO
# ─────────────────────────────────────────────

def load_model(model_path: str):
    """
    Detecta automáticamente si es SavedModel (directorio) o Keras (.keras/.h5)
    y devuelve una función predict(features) → np.ndarray con probabilidades.
    """
    import tensorflow as tf

    # ── SavedModel ────────────────────────────
    if os.path.isdir(model_path):
        print(f"  Tipo de modelo  : SavedModel")
        loaded    = tf.saved_model.load(model_path)
        infer_fn  = loaded.signatures["serving_default"]
        input_key = list(infer_fn.structured_input_signature[1].keys())[0]
        out_key   = list(infer_fn.structured_outputs.keys())[0]

        total_kb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(model_path)
            for f in files
        ) / 1024
        print(f"  Tamaño          : {total_kb:.1f} KB")
        print(f"  Input key       : {input_key}")

        def predict(features: np.ndarray) -> np.ndarray:
            tensor = tf.constant(features)
            result = infer_fn(**{input_key: tensor})
            return result[out_key].numpy()[0]

        return predict

    # ── Keras (.keras / .h5) ──────────────────
    elif os.path.isfile(model_path):
        print(f"  Tipo de modelo  : Keras")
        model   = tf.keras.models.load_model(model_path)
        size_kb = os.path.getsize(model_path) / 1024
        print(f"  Tamaño          : {size_kb:.1f} KB")
        print(f"  Input shape     : {model.input_shape}")

        def predict(features: np.ndarray) -> np.ndarray:
            return model.predict(features, verbose=0)[0]

        return predict

    else:
        raise FileNotFoundError(
            f"Modelo no encontrado: '{model_path}'\n"
            "  SavedModel → debe ser un directorio (ej: models/cnn_savedmodel)\n"
            "  Keras      → debe ser un archivo   (ej: models/cnn_model.keras)"
        )


# ─────────────────────────────────────────────
# GRABACIÓN DE AUDIO
# ─────────────────────────────────────────────

def record_audio(duration_s: float, sample_rate: int, device=None) -> np.ndarray:
    """
    Graba 'duration_s' segundos con el micrófono y devuelve
    array float32 normalizado en [-1, 1].
    """
    import sounddevice as sd

    print(f"\n  🎙  Grabando {duration_s:.0f} segundos... (habla ahora)")

    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate = sample_rate,
        channels   = 1,
        dtype      = np.int16,
        device     = device,
    )
    sd.wait()
    print("  ✓  Grabación terminada.")
    return audio[:, 0].astype(np.float32) / 32768.0


# ─────────────────────────────────────────────
# VISUALIZACIÓN DE RESULTADOS
# ─────────────────────────────────────────────

def print_results(probs: np.ndarray, classes, threshold: float):
    """Imprime una barra de probabilidades por clase."""
    print()
    print("  ─" * 28)
    print("  RESULTADOS")
    print("  ─" * 28)

    sorted_idx = np.argsort(probs)[::-1]

    for i, idx in enumerate(sorted_idx):
        cls        = classes[idx]
        prob       = probs[idx]
        bar_len    = int(prob * 30)
        bar        = "█" * bar_len + "░" * (30 - bar_len)
        marker     = " ✔" if (i == 0 and prob >= threshold) else ("  " if i > 0 else " ✖")
        print(f"  {marker} {cls:<14} {bar}  {prob:.3f}")

    print("  ─" * 28)
    top_idx    = int(np.argmax(probs))
    top_class  = classes[top_idx]
    top_conf   = float(probs[top_idx])

    if top_conf >= threshold:
        print(f"\n  → COMANDO: '{top_class.upper()}' ({top_conf:.1%} confianza)")
    else:
        print(f"\n  → RECHAZADO: '{top_class}' ({top_conf:.1%} < umbral {threshold:.0%})")
    print()


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def run_test(predict_fn, classes, threshold, device, n_rounds):
    """Loop interactivo: graba → infiere → muestra resultado → repite."""

    round_num = 0

    while True:
        round_num += 1

        if n_rounds and round_num > n_rounds:
            print("\n  Número máximo de rondas alcanzado.")
            break

        print(f"\n{'═' * 55}")
        print(f"  RONDA {round_num}" + (f" / {n_rounds}" if n_rounds else ""))
        print(f"{'═' * 55}")
        print(f"  Comandos disponibles: {list(classes)}")
        print(f"  Umbral de confianza : {threshold}")
        print()
        print("  Presiona Enter para grabar  |  'q' + Enter para salir")

        try:
            user_input = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Saliendo...")
            break

        if user_input == "q":
            print("\n  Saliendo...")
            break

        # ── Grabación ─────────────────────────
        try:
            audio = record_audio(DURATION_S, SAMPLE_RATE, device)
        except Exception as e:
            print(f"\n  ✖ Error al grabar: {e}")
            print("  Verifica que el micrófono esté conectado.")
            print("  Usa --list-devices para ver los dispositivos disponibles.")
            continue

        # ── Energía de la señal grabada ────────
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"  Energía RMS     : {rms:.5f}", end="")
        if rms < 0.005:
            print("  ⚠ Señal muy baja — ¿micrófono conectado?")
        else:
            print()

        # ── Extracción de features ─────────────
        t0 = time.perf_counter()
        try:
            features = extract_features(audio, MFCC_CFG)
        except Exception as e:
            print(f"\n  ✖ Error en extracción MFCC: {e}")
            continue
        t_mfcc = (time.perf_counter() - t0) * 1000

        # ── Inferencia ─────────────────────────
        t1 = time.perf_counter()
        try:
            probs = predict_fn(features)
        except Exception as e:
            print(f"\n  ✖ Error en inferencia: {e}")
            continue
        t_inf = (time.perf_counter() - t1) * 1000

        print(f"  MFCC            : {t_mfcc:.1f} ms")
        print(f"  Inferencia      : {t_inf:.1f} ms")
        print(f"  Total           : {t_mfcc + t_inf:.1f} ms")

        # ── Resultados ─────────────────────────
        print_results(probs, classes, threshold)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test de inferencia por micrófono — CNN voz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python test_mic_inference.py
  python test_mic_inference.py --model models/cnn_model.keras
  python test_mic_inference.py --model models/cnn_savedmodel --threshold 0.65
  python test_mic_inference.py --device 2 --rounds 5
  python test_mic_inference.py --list-devices
        """
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ruta al SavedModel (directorio) o .keras/.h5 (archivo). "
             f"Default: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--splits", default=DEFAULT_SPLITS,
        help=f"Carpeta con label_encoder.pkl. Default: {DEFAULT_SPLITS}"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Confianza mínima para aceptar un comando. Default: {DEFAULT_THRESHOLD}"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Índice del dispositivo de audio (default: dispositivo del sistema)"
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Número de rondas a ejecutar (default: infinito hasta 'q')"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Listar dispositivos de audio disponibles y salir"
    )
    args = parser.parse_args()

    # ── Listar dispositivos ───────────────────
    if args.list_devices:
        import sounddevice as sd
        print("\nDispositivos de audio disponibles:")
        print(sd.query_devices())
        return

    # ── Header ────────────────────────────────
    print()
    print("═" * 55)
    print("  TEST DE INFERENCIA — PANEL DOMÓTICO POR VOZ")
    print("═" * 55)

    # ── Verificar TensorFlow ──────────────────
    try:
        import tensorflow as tf
        print(f"  TensorFlow      : {tf.__version__}")
    except ImportError:
        print("✖ TensorFlow no está instalado.")
        print("  pip install tensorflow")
        sys.exit(1)

    # ── Cargar label encoder ──────────────────
    le_path = os.path.join(args.splits, "label_encoder.pkl")
    if not os.path.exists(le_path):
        print(f"✖ label_encoder.pkl no encontrado en: {args.splits}")
        print("  Ejecuta prepare_splits.py (módulo 4) primero.")
        sys.exit(1)

    with open(le_path, "rb") as f:
        le = pickle.load(f)
    classes = le.classes_
    print(f"  Clases          : {list(classes)}")

    # Cargar feature_config si existe (sobreescribe defaults)
    cfg_path = os.path.join(args.splits, "feature_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            MFCC_CFG.update(json.load(f))
        print(f"  feature_config  : cargado desde {cfg_path}")

    # ── Cargar modelo ─────────────────────────
    print(f"\n  Cargando modelo : {args.model}")
    try:
        predict_fn = load_model(args.model)
    except FileNotFoundError as e:
        print(f"\n✖ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✖ Error al cargar el modelo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Verificar sounddevice ─────────────────
    try:
        import sounddevice as sd
        device_info = sd.query_devices(args.device, "input")
        print(f"\n  Micrófono       : {device_info['name']}")
        print(f"  Sample rate     : {SAMPLE_RATE} Hz")
        print(f"  Duración buffer : {DURATION_S:.1f} s")
    except Exception as e:
        print(f"\n✖ Error con sounddevice: {e}")
        print("  pip install sounddevice")
        sys.exit(1)

    # ── Comenzar loop de test ─────────────────
    run_test(
        predict_fn = predict_fn,
        classes    = classes,
        threshold  = args.threshold,
        device     = args.device,
        n_rounds   = args.rounds,
    )

    print("  Test finalizado.\n")


if __name__ == "__main__":
    main()
