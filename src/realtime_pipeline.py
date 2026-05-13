"""
MÓDULO 9 — Pipeline de inferencia en tiempo real
Panel de domótica por voz — Raspberry Pi 4

Flujo completo:
    Micrófono USB
    → Captura continua (sounddevice, 16 kHz)
    → VAD por energía (NumPy puro — sin librerías externas)
    → Buffer de 2 segundos
    → Extracción MFCC (librosa)
    → Inferencia TFLite
    → Umbral de confianza
    → Despacho a ActuatorController (GPIO)

Uso:
    python realtime_pipeline.py
    python realtime_pipeline.py --model models/cnn_model.tflite
    python realtime_pipeline.py --model models/lstm_model.tflite
    python realtime_pipeline.py --threshold 0.75 --verbose

Requisitos en RPi4:
    pip install sounddevice librosa numpy tensorflow
    (gpiozero ya viene en Raspberry Pi OS)
"""

import os
import sys
import json
import pickle
import time
import queue
import threading
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN — debe coincidir con módulo 2
# ─────────────────────────────────────────────

# Ruta por defecto al modelo y splits
DEFAULT_MODEL    = "models/cnn_model.tflite"
DEFAULT_SPLITS   = "splits"
DEFAULT_THRESHOLD= 0.75      # confianza mínima para actuar
DEFAULT_DEVICE   = None      # None = dispositivo por defecto del sistema

# Parámetros de audio
SAMPLE_RATE      = 16000
DURATION_S       = 2.0
N_SAMPLES        = int(SAMPLE_RATE * DURATION_S)   # 32 000 muestras

# MFCC — igual que feature_extraction.py (módulo 2)
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

# VAD por energía
VAD_ENERGY_THRESHOLD = 0.01   # RMS mínimo para considerar que hay voz
                               # Ajustar según el ruido ambiente del lab
VAD_MIN_VOICE_FRAMES = 3      # mínimo de frames consecutivos con voz
                               # antes de capturar el buffer completo

# ─────────────────────────────────────────────
# VAD POR ENERGÍA — implementado con NumPy puro
# No usa ninguna librería externa, cumple con
# las restricciones del enunciado y puede
# explicarse matemáticamente en la defensa.
# ─────────────────────────────────────────────

class EnergyVAD:
    """
    Voice Activity Detector basado en energía RMS por frames.

    Algoritmo:
      1. Divide la señal en frames de frame_ms milisegundos.
      2. Calcula el RMS de cada frame:
             RMS = sqrt(mean(x²))
      3. Un frame se clasifica como "voz" si RMS > threshold.
      4. Se activa cuando al menos min_voice_frames consecutivos
         tienen voz (evita falsos positivos por clics o ruidos breves).
      5. Una vez activado, captura el buffer completo de 2 segundos
         para procesar con el modelo.

    Parámetros:
        threshold       : RMS mínimo para voz (float en escala [-1, 1])
        frame_ms        : duración de cada frame en ms
        min_voice_frames: frames consecutivos con voz para activar
        sample_rate     : frecuencia de muestreo
    """

    def __init__(self, threshold=VAD_ENERGY_THRESHOLD,
                 frame_ms=20, min_voice_frames=VAD_MIN_VOICE_FRAMES,
                 sample_rate=SAMPLE_RATE):
        self.threshold        = threshold
        self.frame_size       = int(sample_rate * frame_ms / 1000)
        self.min_voice_frames = min_voice_frames
        self._voice_count     = 0

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Recibe un chunk de audio float32 y devuelve True si
        contiene voz según el criterio de energía RMS.
        """
        # Dividir en frames y calcular RMS de cada uno
        n_frames  = len(audio_chunk) // self.frame_size
        if n_frames == 0:
            return False

        frames     = audio_chunk[:n_frames * self.frame_size]
        frames     = frames.reshape(n_frames, self.frame_size)
        rms_values = np.sqrt(np.mean(frames ** 2, axis=1))

        # Contar frames consecutivos con voz
        voice_frames = np.sum(rms_values > self.threshold)

        if voice_frames >= self.min_voice_frames:
            self._voice_count += 1
        else:
            self._voice_count = 0

        return self._voice_count >= 1

    def reset(self):
        self._voice_count = 0

    def calibrate(self, noise_sample: np.ndarray, margin=3.0):
        """
        Calibra el umbral automáticamente a partir de una muestra
        de ruido ambiente.
        threshold = mean(RMS) + margin * std(RMS)
        Llamar al inicio del pipeline con 1-2 segundos de silencio.
        """
        n_frames   = len(noise_sample) // self.frame_size
        if n_frames == 0:
            return
        frames     = noise_sample[:n_frames * self.frame_size]
        frames     = frames.reshape(n_frames, self.frame_size)
        rms_values = np.sqrt(np.mean(frames ** 2, axis=1))
        self.threshold = float(np.mean(rms_values) + margin * np.std(rms_values))
        print(f"  [VAD] Umbral calibrado: {self.threshold:.5f}")


# ─────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES — igual que módulo 2
# ─────────────────────────────────────────────

def extract_features(audio: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Extrae MFCC + Delta + Delta² de un array float32.
    Devuelve tensor (1, T, C) listo para TFLite.

    La dimensión extra al inicio es el batch=1 que TFLite necesita.
    El orden (T, C) = (125, 39) coincide con cómo se entrenó el modelo.
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

    # (39, 125) → (125, 39) → (1, 125, 39)  batch=1
    feat = feat.T[np.newaxis, :, :].astype(np.float32)
    return feat


# ─────────────────────────────────────────────
# INFERENCIA TFLITE
# ─────────────────────────────────────────────

class TFLiteClassifier:
    """
    Wrapper liviano sobre TFLite Interpreter.
    Carga el modelo una sola vez y reutiliza el intérprete
    para todas las predicciones (evita overhead de carga).
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo TFLite no encontrado: {model_path}\n"
                "Ejecuta train_cnn.py (módulo 5) para generarlo."
            )

        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=model_path)
        except ImportError:
            # Intentar con tflite_runtime (más liviano, recomendado en RPi)
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=model_path)
            except ImportError:
                raise ImportError(
                    "Instala TensorFlow o tflite_runtime:\n"
                    "  pip install tensorflow\n"
                    "  # o en RPi (más liviano):\n"
                    "  pip install tflite-runtime"
                )

        self._interpreter.allocate_tensors()
        self._inp_idx = self._interpreter.get_input_details()[0]["index"]
        self._out_idx = self._interpreter.get_output_details()[0]["index"]

        inp_shape = self._interpreter.get_input_details()[0]["shape"]
        out_shape = self._interpreter.get_output_details()[0]["shape"]
        print(f"  [TFLite] Modelo cargado: {model_path}")
        print(f"           Input  shape : {inp_shape}")
        print(f"           Output shape : {out_shape}")
        print(f"           Tamaño       : {os.path.getsize(model_path)/1024:.1f} KB")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Corre inferencia sobre features shape (1, T, C).
        Devuelve array de probabilidades shape (n_classes,).
        """
        self._interpreter.set_tensor(self._inp_idx, features)
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(self._out_idx)[0]
        return probs


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

class RealtimePipeline:
    """
    Coordina captura de audio, VAD, extracción de features,
    inferencia y despacho a GPIO en un pipeline de baja latencia.

    Arquitectura de hilos:
        Hilo principal   → sounddevice callback (alta prioridad)
        audio_queue      → buffer de chunks de audio entre hilos
        Hilo de proceso  → VAD + MFCC + TFLite + GPIO
                           (corre en paralelo al audio capture)

    Esta separación garantiza que el callback de audio nunca
    se bloquee esperando la inferencia, evitando dropouts de audio.
    """

    def __init__(self, model_path, splits_dir, threshold=DEFAULT_THRESHOLD,
                 device=None, verbose=False):

        self.threshold = threshold
        self.device    = device
        self.verbose   = verbose
        self._running  = False
        self._audio_q  = queue.Queue(maxsize=20)

        # ── Cargar label encoder ──────────────
        le_path = os.path.join(splits_dir, "label_encoder.pkl")
        if not os.path.exists(le_path):
            raise FileNotFoundError(
                f"label_encoder.pkl no encontrado en {splits_dir}\n"
                "Asegúrate de haber ejecutado prepare_splits.py (módulo 4)."
            )
        with open(le_path, "rb") as f:
            self._le = pickle.load(f)
        self._classes = self._le.classes_
        print(f"  [Pipeline] Clases: {list(self._classes)}")

        # ── Cargar feature_config si existe ───
        cfg_path = os.path.join(splits_dir, "feature_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                loaded = json.load(f)
            MFCC_CFG.update(loaded)
            print(f"  [Pipeline] feature_config.json cargado desde {cfg_path}")

        # ── Inicializar componentes ───────────
        self._vad        = EnergyVAD()
        self._classifier = TFLiteClassifier(model_path)

        # Importar aquí para que el error sea claro si no está instalado
        from gpio_controller import ActuatorController
        self._actuator = ActuatorController()

        # ── Buffer de audio deslizante ─────────
        # Mantiene los últimos N_SAMPLES en un ring buffer.
        # Cuando el VAD detecta voz, toma el buffer completo.
        self._ring_buffer = np.zeros(N_SAMPLES, dtype=np.float32)
        self._ring_idx    = 0

        # ── Estadísticas en vivo ───────────────
        self._stats = {
            "total_detections"  : 0,
            "accepted"          : 0,
            "rejected_threshold": 0,
            "latencies_ms"      : [],
        }

    # ─────────────────────────────────────────
    # CALLBACK DE AUDIO (hilo de sounddevice)
    # ─────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Llamado por sounddevice cada vez que hay un nuevo chunk de audio.
        DEBE ser rápido — no hacer operaciones costosas aquí.
        Solo normaliza y encola el chunk.
        """
        if status and self.verbose:
            print(f"  [Audio] {status}")

        # Normalizar int16 → float32 [-1, 1]
        chunk = indata[:, 0].astype(np.float32) / 32768.0

        # Actualizar ring buffer deslizante
        chunk_len = len(chunk)
        if chunk_len >= N_SAMPLES:
            self._ring_buffer = chunk[-N_SAMPLES:].copy()
        else:
            self._ring_buffer = np.roll(self._ring_buffer, -chunk_len)
            self._ring_buffer[-chunk_len:] = chunk

        # Encolar para el hilo de procesamiento
        try:
            self._audio_q.put_nowait(chunk.copy())
        except queue.Full:
            pass   # descartar si el hilo de proceso va lento

    # ─────────────────────────────────────────
    # HILO DE PROCESAMIENTO
    # ─────────────────────────────────────────

    def _process_loop(self):
        """
        Hilo de procesamiento:
        Consume chunks de la cola, aplica VAD, y cuando detecta
        voz captura el buffer completo para inferencia.
        """
        cooldown_until = 0.0    # evita procesar el mismo evento dos veces

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.perf_counter()

            # ── VAD ───────────────────────────
            if not self._vad.is_speech(chunk):
                continue

            # ── Cooldown ──────────────────────
            # Tras cada predicción, esperar al menos DURATION_S
            # antes de aceptar otra activación (evita procesar
            # el eco del comando o predicciones duplicadas).
            if now < cooldown_until:
                continue

            # ── Capturar buffer completo ───────
            t0    = time.perf_counter()
            audio = self._ring_buffer.copy()   # 2 segundos

            # ── Extraer features ───────────────
            t1 = time.perf_counter()
            try:
                features = extract_features(audio, MFCC_CFG)
            except Exception as e:
                print(f"  [MFCC] Error: {e}")
                continue

            # ── Inferencia TFLite ──────────────
            t2    = time.perf_counter()
            probs = self._classifier.predict(features)
            t3    = time.perf_counter()

            pred_idx    = int(np.argmax(probs))
            confidence  = float(probs[pred_idx])
            command     = self._classes[pred_idx]

            # ── Latencia total ─────────────────
            latency_ms = (t3 - t0) * 1000
            self._stats["total_detections"] += 1
            self._stats["latencies_ms"].append(latency_ms)

            # ── Log siempre ───────────────────
            bar    = "█" * int(confidence * 20)
            status = "✔" if confidence >= self.threshold else "✖"
            print(
                f"\n[{status}] '{command.upper():<14}' "
                f"conf={confidence:.3f} {bar:<20} "
                f"lat={latency_ms:.1f}ms"
            )

            if self.verbose:
                t_cap  = (t1 - t0) * 1000
                t_mfcc = (t2 - t1) * 1000
                t_inf  = (t3 - t2) * 1000
                print(
                    f"    captura={t_cap:.1f}ms  "
                    f"mfcc={t_mfcc:.1f}ms  "
                    f"inferencia={t_inf:.1f}ms"
                )
                print(f"    Probabilidades: "
                      + "  ".join(
                          f"{c}={p:.2f}"
                          for c, p in zip(self._classes, probs)
                      ))

            # ── Umbral de confianza ────────────
            if confidence < self.threshold:
                self._stats["rejected_threshold"] += 1
                print(f"    → Rechazado (confianza < {self.threshold})")
                cooldown_until = now + 0.5   # cooldown corto al rechazar
                continue

            # ── Despacho a GPIO ────────────────
            t4 = time.perf_counter()
            self._actuator.execute(command, confidence=confidence)
            t5 = time.perf_counter()

            gpio_ms = (t5 - t4) * 1000
            total_ms = (t5 - t0) * 1000

            if self.verbose:
                print(f"    gpio={gpio_ms:.1f}ms  |  TOTAL={total_ms:.1f}ms")

            if total_ms > 500:
                print(f"    ⚠ Latencia total {total_ms:.1f}ms > 500ms")

            self._stats["accepted"] += 1

            # Cooldown: no procesar durante el siguiente DURATION_S
            cooldown_until = now + DURATION_S
            self._vad.reset()

    # ─────────────────────────────────────────
    # CALIBRACIÓN DE VAD
    # ─────────────────────────────────────────

    def _calibrate_vad(self, duration_s=2.0):
        """
        Graba 'duration_s' segundos de ruido ambiente para
        calibrar el umbral del VAD automáticamente.
        """
        import sounddevice as sd

        print(f"\n  Calibrando VAD — silencio por {duration_s:.0f} segundos...")
        print("  (no hables durante la calibración)\n")
        time.sleep(0.5)

        noise = sd.rec(
            int(duration_s * SAMPLE_RATE),
            samplerate = SAMPLE_RATE,
            channels   = 1,
            dtype      = np.int16,
            device     = self.device,
        )
        sd.wait()
        noise_float = noise[:, 0].astype(np.float32) / 32768.0
        self._vad.calibrate(noise_float)

    # ─────────────────────────────────────────
    # INICIO Y PARADA
    # ─────────────────────────────────────────

    def run(self, calibrate=True):
        """
        Inicia el pipeline de inferencia en tiempo real.
        Bloquea hasta que el usuario presione Ctrl+C.
        """
        import sounddevice as sd

        # Mostrar dispositivos disponibles
        if self.verbose:
            print("\nDispositivos de audio disponibles:")
            print(sd.query_devices())

        # Calibrar VAD
        if calibrate:
            self._calibrate_vad()

        print("\n" + "─" * 55)
        print("  PANEL DE DOMÓTICA — ESCUCHANDO")
        print(f"  Modelo    : {DEFAULT_MODEL}")
        print(f"  Umbral    : {self.threshold}")
        print(f"  Comandos  : {list(self._classes)}")
        print("  Presiona Ctrl+C para detener")
        print("─" * 55 + "\n")

        # Iniciar hilo de procesamiento
        self._running       = True
        self._process_thread = threading.Thread(
            target   = self._process_loop,
            daemon   = True,
            name     = "ProcessThread",
        )
        self._process_thread.start()

        # Iniciar captura de audio
        chunk_frames = int(SAMPLE_RATE * 0.1)   # chunks de 100ms

        try:
            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                channels   = 1,
                dtype      = np.int16,
                blocksize  = chunk_frames,
                device     = self.device,
                callback   = self._audio_callback,
            ):
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n[!] Deteniendo pipeline...")
        finally:
            self._running = False
            self._process_thread.join(timeout=2.0)
            self._actuator.close()
            self._print_stats()

    def _print_stats(self):
        """Imprime estadísticas de la sesión al terminar."""
        s = self._stats
        lat = s["latencies_ms"]

        print("\n" + "=" * 55)
        print("  ESTADÍSTICAS DE LA SESIÓN")
        print("─" * 55)
        print(f"  Detecciones totales     : {s['total_detections']}")
        print(f"  Comandos aceptados      : {s['accepted']}")
        print(f"  Rechazados (umbral)     : {s['rejected_threshold']}")

        if lat:
            print(f"  Latencia promedio       : {np.mean(lat):.1f} ms")
            print(f"  Latencia mínima         : {np.min(lat):.1f} ms")
            print(f"  Latencia máxima         : {np.max(lat):.1f} ms")
            print(f"  Latencia P95            : {np.percentile(lat, 95):.1f} ms")

        print("=" * 55 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Módulo 9 — Pipeline de inferencia en tiempo real"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ruta al modelo .tflite (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--splits", default=DEFAULT_SPLITS,
        help=f"Carpeta con label_encoder.pkl (default: {DEFAULT_SPLITS})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Umbral de confianza mínimo (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Índice del dispositivo de audio (default: dispositivo del sistema)"
    )
    parser.add_argument(
        "--no-calibrate", action="store_true",
        help="Omitir calibración de VAD al inicio"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostrar desglose de latencia por etapa y todas las probabilidades"
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=VAD_ENERGY_THRESHOLD,
        help=f"Umbral de energía VAD manual (default: {VAD_ENERGY_THRESHOLD})"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Listar dispositivos de audio disponibles y salir"
    )
    args = parser.parse_args()

    # ── Listar dispositivos ───────────────────
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    # ── Verificar archivos necesarios ─────────
    if not os.path.exists(args.model):
        print(f"✖ Modelo no encontrado: {args.model}")
        print("  Ejecuta train_cnn.py (módulo 5) para generarlo.")
        sys.exit(1)

    le_path = os.path.join(args.splits, "label_encoder.pkl")
    if not os.path.exists(le_path):
        print(f"✖ label_encoder.pkl no encontrado en: {args.splits}")
        print("  Ejecuta prepare_splits.py (módulo 4) primero.")
        sys.exit(1)

    print("\n" + "═" * 55)
    print("  PANEL DE DOMÓTICA POR VOZ")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 55)
    print(f"  Modelo    : {args.model}")
    print(f"  Umbral    : {args.threshold}")
    print(f"  VAD       : energía RMS (NumPy puro)")

    # ── Inicializar y correr pipeline ─────────
    try:
        pipeline = RealtimePipeline(
            model_path = args.model,
            splits_dir = args.splits,
            threshold  = args.threshold,
            device     = args.device,
            verbose    = args.verbose,
        )

        # Ajuste manual del umbral VAD si se especificó
        if args.vad_threshold != VAD_ENERGY_THRESHOLD:
            pipeline._vad.threshold = args.vad_threshold
            print(f"  VAD umbral: {args.vad_threshold} (manual)")

        pipeline.run(calibrate=not args.no_calibrate)

    except Exception as e:
        print(f"\n✖ Error al iniciar el pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()