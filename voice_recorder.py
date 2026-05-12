import os
import sounddevice as sd
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from datetime import datetime

# =========================
# CONFIGURACIÓN
# =========================

SAMPLE_RATE = 16000
DURATION = 2  # segundos
OUTPUT_FOLDER = "dataset"

COMMANDS = [
    "enciende",
    "apaga",
    "ventilador",
    "abrir",
    "cerrar",
    "musica",
    "detente",
    "ruido_fondo"
]

SAMPLES_PER_COMMAND = 30

person = input("Nombre del participante: ").strip().lower()
environment = input("Ambiente (silencioso/ruido/lab/etc): ").strip().lower()

# =========================
# CREAR ESTRUCTURA
# =========================

for cmd in COMMANDS:
    os.makedirs(os.path.join(OUTPUT_FOLDER, cmd), exist_ok=True)

metadata = []

print("\n=== INICIO DE GRABACIÓN DE DATASET ===\n")

# =========================
# GRABACIÓN CONTROLADA
# =========================

for cmd in COMMANDS:
    print(f"\n>>> COMANDO: {cmd.upper()} <<<\n")

    for i in range(SAMPLES_PER_COMMAND):

        input(f"[{i+1}/{SAMPLES_PER_COMMAND}] Presiona ENTER para grabar...")

        print("Grabando...")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )

        sd.wait()

        print("OK")

        filename = f"{cmd}_{person}_{i+1:03}.wav"

        path = os.path.join(OUTPUT_FOLDER, cmd, filename)

        write(path, SAMPLE_RATE, audio)

        metadata.append({
            "file": filename,
            "command": cmd,
            "person": person,
            "environment": environment,
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"Guardado: {path}\n")

print("\n=== GRABACIÓN FINALIZADA ===")

# =========================
# GUARDAR METADATA
# =========================

df = pd.DataFrame(metadata)
csv_path = os.path.join(OUTPUT_FOLDER, "metadata.csv")
df.to_csv(csv_path, index=False)

print(f"Metadata guardada en: {csv_path}")