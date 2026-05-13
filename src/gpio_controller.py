"""
MÓDULO 8 — Control GPIO de actuadores
Panel de domótica por voz — Raspberry Pi 4

Clase ActuatorController con métodos para cada actuador:
    LED       → GPIO 17  (digital ON/OFF + parpadeo)
    Ventilador→ GPIO 18  (PWM 0.0–1.0 vía transistor 2N2222)
    Servo SG90→ GPIO 12  (ángulo -90° a +90°, PWM hardware)
    Buzzer    → GPIO 13  (PWM hardware, melodías por frecuencia)

Uso como módulo importado (desde realtime_pipeline.py):
    from gpio_controller import ActuatorController
    ctrl = ActuatorController()
    ctrl.execute("enciende")

Uso standalone para probar cada actuador:
    python gpio_controller.py
    python gpio_controller.py --test led
    python gpio_controller.py --test ventilador
    python gpio_controller.py --test servo
    python gpio_controller.py --test buzzer
    python gpio_controller.py --test all
    python gpio_controller.py --demo          ← demo completa secuencial

IMPORTANTE: Este módulo debe ejecutarse en la Raspberry Pi 4.
En laptop/PC muestra un modo simulado (mock) para desarrollo.
"""

import time
import threading
import sys
import argparse

# ─────────────────────────────────────────────
# DETECCIÓN DE PLATAFORMA
# Permite desarrollar y probar lógica en laptop
# antes de copiar a la Raspberry Pi.
# ─────────────────────────────────────────────

def _is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False

ON_RPI = _is_raspberry_pi()

if ON_RPI:
    from gpiozero import LED, PWMOutputDevice, AngularServo
    from gpiozero.tones import Tone
    import gpiozero
    print("[gpio_controller] Raspberry Pi detectada — GPIO real activo.")
else:
    print("[gpio_controller] No es Raspberry Pi — modo simulado (mock).")

# ─────────────────────────────────────────────
# PINES GPIO (BCM)
# ─────────────────────────────────────────────

PIN_LED        = 17   # GPIO 17 → Pin físico 11
PIN_VENTILADOR = 18   # GPIO 18 → Pin físico 12  (PWM, vía transistor)
PIN_SERVO      = 12   # GPIO 12 → Pin físico 32  (PWM hardware)
PIN_BUZZER     = 13   # GPIO 13 → Pin físico 33  (PWM hardware)

# ─────────────────────────────────────────────
# NOTAS MUSICALES (frecuencia en Hz)
# Escala de Do mayor para melodías del buzzer
# ─────────────────────────────────────────────

NOTAS = {
    "C4" : 262,   # Do 4
    "D4" : 294,   # Re 4
    "E4" : 330,   # Mi 4
    "F4" : 349,   # Fa 4
    "G4" : 392,   # Sol 4
    "A4" : 440,   # La 4
    "B4" : 494,   # Si 4
    "C5" : 523,   # Do 5
    "G3" : 196,   # Sol 3 (tono grave de alerta)
    "A3" : 220,   # La 3
    "E5" : 659,   # Mi 5 (tono agudo de confirmación)
}

# Melodías predefinidas: lista de (nota, duración_seg)
MELODIAS = {
    "confirmacion": [
        ("E4", 0.10), ("G4", 0.10), ("C5", 0.18),
    ],
    "alerta": [
        ("G3", 0.15), ("G3", 0.15), ("G3", 0.15),
    ],
    "apagado": [
        ("C5", 0.10), ("G4", 0.10), ("E4", 0.18),
    ],
    "mario": [
        ("E4", 0.12), ("E4", 0.12), ("E4", 0.18),
        ("C4", 0.12), ("E4", 0.18),
        ("G4", 0.25), ("G3", 0.25),
    ],
    "triunfo": [
        ("C4", 0.10), ("C4", 0.10), ("C4", 0.10), ("C4", 0.18),
        ("G3", 0.18), ("A3", 0.18), ("C4", 0.25),
    ],
}

# ─────────────────────────────────────────────
# MOCK — clases que simulan gpiozero en laptop
# ─────────────────────────────────────────────

class _MockLED:
    def __init__(self, pin): self.pin = pin; self._on = False
    def on(self):   self._on = True;  print(f"    [MOCK] LED  pin {self.pin} → ON")
    def off(self):  self._on = False; print(f"    [MOCK] LED  pin {self.pin} → OFF")
    def blink(self, on_time=0.5, off_time=0.5, n=None, background=True):
        print(f"    [MOCK] LED  pin {self.pin} → BLINK on={on_time}s off={off_time}s n={n}")
    def close(self): pass

class _MockPWM:
    def __init__(self, pin, freq=100): self.pin = pin; self.value = 0
    @property
    def value(self): return self._value
    @value.setter
    def value(self, v):
        self._value = v
        print(f"    [MOCK] PWM  pin {self.pin} → value={v:.2f}")
    def off(self): self.value = 0
    def close(self): pass

class _MockServo:
    def __init__(self, pin, **kw):
        self.pin = pin; self._angle = 0
    @property
    def angle(self): return self._angle
    @angle.setter
    def angle(self, a):
        self._angle = a
        print(f"    [MOCK] Servo pin {self.pin} → angle={a}°")
    def mid(self): self.angle = 0
    def close(self): pass

# ─────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────

class ActuatorController:
    """
    Interfaz unificada para los 4 actuadores del panel de domótica.

    Métodos públicos:
        execute(command)          → despacha un comando al actuador correcto
        led_on() / led_off()      → LED on/off
        led_blink(n, t)           → parpadeo n veces cada t segundos
        fan_set(speed)            → velocidad 0.0–1.0
        servo_set(angle)          → ángulo -90 a +90
        buzzer_tone(freq, dur)    → tono a frecuencia dada
        buzzer_melody(name)       → melodía predefinida (no bloquea)
        all_off()                 → apaga todo
        close()                   → libera GPIO
    """

    # Mapeo comando → método interno
    # Agrega aquí tus propios comandos según el dataset grabado
    COMMAND_MAP = {
        # Encendido general
        "enciende"      : "_cmd_enciende",
        # Apagado general
        "apaga"         : "_cmd_apaga",
        # Ventilador
        "ventilador"    : "_cmd_ventilador_toggle",
        # Servo (abrir / cerrar)
        "abrir"         : "_cmd_servo_abrir",
        "cerrar"        : "_cmd_servo_cerrar",
        # Buzzer / música
        "musica"        : "_cmd_musica",
        # Detener todo
        "detente"       : "_cmd_detente",
        # Ruido de fondo → no hacer nada
        "ruido_fondo"   : "_cmd_ignorar",
    }

    def __init__(self):
        self._fan_on   = False
        self._led_on   = False
        self._melody_thread = None
        self._lock     = threading.Lock()

        # ── Inicializar actuadores ───────────
        if ON_RPI:
            self._led  = LED(PIN_LED)
            self._fan  = PWMOutputDevice(PIN_VENTILADOR, frequency=100)
            self._servo= AngularServo(
                PIN_SERVO,
                min_angle = -90,
                max_angle =  90,
                min_pulse_width = 0.0005,
                max_pulse_width = 0.0025,
            )
            self._buzzer = PWMOutputDevice(PIN_BUZZER, frequency=440)
            self._buzzer.off()
        else:
            self._led   = _MockLED(PIN_LED)
            self._fan   = _MockPWM(PIN_VENTILADOR)
            self._servo = _MockServo(PIN_SERVO)
            self._buzzer= _MockPWM(PIN_BUZZER)

        self.all_off()
        print(f"[ActuatorController] Inicializado."
              f"  GPIO: {'real (RPi)' if ON_RPI else 'mock (laptop)'}")

    # ─────────────────────────────────────────
    # DESPACHO DE COMANDOS
    # ─────────────────────────────────────────

    def execute(self, command: str, confidence: float = 1.0) -> bool:
        """
        Ejecuta la acción correspondiente a 'command'.

        Parámetros:
            command    : nombre del comando (en minúsculas, sin espacios)
            confidence : probabilidad del modelo (0.0–1.0)

        Retorna True si el comando fue reconocido y ejecutado.
        """
        cmd = command.strip().lower()

        if cmd not in self.COMMAND_MAP:
            print(f"[CMD] '{cmd}' — comando desconocido (ignorado)")
            return False

        method_name = self.COMMAND_MAP[cmd]
        method      = getattr(self, method_name)

        print(f"[CMD] '{cmd}' (conf={confidence:.2f}) → {method_name}")
        method()
        return True

    # ─────────────────────────────────────────
    # IMPLEMENTACIONES DE COMANDOS
    # ─────────────────────────────────────────

    def _cmd_enciende(self):
        """ENCIENDE: LED ON + buzzer confirmación."""
        self.led_on()
        self.buzzer_melody("confirmacion", block=False)

    def _cmd_apaga(self):
        """APAGA: LED OFF + buzzer apagado."""
        self.led_off()
        self.buzzer_melody("apagado", block=False)

    def _cmd_ventilador_toggle(self):
        """VENTILADOR: alterna entre encendido (70%) y apagado."""
        with self._lock:
            if self._fan_on:
                self.fan_set(0.0)
                self._fan_on = False
                print("    → Ventilador OFF")
            else:
                self.fan_set(0.70)
                self._fan_on = True
                print("    → Ventilador ON (70%)")

    def _cmd_servo_abrir(self):
        """ABRIR: servo a +80° (posición abierta)."""
        self.servo_set(80)
        self.buzzer_tone(NOTAS["E5"], 0.08)

    def _cmd_servo_cerrar(self):
        """CERRAR: servo a -80° (posición cerrada)."""
        self.servo_set(-80)
        self.buzzer_tone(NOTAS["G3"], 0.08)

    def _cmd_musica(self):
        """MUSICA: reproduce melodía 'mario' en hilo separado."""
        self.buzzer_melody("mario", block=False)

    def _cmd_detente(self):
        """DETENTE: apaga ventilador, servo al centro, LED alerta."""
        self.fan_set(0.0)
        self._fan_on = False
        self.servo_set(0)
        self.led_blink(n=3, on_time=0.1, off_time=0.1)
        self.buzzer_melody("alerta", block=False)
        print("    → Todo detenido")

    def _cmd_ignorar(self):
        """RUIDO_FONDO: no ejecutar ninguna acción."""
        print("    → Ruido de fondo — sin acción")

    # ─────────────────────────────────────────
    # API DE ACTUADORES
    # ─────────────────────────────────────────

    def led_on(self):
        """Enciende el LED."""
        self._led.on()
        self._led_on = True

    def led_off(self):
        """Apaga el LED."""
        self._led.off()
        self._led_on = False

    def led_blink(self, n=3, on_time=0.2, off_time=0.2):
        """
        Parpadea el LED n veces.
        Usa el método blink() de gpiozero en background thread.
        """
        if ON_RPI:
            self._led.blink(on_time=on_time, off_time=off_time,
                            n=n, background=True)
        else:
            self._led.blink(on_time=on_time, off_time=off_time, n=n)

    def fan_set(self, speed: float):
        """
        Controla el ventilador.
        speed: 0.0 = apagado, 1.0 = máxima velocidad.
        El transistor 2N2222 convierte la señal PWM de 3.3V
        en la corriente necesaria para el motor DC de 5V.
        """
        speed = float(max(0.0, min(1.0, speed)))
        self._fan.value = speed

    def servo_set(self, angle: float):
        """
        Mueve el servo SG90 al ángulo indicado.
        angle: -90 (izquierda) a +90 (derecha), 0 = centro.
        """
        angle = float(max(-90, min(90, angle)))
        self._servo.angle = angle

    def buzzer_tone(self, frequency: float, duration: float):
        """
        Emite un tono a 'frequency' Hz durante 'duration' segundos.
        Bloquea el hilo actual por 'duration' segundos.
        El buzzer pasivo necesita una señal PWM con la frecuencia
        de la nota deseada — no un duty cycle fijo.
        """
        if frequency <= 0:
            return
        if ON_RPI:
            # gpiozero PWMOutputDevice usa frequency en Hz
            self._buzzer.frequency = frequency
            self._buzzer.value     = 0.5        # 50% duty cycle = volumen máximo
            time.sleep(duration)
            self._buzzer.off()
        else:
            print(f"    [MOCK] Buzzer → {frequency:.0f} Hz por {duration:.2f}s")
            time.sleep(duration)

    def buzzer_melody(self, name: str, block: bool = False):
        """
        Reproduce una melodía predefinida.
        block=False: corre en hilo separado (no bloquea el pipeline).
        block=True : bloquea hasta terminar.
        """
        if name not in MELODIAS:
            print(f"    [BUZZER] Melodía '{name}' no definida.")
            return

        def _play():
            for nota, duracion in MELODIAS[name]:
                freq = NOTAS.get(nota, 440)
                self.buzzer_tone(freq, duracion)
                time.sleep(0.02)    # pequeño silencio entre notas

        if block:
            _play()
        else:
            # Cancelar melodía anterior si sigue corriendo
            if self._melody_thread and self._melody_thread.is_alive():
                return   # no interrumpir una melodía en curso
            self._melody_thread = threading.Thread(target=_play, daemon=True)
            self._melody_thread.start()

    def all_off(self):
        """Apaga todos los actuadores de forma segura."""
        try:
            self._led.off()
            self._fan.off() if hasattr(self._fan, "off") else setattr(self._fan, "value", 0)
            self._servo.angle = 0
        except Exception:
            pass
        try:
            self._buzzer.off() if hasattr(self._buzzer, "off") else None
        except Exception:
            pass
        self._fan_on  = False
        self._led_on  = False

    def close(self):
        """Libera todos los recursos GPIO. Llamar al terminar el programa."""
        self.all_off()
        time.sleep(0.1)
        for dev in [self._led, self._fan, self._servo, self._buzzer]:
            try:
                dev.close()
            except Exception:
                pass
        print("[ActuatorController] GPIO liberado.")

# ─────────────────────────────────────────────
# PRUEBAS STANDALONE
# ─────────────────────────────────────────────

def test_led(ctrl):
    print("\n── TEST LED ─────────────────────────────")
    print("  Encendiendo LED...")
    ctrl.led_on();   time.sleep(1.0)
    print("  Apagando LED...")
    ctrl.led_off();  time.sleep(0.5)
    print("  Parpadeo 5 veces...")
    ctrl.led_blink(n=5, on_time=0.2, off_time=0.2)
    time.sleep(2.5)
    print("  ✔ TEST LED completado")


def test_ventilador(ctrl):
    print("\n── TEST VENTILADOR ──────────────────────")
    for speed, label in [(0.3, "30%"), (0.6, "60%"), (1.0, "100%"), (0.0, "OFF")]:
        print(f"  Velocidad {label}...")
        ctrl.fan_set(speed)
        time.sleep(1.5)
    print("  ✔ TEST VENTILADOR completado")


def test_servo(ctrl):
    print("\n── TEST SERVO SG90 ──────────────────────")
    for angle, label in [(0, "centro"), (80, "derecha"), (-80, "izquierda"), (0, "centro")]:
        print(f"  Ángulo: {label} ({angle}°)...")
        ctrl.servo_set(angle)
        time.sleep(1.2)
    print("  ✔ TEST SERVO completado")


def test_buzzer(ctrl):
    print("\n── TEST BUZZER ──────────────────────────")
    print("  Escala de Do mayor...")
    for nota in ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]:
        ctrl.buzzer_tone(NOTAS[nota], 0.18)
        time.sleep(0.05)

    time.sleep(0.3)
    print("  Melodía 'mario'...")
    ctrl.buzzer_melody("mario", block=True)

    time.sleep(0.3)
    print("  Melodía 'triunfo'...")
    ctrl.buzzer_melody("triunfo", block=True)
    print("  ✔ TEST BUZZER completado")


def test_all(ctrl):
    print("\n── TEST TODOS LOS ACTUADORES ────────────")
    test_led(ctrl)
    test_ventilador(ctrl)
    test_servo(ctrl)
    test_buzzer(ctrl)
    print("\n  ✔ TODOS LOS TESTS COMPLETADOS")


def demo_completa(ctrl):
    print("\n── DEMO PANEL DE DOMÓTICA ───────────────")
    commands_seq = [
        ("enciende",    0.95),
        ("ventilador",  0.91),
        ("abrir",       0.88),
        ("musica",      0.93),
        ("cerrar",      0.87),
        ("apaga",       0.96),
        ("detente",     0.92),
        ("ruido_fondo", 0.78),
    ]
    for cmd, conf in commands_seq:
        print(f"\n  Simulando comando: '{cmd.upper()}'")
        ctrl.execute(cmd, confidence=conf)
        time.sleep(2.0)

    ctrl.all_off()
    print("\n  ✔ DEMO COMPLETADA")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Módulo 8 — Prueba de actuadores GPIO"
    )
    parser.add_argument(
        "--test",
        choices=["led", "ventilador", "servo", "buzzer", "all"],
        help="Componente a probar",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo completa secuencial de todos los comandos",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        help="Ejecutar un comando específico directamente, ej: --cmd enciende",
    )
    args = parser.parse_args()

    ctrl = ActuatorController()

    try:
        if args.cmd:
            ctrl.execute(args.cmd)
            time.sleep(2.0)

        elif args.demo:
            demo_completa(ctrl)

        elif args.test == "led":
            test_led(ctrl)
        elif args.test == "ventilador":
            test_ventilador(ctrl)
        elif args.test == "servo":
            test_servo(ctrl)
        elif args.test == "buzzer":
            test_buzzer(ctrl)
        elif args.test == "all":
            test_all(ctrl)
        else:
            # Sin argumentos → demo rápida
            print("\nEjecutando demo rápida (usa --test <componente> para prueba individual)...")
            demo_completa(ctrl)

    except KeyboardInterrupt:
        print("\n[!] Interrumpido por usuario.")
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()