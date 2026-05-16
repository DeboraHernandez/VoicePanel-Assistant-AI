"""
gpio_controller.py — versión Arduino UNO
Panel de domótica por voz

Reemplaza gpiozero por comunicación Serial con Arduino UNO.
La laptop envía comandos de texto por USB; el Arduino los ejecuta.

Uso:
    python gpio_controller.py --test led
    python gpio_controller.py --demo
    python gpio_controller.py --list-ports   ← ver puertos disponibles
"""

import serial
import serial.tools.list_ports
import time
import threading
import sys
import argparse

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

BAUD_RATE    = 9600
TIMEOUT_S    = 2.0      # tiempo máximo esperando respuesta del Arduino
STARTUP_WAIT = 2.0      # espera al conectar (Arduino hace reset al conectar Serial)

def find_arduino_port():
    """
    Detecta automáticamente el puerto del Arduino.
    Busca dispositivos con 'Arduino' o 'CH340' en el nombre.
    CH340 es el chip USB que usan los Arduino UNO clones.
    """
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "ch341", "cp210", "usb serial"]):
            return p.device
    # Si no encuentra, devolver el primero disponible
    if ports:
        return ports[0].device
    return None

# ─────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────

class ActuatorController:
    """
    Interfaz idéntica a la versión RPi pero comunicándose
    con Arduino UNO por Serial USB en lugar de GPIO directo.

    La API pública es exactamente igual:
        ctrl.execute("enciende")
        ctrl.led_on()
        ctrl.fan_set(0.7)
        ctrl.all_off()
        ctrl.close()

    Así realtime_pipeline.py no necesita ningún cambio.
    """

    COMMAND_MAP = {
        "enciende"    : "_cmd_enciende",
        "apaga"       : "_cmd_apaga",
        "ventilador"  : "_cmd_ventilador_toggle",
        "abrir"       : "_cmd_servo_abrir",
        "cerrar"      : "_cmd_servo_cerrar",
        "musica"      : "_cmd_musica",
        "detente"     : "_cmd_detente",
        "ruido_fondo" : "_cmd_ignorar",
    }

    def __init__(self, port=None, baud=BAUD_RATE):
        self._lock   = threading.Lock()
        self._fan_on = False
        self._ser    = None

        # Detectar puerto automáticamente si no se especifica
        if port is None:
            port = find_arduino_port()

        if port is None:
            print("[ActuatorController] ⚠ Arduino no detectado — modo simulado")
            self._simulated = True
            return

        try:
            self._ser = serial.Serial(port, baud, timeout=TIMEOUT_S)
            time.sleep(STARTUP_WAIT)   # esperar reset del Arduino

            # Leer mensaje de bienvenida
            if self._ser.in_waiting:
                welcome = self._ser.readline().decode().strip()
                print(f"[ActuatorController] Arduino conectado en {port}: {welcome}")
            else:
                print(f"[ActuatorController] Arduino conectado en {port}")

            self._simulated = False

        except serial.SerialException as e:
            print(f"[ActuatorController] Error al conectar {port}: {e}")
            print("[ActuatorController] Usando modo simulado")
            self._simulated = True

    # ─────────────────────────────────────────
    # COMUNICACIÓN SERIAL
    # ─────────────────────────────────────────

    def _send(self, command: str) -> str:
        """
        Envía un comando al Arduino y espera la respuesta.
        Thread-safe gracias al lock.
        """
        if self._simulated or self._ser is None:
            print(f"    [SIMUL] → {command}")
            return "OK_SIMUL"

        with self._lock:
            try:
                self._ser.write(f"{command}\n".encode())
                response = self._ser.readline().decode().strip()
                if response.startswith("ERR"):
                    print(f"    [Arduino] Error: {response}")
                return response
            except serial.SerialException as e:
                print(f"    [Serial] Error: {e}")
                return "ERR_SERIAL"

    # ─────────────────────────────────────────
    # DESPACHO DE COMANDOS
    # ─────────────────────────────────────────

    def execute(self, command: str, confidence: float = 1.0) -> bool:
        cmd = command.strip().lower()
        if cmd not in self.COMMAND_MAP:
            print(f"[CMD] '{cmd}' — desconocido")
            return False
        method = getattr(self, self.COMMAND_MAP[cmd])
        print(f"[CMD] '{cmd}' (conf={confidence:.2f})")
        method()
        return True

    # ─────────────────────────────────────────
    # IMPLEMENTACIONES DE COMANDOS
    # ─────────────────────────────────────────

    def _cmd_enciende(self):
        self.led_on()
        self._send("BUZZ_CONFIRM")

    def _cmd_apaga(self):
        self.led_off()
        self._send("BUZZ_OFF")

    def _cmd_ventilador_toggle(self):
        resp = self._send("FAN_TOGGLE")
        self._fan_on = "ON" in resp and "OFF" not in resp

    def _cmd_servo_abrir(self):
        self._send("SERVO_OPEN")
        self._send("BUZZ_CONFIRM")

    def _cmd_servo_cerrar(self):
        self._send("SERVO_CLOSE")
        self._send("BUZZ_ALERT")

    def _cmd_musica(self):
        # No bloqueante — lanzar en hilo para no pausar el pipeline
        t = threading.Thread(target=lambda: self._send("BUZZ_MARIO"), daemon=True)
        t.start()

    def _cmd_detente(self):
        self.all_off()
        self._send("BUZZ_ALERT")

    def _cmd_ignorar(self):
        print("    → Ruido de fondo — sin acción")

    # ─────────────────────────────────────────
    # API DE ACTUADORES
    # ─────────────────────────────────────────

    def led_on(self):
        self._send("LED_ON")

    def led_off(self):
        self._send("LED_OFF")

    def led_blink(self, n=3, on_time=0.2, off_time=0.2):
        """Parpadeo implementado en la laptop (loop con delays)."""
        def _blink():
            for _ in range(n):
                self._send("LED_ON")
                time.sleep(on_time)
                self._send("LED_OFF")
                time.sleep(off_time)
        threading.Thread(target=_blink, daemon=True).start()

    def fan_set(self, speed: float):
        """speed: 0.0 = apagado, >0 = encendido."""
        if speed <= 0:
            self._send("FAN_OFF")
            self._fan_on = False
        else:
            self._send("FAN_ON")
            self._fan_on = True

    def servo_set(self, angle: float):
        """angle: -90=cerrar, 0=centro, +90=abrir (igual que módulo 8 original)."""
        if angle >= 45:
            self._send("SERVO_OPEN")
        elif angle <= -45:
            self._send("SERVO_CLOSE")
        else:
            self._send("SERVO_MID")

    def buzzer_tone(self, frequency: float, duration: float):
        self._send("BUZZ_CONFIRM")

    def buzzer_melody(self, name: str, block: bool = False):
        melody_map = {
            "confirmacion": "BUZZ_CONFIRM",
            "alerta"      : "BUZZ_ALERT",
            "apagado"     : "BUZZ_OFF",
            "mario"       : "BUZZ_MARIO",
            "triunfo"     : "BUZZ_CONFIRM",
        }
        cmd = melody_map.get(name, "BUZZ_CONFIRM")
        if block:
            self._send(cmd)
        else:
            threading.Thread(target=lambda: self._send(cmd), daemon=True).start()

    def all_off(self):
        self._send("ALL_OFF")
        self._fan_on = False

    def close(self):
        self.all_off()
        time.sleep(0.1)
        if self._ser and self._ser.is_open:
            self._ser.close()
        print("[ActuatorController] Puerto serial cerrado.")

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No se encontraron puertos seriales.")
        return
    print("Puertos disponibles:")
    for p in ports:
        print(f"  {p.device:<15} {p.description}")

# ─────────────────────────────────────────────
# PRUEBAS STANDALONE
# ─────────────────────────────────────────────

def demo_completa(ctrl):
    commands = [
        ("enciende",   0.95),
        ("ventilador", 0.91),
        ("abrir",      0.88),
        ("musica",     0.93),
        ("cerrar",     0.87),
        ("apaga",      0.96),
        ("detente",    0.92),
        ("ruido_fondo",0.78),
    ]
    print("\n── DEMO PANEL DE DOMÓTICA ───────────────")
    for cmd, conf in commands:
        print(f"\n  Comando: '{cmd.upper()}'")
        ctrl.execute(cmd, confidence=conf)
        time.sleep(2.0)
    ctrl.all_off()
    print("\n  ✔ DEMO COMPLETADA")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="gpio_controller.py — versión Arduino")
    parser.add_argument("--port",        type=str,   help="Puerto serial (ej: COM3, /dev/ttyUSB0)")
    parser.add_argument("--test",        choices=["led","ventilador","servo","buzzer","all"])
    parser.add_argument("--demo",        action="store_true")
    parser.add_argument("--cmd",         type=str,   help="Ejecutar un comando (ej: --cmd enciende)")
    parser.add_argument("--list-ports",  action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    ctrl = ActuatorController(port=args.port)

    try:
        if args.cmd:
            ctrl.execute(args.cmd)
            time.sleep(2.0)

        elif args.demo:
            demo_completa(ctrl)

        elif args.test == "led":
            print("Test LED...")
            ctrl.led_on();  time.sleep(1)
            ctrl.led_off(); time.sleep(0.5)
            ctrl.led_blink(n=5); time.sleep(3)

        elif args.test == "ventilador":
            print("Test ventilador...")
            ctrl.fan_set(0.7); time.sleep(2)
            ctrl.fan_set(0.0); time.sleep(1)

        elif args.test == "servo":
            print("Test servo...")
            ctrl.servo_set(80);  time.sleep(1.5)
            ctrl.servo_set(-80); time.sleep(1.5)
            ctrl.servo_set(0);   time.sleep(1)

        elif args.test == "buzzer":
            print("Test buzzer...")
            ctrl._send("BUZZ_CONFIRM"); time.sleep(1)
            ctrl._send("BUZZ_MARIO");   time.sleep(3)

        elif args.test == "all":
            ctrl.execute("enciende");   time.sleep(1.5)
            ctrl.execute("ventilador"); time.sleep(1.5)
            ctrl.execute("abrir");      time.sleep(1.5)
            ctrl.execute("musica");     time.sleep(3)
            ctrl.execute("detente");    time.sleep(1.5)

        else:
            demo_completa(ctrl)

    except KeyboardInterrupt:
        print("\n[!] Interrumpido.")
    finally:
        ctrl.close()

if __name__ == "__main__":
    main()