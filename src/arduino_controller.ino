/*
 * arduino_controller.ino
 * Panel de domótica por voz — Arduino UNO
 * 
 * Recibe comandos por Serial USB desde la laptop
 * y controla los actuadores físicos.
 * 
 * Baud rate: 9600
 * Formato:   comando\n  (ej: "LED_ON\n", "FAN_ON\n")
 */

#include <Servo.h>

// ── Pines ────────────────────────────────────
#define PIN_LED     7
#define PIN_FAN     9    // PWM~
#define PIN_SERVO   6    // PWM~
#define PIN_BUZZER  11   // PWM~

// ── Notas musicales (frecuencia Hz) ──────────
#define NOTE_C4  262
#define NOTE_D4  294
#define NOTE_E4  330
#define NOTE_G4  392
#define NOTE_G3  196
#define NOTE_C5  523
#define NOTE_E5  659

Servo myServo;
String inputBuffer = "";
bool fanOn = false;/*
 * arduino_controller.ino
 * Panel de domótica por voz — Arduino UNO
 * 
 * Recibe comandos por Serial USB desde la laptop
 * y controla los actuadores físicos.
 * 
 * Baud rate: 9600
 * Formato:   comando\n  (ej: "LED_ON\n", "FAN_ON\n")
 */

#include <Servo.h>

// ── Pines ────────────────────────────────────
#define PIN_LED     7
#define PIN_FAN     9    // PWM~
#define PIN_SERVO   6    // PWM~
#define PIN_BUZZER  11   // PWM~

// ── Notas musicales (frecuencia Hz) ──────────
#define NOTE_C4  262
#define NOTE_D4  294
#define NOTE_E4  330
#define NOTE_G4  392
#define NOTE_G3  196
#define NOTE_C5  523
#define NOTE_E5  659

Servo myServo;
String inputBuffer = "";
bool fanOn = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_LED,    OUTPUT);
  pinMode(PIN_FAN,    OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  
  myServo.attach(PIN_SERVO);
  myServo.write(90);   // posición centro (0°–180° en Arduino)
  
  // Apagar todo al iniciar
  digitalWrite(PIN_LED, LOW);
  analogWrite(PIN_FAN, 0);
  noTone(PIN_BUZZER);
  
  Serial.println("ARDUINO_READY");
}

void loop() {
  // Leer comando completo hasta '\n'
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  if      (cmd == "LED_ON")     { digitalWrite(PIN_LED, HIGH); Serial.println("OK_LED_ON"); }
  else if (cmd == "LED_OFF")    { digitalWrite(PIN_LED, LOW);  Serial.println("OK_LED_OFF"); }
  else if (cmd == "FAN_ON")     { analogWrite(PIN_FAN, 180); fanOn = true;  Serial.println("OK_FAN_ON"); }
  else if (cmd == "FAN_OFF")    { analogWrite(PIN_FAN, 0);   fanOn = false; Serial.println("OK_FAN_OFF"); }
  else if (cmd == "FAN_TOGGLE") {
    if (fanOn) { analogWrite(PIN_FAN, 0);   fanOn = false; Serial.println("OK_FAN_OFF"); }
    else       { analogWrite(PIN_FAN, 180); fanOn = true;  Serial.println("OK_FAN_ON"); }
  }
  else if (cmd == "SERVO_OPEN")  { myServo.write(170); Serial.println("OK_SERVO_OPEN"); }
  else if (cmd == "SERVO_CLOSE") { myServo.write(10);  Serial.println("OK_SERVO_CLOSE"); }
  else if (cmd == "SERVO_MID")   { myServo.write(90);  Serial.println("OK_SERVO_MID"); }
  else if (cmd == "BUZZ_CONFIRM"){ playConfirm(); Serial.println("OK_BUZZ"); }
  else if (cmd == "BUZZ_ALERT")  { playAlert();   Serial.println("OK_BUZZ"); }
  else if (cmd == "BUZZ_OFF")    { playOff();     Serial.println("OK_BUZZ"); }
  else if (cmd == "BUZZ_MARIO")  { playMario();   Serial.println("OK_BUZZ"); }
  else if (cmd == "ALL_OFF")     {
    digitalWrite(PIN_LED, LOW);
    analogWrite(PIN_FAN, 0);
    fanOn = false;
    myServo.write(90);
    noTone(PIN_BUZZER);
    Serial.println("OK_ALL_OFF");
  }
  else { Serial.println("ERR_UNKNOWN:" + cmd); }
}

// ── Melodías ──────────────────────────────────
void playNote(int freq, int dur) {
  tone(PIN_BUZZER, freq, dur);
  delay(dur + 20);
}

void playConfirm() {
  playNote(NOTE_E4, 100);
  playNote(NOTE_G4, 100);
  playNote(NOTE_C5, 180);
}

void playAlert() {
  playNote(NOTE_G3, 150);
  playNote(NOTE_G3, 150);
  playNote(NOTE_G3, 150);
}

void playOff() {
  playNote(NOTE_C5, 100);
  playNote(NOTE_G4, 100);
  playNote(NOTE_E4, 180);
}

void playMario() {
  playNote(NOTE_E4, 120);
  playNote(NOTE_E4, 120);
  playNote(NOTE_E4, 180);
  playNote(NOTE_C4, 120);
  playNote(NOTE_E4, 180);
  playNote(NOTE_G4, 250);
  playNote(NOTE_G3, 250);
}

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_LED,    OUTPUT);
  pinMode(PIN_FAN,    OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  
  myServo.attach(PIN_SERVO);
  myServo.write(90);   // posición centro (0°–180° en Arduino)
  
  // Apagar todo al iniciar
  digitalWrite(PIN_LED, LOW);
  analogWrite(PIN_FAN, 0);
  noTone(PIN_BUZZER);
  
  Serial.println("ARDUINO_READY");
}

void loop() {
  // Leer comando completo hasta '\n'
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  if      (cmd == "LED_ON")     { digitalWrite(PIN_LED, HIGH); Serial.println("OK_LED_ON"); }
  else if (cmd == "LED_OFF")    { digitalWrite(PIN_LED, LOW);  Serial.println("OK_LED_OFF"); }
  else if (cmd == "FAN_ON")     { analogWrite(PIN_FAN, 180); fanOn = true;  Serial.println("OK_FAN_ON"); }
  else if (cmd == "FAN_OFF")    { analogWrite(PIN_FAN, 0);   fanOn = false; Serial.println("OK_FAN_OFF"); }
  else if (cmd == "FAN_TOGGLE") {
    if (fanOn) { analogWrite(PIN_FAN, 0);   fanOn = false; Serial.println("OK_FAN_OFF"); }
    else       { analogWrite(PIN_FAN, 180); fanOn = true;  Serial.println("OK_FAN_ON"); }
  }
  else if (cmd == "SERVO_OPEN")  { myServo.write(170); Serial.println("OK_SERVO_OPEN"); }
  else if (cmd == "SERVO_CLOSE") { myServo.write(10);  Serial.println("OK_SERVO_CLOSE"); }
  else if (cmd == "SERVO_MID")   { myServo.write(90);  Serial.println("OK_SERVO_MID"); }
  else if (cmd == "BUZZ_CONFIRM"){ playConfirm(); Serial.println("OK_BUZZ"); }
  else if (cmd == "BUZZ_ALERT")  { playAlert();   Serial.println("OK_BUZZ"); }
  else if (cmd == "BUZZ_OFF")    { playOff();     Serial.println("OK_BUZZ"); }
  else if (cmd == "BUZZ_MARIO")  { playMario();   Serial.println("OK_BUZZ"); }
  else if (cmd == "ALL_OFF")     {
    digitalWrite(PIN_LED, LOW);
    analogWrite(PIN_FAN, 0);
    fanOn = false;
    myServo.write(90);
    noTone(PIN_BUZZER);
    Serial.println("OK_ALL_OFF");
  }
  else { Serial.println("ERR_UNKNOWN:" + cmd); }
}

// ── Melodías ──────────────────────────────────
void playNote(int freq, int dur) {
  tone(PIN_BUZZER, freq, dur);
  delay(dur + 20);
}

void playConfirm() {
  playNote(NOTE_E4, 100);
  playNote(NOTE_G4, 100);
  playNote(NOTE_C5, 180);
}

void playAlert() {
  playNote(NOTE_G3, 150);
  playNote(NOTE_G3, 150);
  playNote(NOTE_G3, 150);
}

void playOff() {
  playNote(NOTE_C5, 100);
  playNote(NOTE_G4, 100);
  playNote(NOTE_E4, 180);
}

void playMario() {
  playNote(NOTE_E4, 120);
  playNote(NOTE_E4, 120);
  playNote(NOTE_E4, 180);
  playNote(NOTE_C4, 120);
  playNote(NOTE_E4, 180);
  playNote(NOTE_G4, 250);
  playNote(NOTE_G3, 250);
}