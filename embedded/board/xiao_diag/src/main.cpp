// Wiring check of the online set-up (output on USB serial, 115200).
// 1. Display lines D8 / D9: idle level with pull-up, without, and with pull-down. A line that still reads 1 with the
//    internal pull-down is driven by an external pull-up (the OLED module's), so the module is powered. Then a short
//    test: D8 driven low / high while D9 is read (and vice versa).
// 2. Display bus scanned with the second hardware I2C block (Wire1, TWIM1) at 100 kHz, D9 SDA / D8 SCL and swapped,
//    and with a slow bit-banged probe (~10 kHz, 9-clock bus recovery first) in both orientations.
// 3. INA228 on Wire (SDA = D5, SCL = D4, as soldered): identification and raw readings once a second.
#include <Arduino.h>
#include <Adafruit_TinyUSB.h>
#include <Wire.h>

static const float RSHUNT = 0.015f;                      // Adafruit INA228 board: R015

// ---- minimal open-drain bit-banged I2C (address probe only)
static int b_sda = D9, b_scl = D8, b_us = 50;
static void b_hi(int p) { pinMode(p, INPUT_PULLUP); }
static void b_lo(int p) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
static void b_d() { delayMicroseconds(b_us); }
static void b_start() { b_hi(b_sda); b_hi(b_scl); b_d(); b_lo(b_sda); b_d(); b_lo(b_scl); b_d(); }
static void b_stop() { b_lo(b_sda); b_d(); b_hi(b_scl); b_d(); b_hi(b_sda); b_d(); }
static bool b_write(uint8_t v) {
    for (int i = 7; i >= 0; --i) {
        if (v & (1 << i)) b_hi(b_sda); else b_lo(b_sda);
        b_d(); b_hi(b_scl); b_d(); b_lo(b_scl); b_d();
    }
    b_hi(b_sda); b_d(); b_hi(b_scl); b_d();
    bool ack = digitalRead(b_sda) == LOW;
    b_lo(b_scl); b_d();
    return ack;
}
static void b_recover() {                                // 9 clocks with SDA released, then a stop
    b_hi(b_sda);
    for (int i = 0; i < 9; ++i) { b_lo(b_scl); b_d(); b_hi(b_scl); b_d(); }
    b_stop();
}

static int level(int pin, uint32_t mode) { pinMode(pin, mode); delay(5); return digitalRead(pin); }

static void scan_hw(int sda, int scl, const char* label) {
    Wire1.setPins(sda, scl);
    Wire1.begin();
    Wire1.setClock(100000);
    Serial.print("# hardware I2C (Wire1, 100 kHz) "); Serial.print(label); Serial.print(":");
    int n = 0;
    for (uint8_t a = 0x08; a < 0x78; ++a) {
        Wire1.beginTransmission(a);
        if (Wire1.endTransmission() == 0) { Serial.print(" 0x"); Serial.print(a, HEX); ++n; }
    }
    Serial.println(n ? "" : " nothing");
    Wire1.end();
    pinMode(sda, INPUT); pinMode(scl, INPUT);
}

static void scan_bb(int sda, int scl, const char* label) {
    b_sda = sda; b_scl = scl;
    b_recover();
    Serial.print("# bit-banged (~10 kHz) "); Serial.print(label); Serial.print(":");
    int n = 0;
    for (uint8_t a = 0x08; a < 0x78; ++a) {
        b_start(); bool ok = b_write(a << 1); b_stop();
        if (ok) { Serial.print(" 0x"); Serial.print(a, HEX); ++n; }
    }
    Serial.println(n ? "" : " nothing");
    pinMode(sda, INPUT); pinMode(scl, INPUT);
}

static int ina = -1;
static bool is228 = false;
static uint16_t rd16(uint8_t a, uint8_t reg) {
    Wire.beginTransmission(a); Wire.write(reg); Wire.endTransmission(false);
    Wire.requestFrom(a, (uint8_t)2);
    uint16_t v = (uint16_t)Wire.read() << 8; v |= Wire.read();
    return v;
}
static uint32_t rd24(uint8_t a, uint8_t reg) {
    Wire.beginTransmission(a); Wire.write(reg); Wire.endTransmission(false);
    Wire.requestFrom(a, (uint8_t)3);
    uint32_t v = (uint32_t)Wire.read() << 16; v |= (uint32_t)Wire.read() << 8; v |= Wire.read();
    return v;
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    delay(500);

    // 1. levels
    const int pins[2] = {D8, D9};
    const char* names[2] = {"D8", "D9"};
    for (int i = 0; i < 2; ++i) {
        int up = level(pins[i], INPUT_PULLUP), fl = level(pins[i], INPUT), dn = level(pins[i], INPUT_PULLDOWN);
        Serial.print("# "); Serial.print(names[i]); Serial.print(": pull-up "); Serial.print(up);
        Serial.print(", floating "); Serial.print(fl); Serial.print(", pull-down "); Serial.print(dn);
        Serial.println(dn ? "  -> external pull-up present (module powered)" : "  -> no external pull-up");
        pinMode(pins[i], INPUT);
    }
    for (int i = 0; i < 2; ++i) {                        // short test: drive one line, read the other (pull-down)
        int drv = pins[i], rd = pins[1 - i];
        pinMode(rd, INPUT_PULLDOWN);
        pinMode(drv, OUTPUT);
        digitalWrite(drv, LOW); delay(2); int lo = digitalRead(rd);
        digitalWrite(drv, HIGH); delay(2); int hi = digitalRead(rd);
        Serial.print("# drive "); Serial.print(names[i]); Serial.print(" low/high -> "); Serial.print(names[1 - i]);
        Serial.print(" reads "); Serial.print(lo); Serial.print("/"); Serial.println(hi);
        pinMode(drv, INPUT); pinMode(rd, INPUT);
    }

    // 2. display bus
    scan_hw(D9, D8, "SDA=D9 SCL=D8");
    scan_hw(D8, D9, "SDA=D8 SCL=D9");
    scan_bb(D9, D8, "SDA=D9 SCL=D8");
    scan_bb(D8, D9, "SDA=D8 SCL=D9");

    // 3. INA228 on the main bus
    Wire.setPins(D5, D4);
    Wire.begin();
    for (uint8_t a = 0x40; a <= 0x4F; ++a) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) { ina = a; break; }
    }
    if (ina >= 0) {
        uint16_t man = rd16(ina, 0x3E), dev = rd16(ina, 0x3F);
        is228 = man == 0x5449 && (dev >> 4) == 0x228;
        Serial.print("# INA at 0x"); Serial.print(ina, HEX); Serial.println(is228 ? ": INA228" : ": not an INA228");
        if (is228) {                                     // reset, then continuous shunt + bus + temperature
            Wire.beginTransmission(ina); Wire.write(0x00); Wire.write(0x80); Wire.write(0x00); Wire.endTransmission();
            delay(5);
            Wire.beginTransmission(ina); Wire.write(0x01); Wire.write(0xFB); Wire.write(0x6F); Wire.endTransmission();
        }
    } else {
        Serial.println("# no INA in 0x40-0x4F on D5 SDA / D4 SCL");
    }
}

void loop() {
    if (is228) {
        // 20-bit values in bits 23..4; shunt LSB 312.5 nV (ADCRANGE 0), bus LSB 195.3125 uV, die temp 7.8125 m degC
        int32_t sh = (int32_t)(rd24(ina, 0x04) << 8) >> 12;
        uint32_t bus = rd24(ina, 0x05) >> 4;
        int16_t tdie = (int16_t)rd16(ina, 0x06);
        float vsh = sh * 312.5e-9f;
        Serial.print("V_bus "); Serial.print(bus * 195.3125e-6f, 4);
        Serial.print(" V  I "); Serial.print(vsh / RSHUNT * 1e3f, 2);
        Serial.print(" mA  T_die "); Serial.println((tdie >> 4) * 0.125f, 2);
    }
    delay(1000);
}
