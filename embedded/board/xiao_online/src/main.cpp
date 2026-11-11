// Online SOC estimation on a XIAO (nRF52840 or ESP32-S3, board.h): INA228 (Adafruit, 15 mOhm shunt) on Wire, the
// DFN + EKF of embedded/include/dfn (model from platformio.ini), an SSD1306 OLED on Wire1, the DHT22 on the ESP32-S3
// board, and the CSV telemetry on USB serial and Bluetooth LE (Nordic UART, nus.h).
//
// Telemetry: the column names of the Battage firmware where the meaning is the same, so the Battage dashboard
// (BattageDiagnosticSystem/Embedded/dashboard/index.html) shows it unchanged: it reads the header line
// "# t_s,V,..." and takes the columns by name ("–" for the ones not sent). Discharge is positive.
//   t_s        time since start [s]
//   V          measured voltage: mean of the last second of the step [V]
//   I          step mean of the integrated current [A] (+ I_SELF_A, the board's own unmeasured draw, if set)
//   I_inst     latest shunt reading [A]
//   T_cell_C, T_amb_C   DHT22 air temperature [degC] (nan without it)
//   T_model_C  the temperature the model runs at: the measured one when valid (dfn::TH.set: Arrhenius kinetics,
//              diffusivities, R_contact, electrolyte, entropic term, from the header's tables), else the last one
//              (at start: the header's, 20 degC)
//   SoC, SoC_sig        EKF SOC and its 1 sigma
//   SoH, SoH_sig        nan (not estimated by this model)
//   V_est, innov        model voltage and innovation V - V_est [V]
//   Q_Ah       charge counted since the start [A.h]
//   flags      bit 0 EKF update, 3 V < 3.3 V, 4 V < 3.0 V, 6 no NTC, 7 DHT22 invalid, 8 charging
//   Rf, Rf_sig, LLI     film resistance factor and the LLI coordinate (0 when the model does not estimate it)
//   res, nit, step_ms, samples   closure residual, Newton iterations, EKF step time, sampler readings in the step
//
// Measurement: the INA228 converts shunt and bus continuously (1052 us each, 64x average: new values every 0.135 s).
// The sampler (board.h: above the EKF's priority, so the ~0.6 s step does not stop it) reads both every SAMPLE_MS and
// integrates the current in firmware. (The INA228's own CHARGE register accumulates only in the mode with temperature,
// and even there not the current: it counted +1 per second against -27 LSB of current.) Every STEP_S seconds:
// I = the integrated charge / elapsed time, V = the mean bus voltage of the last second (the model's V at the end of
// the step), then one EKF step.
// Start: the state is placed by inverting the model's OCV at the first voltage (assumes the cell is at rest).
// Commands (serial or Bluetooth, one per line): header, status, reset (re-initialize from the OCV now), help.
// Firmware update over Bluetooth (ESP32-S3): ble_ota, the Battage protocol: dashboard "Update firmware" with
// .pio/build/esp32s3/firmware.bin, rollback if the new image does not run for 30 s. The same button installs the
// Battage firmware (its firmware.bin) and back.
#include "board.h"
#include <Wire.h>
#include "nus.h"
#include "dfn/ekf.h"
#if defined(ARDUINO_ARCH_ESP32)
#include "ble_ota.h"
#include <esp_attr.h>
#include <esp_system.h>
#endif
#if HAVE_DHT
#include <DHT.h>
#endif

#ifndef STEP_S
#define STEP_S 5
#endif
#ifndef SAMPLE_MS
#define SAMPLE_MS 100
#endif
#ifndef I_SELF_A
#define I_SELF_A 0.0
#endif
#ifndef BLE_NAME
#define BLE_NAME "MPM-LE EKF"
#endif

static const float RSHUNT = 0.015f;                          // R015
static const double CURRENT_LSB = 10.0 / 524288.0;           // A per LSB (10 A full scale over 2^19)
static const uint8_t INA = 0x40;
static const char HEADER[] =
    "# t_s,V,I,I_inst,T_cell_C,T_amb_C,SoC,SoC_sig,SoH,SoH_sig,V_est,innov,Q_Ah,flags,Rf,Rf_sig,LLI,res,nit,step_ms,"
    "samples,T_model_C";

// ------------------------------------------------------------------ INA228
static void ina_w16(uint8_t reg, uint16_t v) {
    Wire.beginTransmission(INA); Wire.write(reg); Wire.write(v >> 8); Wire.write(v & 0xFF); Wire.endTransmission();
}
static bool ina_read(uint8_t reg, uint8_t* b, int n) {
    Wire.beginTransmission(INA); Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom(INA, (uint8_t)n) != n) return false;
    for (int i = 0; i < n; ++i) b[i] = Wire.read();
    return true;
}
static bool ina_ok = false;
static bool ina_init() {
    uint8_t b[2];
    if (!ina_read(0x3F, b, 2) || ((b[0] << 4) | (b[1] >> 4)) != 0x228) return false;
    ina_w16(0x00, 0x8000);                                   // reset
    delay(5);
    ina_w16(0x02, (uint16_t)(13107.2e6 * CURRENT_LSB * RSHUNT + 0.5));   // SHUNT_CAL (ADCRANGE 0, +-163.84 mV)
    // ADC_CONFIG: MODE 0xB = continuous shunt + bus, 1052 us each, 64x average. (It was 0xDB6B until 2026-11-10:
    // MODE 0xD is bus + temperature, the shunt was never converted and the current stayed at its power-up value.)
    ina_w16(0x01, 0xBB6B);
    return true;
}
static float ina_ishunt() {                                  // instantaneous current from VSHUNT [A]
    uint8_t b[3];
    if (!ina_read(0x04, b, 3)) return NAN;
    int32_t v = (int32_t)(((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8)) >> 12;
    return v * 312.5e-9f / RSHUNT;
}
static float ina_vbus() {
    uint8_t b[3];
    if (!ina_read(0x05, b, 3)) return NAN;
    uint32_t v = ((uint32_t)b[0] << 16 | (uint32_t)b[1] << 8 | b[2]) >> 4;
    return v * 195.3125e-6f;
}

// ------------------------------------------------------------------ sampler
static volatile double acc_q = 0;                            // integrated charge [C] (> 0 discharge)
static volatile int step_n = 0;
static volatile float last_i = 0;
static float vring[1000 / SAMPLE_MS];                        // bus voltages of the last second
static const int VR_N = (int)(sizeof(vring) / sizeof(vring[0]));
static int vring_i = 0, vring_n = 0;
static float iring[1000 / SAMPLE_MS];                        // shunt currents of the last second (screen)
static int iring_i = 0, iring_n = 0;
static uint32_t last_us = 0;

static void sample_cb() {
    float i = ina_ishunt(), v = ina_vbus();
    uint32_t now = micros();
    double dt = (now - last_us) * 1e-6;
    last_us = now;
    BOARD_LOCK();
    if (i == i) {
        acc_q += (double)i * dt; last_i = i;
        iring[iring_i] = i; iring_i = (iring_i + 1) % VR_N;
        if (iring_n < VR_N) ++iring_n;
    }
    if (v == v) {
        vring[vring_i] = v; vring_i = (vring_i + 1) % VR_N;
        if (vring_n < VR_N) ++vring_n;
    }
    ++step_n;
    BOARD_UNLOCK();
}
// the charge integrated since the last call [C], the mean voltage of the last second, and the readings in between
static void take(double* q, float* v, int* n) {
    BOARD_LOCK();
    *q = acc_q; acc_q = 0;
    double sv = 0;
    for (int k = 0; k < vring_n; ++k) sv += vring[k];
    *v = vring_n ? (float)(sv / vring_n) : NAN;
    *n = step_n; step_n = 0;
    BOARD_UNLOCK();
}
static float mean_i() {                                      // mean shunt current of the last second [A]
    BOARD_LOCK();
    double si = 0;
    for (int k = 0; k < iring_n; ++k) si += iring[k];
    float i = iring_n ? (float)(si / iring_n) : NAN;
    BOARD_UNLOCK();
    return i;
}
static float mean_v() {
    BOARD_LOCK();
    double sv = 0;
    for (int k = 0; k < vring_n; ++k) sv += vring[k];
    float v = vring_n ? (float)(sv / vring_n) : NAN;
    BOARD_UNLOCK();
    return v;
}

// ------------------------------------------------------------------ OLED (SSD1306 / SH1106 on Wire1)
static uint8_t oled_addr = 0;
static bool oled_sh1106 = false;                             // set true for a 1.3" SH1106 (2-column offset)
static uint8_t fb[8][128];
static bool o_probe(uint8_t a) { Wire1.beginTransmission(a); return Wire1.endTransmission() == 0; }
static void o_cmds(const uint8_t* c, int n) {
    Wire1.beginTransmission(oled_addr); Wire1.write(0x00);
    for (int i = 0; i < n; ++i) Wire1.write(c[i]);
    Wire1.endTransmission();
}
static bool oled_init() {
    if (o_probe(0x3C)) oled_addr = 0x3C;
    else if (o_probe(0x3D)) oled_addr = 0x3D;
    else return false;
    static const uint8_t init[] = {0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x02, 0xA1, 0xC8,
                                   0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF};
    o_cmds(init, sizeof(init));
    return true;
}
static void oled_flush() {
    if (!oled_addr) return;
    for (int p = 0; p < 8; ++p) {
        const uint8_t col = oled_sh1106 ? 2 : 0;
        uint8_t c[3] = {(uint8_t)(0xB0 + p), (uint8_t)(col & 0x0F), (uint8_t)(0x10 | (col >> 4))};
        o_cmds(c, 3);
        for (int x = 0; x < 128; x += 16) {                  // 16 data bytes per transfer (Wire buffers are small)
            Wire1.beginTransmission(oled_addr); Wire1.write(0x40);
            Wire1.write(&fb[p][x], 16);
            Wire1.endTransmission();
        }
    }
}
// 5x7 glyphs (columns, LSB at the top) for the characters the screen uses
struct Glyph { char c; uint8_t col[5]; };
static const Glyph FONT[] = {
    {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}}, {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}}, {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
    {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}}, {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}}, {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
    {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}}, {'7', {0x01, 0x71, 0x09, 0x05, 0x03}}, {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
    {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}}, {'.', {0x00, 0x60, 0x60, 0x00, 0x00}}, {'%', {0x23, 0x13, 0x08, 0x64, 0x62}},
    {'+', {0x08, 0x08, 0x3E, 0x08, 0x08}}, {'-', {0x08, 0x08, 0x08, 0x08, 0x08}}, {':', {0x00, 0x36, 0x36, 0x00, 0x00}},
    {'=', {0x14, 0x14, 0x14, 0x14, 0x14}}, {'/', {0x20, 0x10, 0x08, 0x04, 0x02}}, {'A', {0x7E, 0x11, 0x11, 0x11, 0x7E}},
    {'B', {0x7F, 0x49, 0x49, 0x49, 0x36}}, {'C', {0x3E, 0x41, 0x41, 0x41, 0x22}}, {'D', {0x7F, 0x41, 0x41, 0x22, 0x1C}},
    {'E', {0x7F, 0x49, 0x49, 0x49, 0x41}}, {'F', {0x7F, 0x09, 0x09, 0x09, 0x01}}, {'I', {0x00, 0x41, 0x7F, 0x41, 0x00}},
    {'K', {0x7F, 0x08, 0x14, 0x22, 0x41}}, {'L', {0x7F, 0x40, 0x40, 0x40, 0x40}}, {'N', {0x7F, 0x04, 0x08, 0x10, 0x7F}},
    {'O', {0x3E, 0x41, 0x41, 0x41, 0x3E}}, {'R', {0x7F, 0x09, 0x19, 0x29, 0x46}}, {'S', {0x46, 0x49, 0x49, 0x49, 0x31}},
    {'T', {0x01, 0x01, 0x7F, 0x01, 0x01}}, {'V', {0x1F, 0x20, 0x40, 0x20, 0x1F}}, {'f', {0x08, 0x7E, 0x09, 0x01, 0x02}},
    {'h', {0x7F, 0x08, 0x04, 0x04, 0x78}}, {'m', {0x7C, 0x04, 0x18, 0x04, 0x78}}, {'s', {0x48, 0x54, 0x54, 0x54, 0x20}},
    {'#', {0x44, 0x44, 0x5F, 0x44, 0x44}},                   // plus-minus sign
    {'*', {0x00, 0x06, 0x09, 0x09, 0x06}},                   // degree sign
};
static void set_px(int x, int y) { if (x >= 0 && x < 128 && y >= 0 && y < 64) fb[y >> 3][x] |= (uint8_t)(1 << (y & 7)); }
static void draw_text(int x, int y0, const char* s, int scale) {   // y0: top pixel row
    for (; *s; ++s) {
        const uint8_t* g = 0;
        for (const Glyph& f : FONT) if (f.c == *s) { g = f.col; break; }
        for (int cx = 0; cx < 5; ++cx) {
            uint8_t colb = g ? g[cx] : 0;
            for (int sx = 0; sx < scale; ++sx, ++x)
                for (int bit = 0; bit < 7; ++bit)
                    if (colb & (1 << bit))
                        for (int sy = 0; sy < scale; ++sy) set_px(x, y0 + bit * scale + sy);
        }
        x += scale;
    }
}
static int text_w(const char* s, int scale) { return (int)strlen(s) * 6 * scale - scale; }
static char* fmt(char* p, double v, int dec) {                // fixed point without printf float support; nan
    if (!(v == v) || v > 2e9 || v < -2e9) { memcpy(p, "nan", 4); return p + 3; }
    if (v < 0) { *p++ = '-'; v = -v; }
    long m = 1; for (int i = 0; i < dec; ++i) m *= 10;
    long q = (long)(v * m + 0.5), ip = q / m, fp = q % m;
    char t[12]; int n = 0;
    do { t[n++] = (char)('0' + ip % 10); ip /= 10; } while (ip);
    while (n) *p++ = t[--n];
    if (dec) { *p++ = '.'; for (long d = m / 10; d; d /= 10) *p++ = (char)('0' + (fp / d) % 10); }
    *p = 0;
    return p;
}

// ------------------------------------------------------------------ output
static void out(const char* line) {
    // USB serial only when the line fits in the TX buffer: with the cable in and no program reading the port, the
    // ESP32-S3's USB CDC blocked loop() despite setTxTimeoutMs(0) (2424 sampler readings in one 5 s step, a BLE
    // update's "begin" unanswered for minutes). Dropped lines still go to Bluetooth.
    size_t n = strlen(line);
    if (Serial.availableForWrite() >= (int)(n + 2)) Serial.println(line);
#if defined(ARDUINO_ARCH_ESP32)
    if (ble_ota::active()) return;                          // a firmware update owns the Bluetooth link
#endif
    nus::send_line(line);
}
static void firmware_line() {                               // the dashboard checks this id after an update
#if defined(ARDUINO_ARCH_ESP32)
    char b[64];
    snprintf(b, sizeof(b), "# firmware %s, CPU %lu MHz", ble_ota::firmwareId(), (unsigned long)getCpuFrequencyMhz());
    out(b);
#endif
}

// ------------------------------------------------------------------ temperature (DHT22, ESP32-S3 board)
static float t_amb = NAN;
#if HAVE_DHT && TEMP_NTC
// KY-013 analog module: 10 k NTC (B 3950) and a 10 k resistor between 3V3 and GND, the midpoint on S. Read
// ratiometrically (reference VDD/4 at gain 1/4: full scale = VDD), so the supply cancels. NTC_ON_TOP: the NTC between
// 3V3 and S (S rises with the temperature) instead of between S and GND.
#ifndef NTC_ON_TOP
#define NTC_ON_TOP 0
#endif
static float ntc_ratio = NAN;                                // last S / VDD (sensors-only output)
struct NtcSensor {
    void begin() { analogReference(AR_VDD4); analogReadResolution(12); }
    float readTemperature() {
        uint32_t s = 0;
        for (int k = 0; k < 16; ++k) s += analogRead(PIN_NTC);
        const float r = s / (16.0f * 4095.0f);
        ntc_ratio = r;
        if (r < 0.02f || r > 0.98f) return NAN;                // open or shorted
        const float R = NTC_ON_TOP ? 10000.0f * (1 - r) / r : 10000.0f * r / (1 - r);
        return 1.0f / (1.0f / 298.15f + logf(R / 10000.0f) / 3950.0f) - 273.15f;
    }
};
static NtcSensor dht;
#elif HAVE_DHT
static DHT dht(PIN_DHT, DHT22);
#endif
#if HAVE_DHT
static unsigned long dht_last = 0, dht_ok_ms = 0;
static void dht_poll() {
    unsigned long now = millis();
    if (now - dht_last < (TEMP_NTC ? 1000UL : 5000UL)) return;
    dht_last = now;
    float t = dht.readTemperature();
    if (t == t) { t_amb = t; dht_ok_ms = now; }
    else if (now - dht_ok_ms > 30000) t_amb = NAN;            // no valid reading for 30 s
}
#endif

// ------------------------------------------------------------------ estimation
static dfn::EKF ekf;
static double Ah_counted = 0;
static unsigned long t_prev, t_start, last_ms = 0;
static bool ready = false;
// plausible cell voltage: outside it (no cell on the INA's VBUS, a loose wire) the EKF is not corrected, and it is
// not started until a voltage inside it arrives
static const float V_MIN = 2.5f, V_MAX = 4.4f;
static bool v_ok(float V) { return V == V && V >= V_MIN && V <= V_MAX; }

// charge below the top of the window [A.h] whose model OCV is V: grid search over the whole window, q in
// [-0.4, 2.4] A.h at 0.5 mA.h, keeping both stoichiometries inside (0.001, 0.999). (It searched [-0.4, 0.6] A.h, as
// datasets.q_from_ocv for the test windows, until 2026-11-10: a cell near empty started at the edge, SOC 69 %.)
static double q_from_ocv(double V) {
    using namespace dfn_model;
    double best = 0, eb = 1e9;
    for (int i = 0; i <= 5600; ++i) {
        if (i % 256 == 0) board_wdt_feed();
        double q = -0.4 + i * 0.0005;
        double x = (double)x100 - q / (double)Q_n, y = (double)y100 + q / (double)Q_p;
        if (x <= 0.001 || x >= 0.999 || y <= 0.001 || y >= 0.999) continue;
        double U = dfn::Up<double>(y) - dfn::Un<double>(x)
                 + (double)dfn::TH.entdT * ((double)ent_d0 + (double)ent_dn * (x - 0.5) + (double)ent_dp * (y - 0.5));
        if (U == U && fabs(U - V) < eb) { eb = fabs(U - V); best = q; }
    }
    return best;
}

// reset cause, and the estimate kept across a reset that does not remove the power (ESP32-S3: RTC memory, which a
// software, watchdog or brown-out reset leaves alone): the EKF resumes from it instead of from the OCV, which under
// a charge or discharge current is off by the polarization
#if defined(ARDUINO_ARCH_ESP32)
static const char* reset_reason() {
    switch (esp_reset_reason()) {
        case ESP_RST_POWERON: return "power-on";
        case ESP_RST_SW: return "software";
        case ESP_RST_PANIC: return "crash";
        case ESP_RST_INT_WDT: case ESP_RST_TASK_WDT: case ESP_RST_WDT: return "watchdog";
        case ESP_RST_BROWNOUT: return "brown-out (supply dip)";
        case ESP_RST_DEEPSLEEP: return "deep sleep";
        case ESP_RST_EXT: return "reset pin";
        default: return "other";
    }
}
struct Kept { uint32_t magic; double q, c0, Ah; uint32_t check; };
RTC_NOINIT_ATTR static Kept kept;
static const uint32_t KEPT_MAGIC = 0xDF0E5A1Cu;
static uint32_t kept_check(const Kept& k) {
    const uint8_t* b = (const uint8_t*)&k;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < offsetof(Kept, check); ++i) h = (h ^ b[i]) * 16777619u;
    return h;
}
static void keep_state(const dfn::EkfOut& o) {
    kept.magic = KEPT_MAGIC;
    kept.q = ((double)dfn::ybar(ekf.x) - (double)dfn_model::y100) * (double)dfn_model::Q_p;   // charge below the top
    (void)o;
    kept.c0 = log((double)o.Rf);
    kept.Ah = Ah_counted;
    kept.check = kept_check(kept);
}
static bool kept_valid() {
    return esp_reset_reason() != ESP_RST_POWERON && kept.magic == KEPT_MAGIC && kept.check == kept_check(kept)
        && kept.q == kept.q && kept.c0 == kept.c0;
}
#else
static const char* reset_reason() { return board_reset_reason(); }
static void keep_state(const dfn::EkfOut&) {}
static bool kept_valid() { return false; }
#endif
static bool resumed_once = false;

static void init_at(double q, double c0) {
    real_t z0[dfn::NZ];
    for (int i = 0; i < dfn::NZ; ++i) z0[i] = (real_t)((double)dfn_model::z0[i] + (q - (double)dfn_model::q0) * (double)dfn_model::dz_dq[i]);
    ekf.init(z0, (real_t)c0, dfn_model::EKF_IDT, 0);
    ready = true;
}

static void start_from_ocv(float V) {
    crumb(5);
    double q = q_from_ocv(V);
    crumb(6);
    init_at(q, 0);
    crumb(7);
    board_wdt_feed();
    char b[120], *p = b;
    p += sprintf(p, "# start from OCV: V ");
    p = fmt(p, V, 4); p += sprintf(p, " V -> q ");
    p = fmt(p, q, 4); p += sprintf(p, " A.h below the top, SoC ");
    p = fmt(p, dfn::soc_of_ybar(dfn::ybar(ekf.x)) * 100.0, 2); p += sprintf(p, " %%");
    crumb(8);
    out(b);
    crumb(9);
}

static void restart(bool allow_resume) {
    double q_; float v_; int n_;
#if defined(ARDUINO_ARCH_ESP32)
    if (allow_resume && !resumed_once && kept_valid()) {
        resumed_once = true;
        init_at(kept.q, kept.c0);
        Ah_counted = kept.Ah;
        take(&q_, &v_, &n_);
        t_prev = millis();
        char b[120], *p = b;
        p += sprintf(p, "# resumed after the %s reset: SoC ", reset_reason());
        p = fmt(p, dfn::soc_of_ybar((real_t)((double)dfn_model::y100 + kept.q / (double)dfn_model::Q_p)) * 100.0, 2);
        p += sprintf(p, " %%, Rf ");
        p = fmt(p, exp(kept.c0), 3);
        out(b);
        return;
    }
#endif
    resumed_once = true;
    crumb(2);
#if HAVE_DHT
    if (!(t_amb == t_amb)) { float t = dht.readTemperature(); if (t == t) t_amb = t; }   // the OCV at the cell's temperature
#endif
    crumb(3);
    if (t_amb == t_amb) dfn::TH.set((real_t)t_amb);
    crumb(4);
    start_from_ocv(mean_v());
    crumb(10);
    take(&q_, &v_, &n_);
    t_prev = millis();
    Ah_counted = 0;
}

static void status() {
    char b[260];
    snprintf(b, sizeof(b), "# status: %s, model %s, step %d s, INA228 %s, OLED %s, BLE %s, %s %s, last EKF step %lu ms, "
             "up %lu s, last reset: %s",
             BOARD_NAME, dfn_model::TAG, STEP_S, ina_ok ? "ok" : "missing", oled_addr ? "ok" : "missing",
             nus::connected() ? "connected" : "advertising", TEMP_NTC ? "NTC" : "DHT22", HAVE_DHT ? (t_amb == t_amb ? "ok" : "no reading") : "none",
             last_ms, millis() / 1000UL, reset_reason());
    out(b);
}

static void command(const char* c) {
    while (*c == ' ') ++c;
    if (!*c) return;
    if (!strcmp(c, "header")) { out(HEADER); status(); }      // the dashboard asks for it on connecting
    else if (!strcmp(c, "status")) { firmware_line(); status(); }
    else if (!strcmp(c, "reset") || !strcmp(c, "r")) restart(false);
    else if (!strcmp(c, "help")) out("# commands: header, status, reset (re-initialize from the OCV now), help");
    else { char b[80]; snprintf(b, sizeof(b), "# unknown command: %.40s (help)", c); out(b); }
}

// Screen: 128x64, the top rows yellow, the rest blue.
//   yellow: SOC (+- 1 sigma)
//   blue:   SOC bar; measured V and I (discharge +; mean of the last second, redrawn every second); DHT22
//           temperature; bottom right DFN, BLE when a client is connected, EKF step time
static void show(float I, float V, const dfn::EkfOut& o, unsigned long ms) {
    if (!oled_addr) return;
    memset(fb, 0, sizeof(fb));
    char b[24], *p;
    draw_text(0, 8, "SOC", 1);
    p = b; p = fmt(p, o.soc * 100.0, 1); *p++ = '%'; *p = 0;
    draw_text(22, 1, b, 2);
    p = b; *p++ = '#'; p = fmt(p, o.sd_soc * 100.0, 1); *p++ = '%'; *p = 0;
    draw_text(128 - text_w(b, 1), 8, b, 1);
    for (int x = 0; x < 128; ++x) { set_px(x, 18); set_px(x, 24); }           // SOC bar
    for (int y = 18; y <= 24; ++y) { set_px(0, y); set_px(127, y); }
    double f = o.soc < 0 ? 0 : o.soc > 1 ? 1 : o.soc;
    for (int x = 2; x < 2 + (int)(f * 124 + 0.5); ++x)
        for (int y = 20; y <= 22; ++y) set_px(x, y);
    p = b; p = fmt(p, V, 3); *p++ = ' '; *p++ = 'V'; *p = 0;
    draw_text(0, 30, b, 1);
    p = b; p = fmt(p, I, 3); *p++ = ' '; *p++ = 'A'; *p = 0;
    draw_text(128 - text_w(b, 1), 30, b, 1);
    p = b; *p++ = 'T'; *p++ = ' ';
    if (t_amb == t_amb) { p = fmt(p, t_amb, 1); *p++ = ' '; *p++ = '*'; *p++ = 'C'; } else { *p++ = '-'; *p++ = '-'; }
    *p = 0;
    draw_text(0, 42, b, 1);
    p = b; memcpy(p, "DFN ", 4); p += 4;
    if (nus::connected()) { memcpy(p, "BLE ", 4); p += 4; }
    p = fmt(p, (double)ms, 0); *p++ = ' '; *p++ = 'm'; *p++ = 's'; *p = 0;
    draw_text(128 - text_w(b, 1), 56, b, 1);
    crumb(29);
    oled_flush();
}

static void app_setup() {
#if !defined(ARDUINO_ARCH_ESP32)
    board_boot_info();                                        // before Bluetooth takes the POWER registers
#endif
    Serial.begin(115200);
#if defined(ARDUINO_ARCH_ESP32)
    Serial.setTxTimeoutMs(0);                                 // no host reading the USB port: drop, do not block
#endif
    unsigned long w = millis();
    while (!Serial && millis() - w < 3000) delay(10);         // runs without a PC too
    board_begin_i2c();
    ina_ok = ina_init();
    if (!ina_ok) { board_swap_ina_pins(); ina_ok = ina_init(); }   // SDA / SCL the other way round
    if (!ina_ok) board_swap_ina_pins();
    oled_init();
    nus::begin(BLE_NAME);
#if HAVE_DHT
    dht.begin();
#endif
    char b[200];
    snprintf(b, sizeof(b), "# DFN online EKF on the %s, model %s"
#ifdef DFN_MODEL_SET
             ", set " DFN_MODEL_SET
#endif
             ", step %d s, Bluetooth name %s", BOARD_NAME, dfn_model::TAG, STEP_S, BLE_NAME);
    out(b);
#if defined(ARDUINO_ARCH_ESP32)
    ble_ota::begin(out, nullptr);                           // rollback check of a just-installed image
#endif
    firmware_line();
#if !defined(ARDUINO_ARCH_ESP32)
    if (board_prev_crumb) {                                   // where the previous run was when it was reset
        snprintf(b, sizeof(b), "# previous run: last progress code %u before the %s reset", board_prev_crumb,
                 board_reset_reason());
        out(b);
    }
    board_wdt_start(30);                                      // a stuck firmware resets after 30 s
#endif
    status();
    last_us = micros();
    board_start_sampler(SAMPLE_MS, sample_cb);
    delay(1200);                                              // a full second of readings before the first voltage
    t_prev = t_start = millis();
    out(HEADER);
}

// no cell voltage yet (or the state was lost): wait for a plausible voltage, then start from the OCV
static void wait_for_cell() {
    static unsigned long last_msg = 0;
    float V = mean_v();
    if (v_ok(V)) {
        crumb(1);
        restart(true);
        out(HEADER);
        crumb(11);
        return;
    }
    unsigned long now = millis();
    if (now - last_msg < 5000) return;
    last_msg = now;
    char b[80], *p = b;
    p += sprintf(p, "# waiting for a cell voltage in [2.5, 4.4] V, now ");
    p = fmt(p, V, 3); sprintf(p, " V");
    out(b);
    if (oled_addr) {
        memset(fb, 0, sizeof(fb));
        draw_text(0, 18, "NO CELL", 2);
        p = b; *p++ = 'V'; *p++ = ' '; p = fmt(p, V, 3); *p = 0;
        draw_text(0, 44, b, 1);
        oled_flush();
    }
}

// last EKF output, kept for the screen: V and I are redrawn every second (mean of the last second), the SOC at
// each EKF step
// Sensors only (env nrf52840_sensors, -DSENSORS_ONLY): no EKF; every second the mean V and I of the last second and
// the DHT22 temperature, on the serial port, Bluetooth and the screen. A check of the wiring.
#ifdef SENSORS_ONLY
static void sensors_loop() {
    static unsigned long last = 0;
    unsigned long now = millis();
    if (now - last < 1000) { delay(5); return; }
    last = now;
    float V = mean_v(), I = mean_i();
    char b[96], *p = b;
    p += sprintf(p, "# sensors: t ");
    p = fmt(p, now * 1e-3, 1); p += sprintf(p, " s, V ");
    p = fmt(p, V, 4); p += sprintf(p, " V, I ");
    p = fmt(p, I * 1e3, 2); p += sprintf(p, " mA, T ");
    p = fmt(p, t_amb, 1); p += sprintf(p, " degC");
#if HAVE_DHT && TEMP_NTC
    p += sprintf(p, " (NTC S/VDD "); p = fmt(p, ntc_ratio, 4); sprintf(p, ")");
#endif
    out(b);
    if (!oled_addr) return;
    memset(fb, 0, sizeof(fb));
    draw_text(0, 1, "SENSORS", 1);
    p = b; p = fmt(p, V, 3); *p++ = ' '; *p++ = 'V'; *p = 0;
    draw_text(0, 18, b, 2);
    p = b; p = fmt(p, I * 1e3, 1); *p++ = ' '; *p++ = 'm'; *p++ = 'A'; *p = 0;
    draw_text(0, 36, b, 2);
    p = b; *p++ = 'T'; *p++ = ' ';
    if (t_amb == t_amb) { p = fmt(p, t_amb, 1); *p++ = ' '; *p++ = '*'; *p++ = 'C'; } else { *p++ = '-'; *p++ = '-'; }
    *p = 0;
    draw_text(0, 56, b, 1);
    oled_flush();
}
#endif

static dfn::EkfOut last_o;
static bool have_o = false;
static unsigned long last_show = 0;

static void app_loop() {
    board_wdt_feed();
    crumb(30);
    static char sline[64]; static size_t sn = 0;              // serial command line
    while (Serial.available()) {
        int c = Serial.read();
        if (c == '\n' || c == '\r') { sline[sn] = 0; command(sline); sn = 0; }
        else if (sn < sizeof(sline) - 1) sline[sn++] = (char)c;
    }
    char cmd[64];
    while (nus::read_line(cmd, sizeof(cmd))) command(cmd);
#if defined(ARDUINO_ARCH_ESP32)
    ble_ota::poll();                                        // update chunks to flash, rollback confirmation, reboot
#endif
    crumb(31);
#if HAVE_DHT
    dht_poll();
#endif
    crumb(32);
#ifdef SENSORS_ONLY
    sensors_loop();
    return;
#endif
    if (!ready) { wait_for_cell(); delay(5); return; }
    unsigned long now = millis();
    if (have_o && now - last_show >= 1000UL) {                  // screen: fresh V and I every second
        last_show = now;
        crumb(33);
        show((float)(mean_i() + I_SELF_A), mean_v(), last_o, last_ms);
    }
    if (now - t_prev < (unsigned long)STEP_S * 1000UL) { delay(5); return; }
    crumb(20);
    double q; float V; int nsamp;
    take(&q, &V, &nsamp);
    if (t_amb == t_amb) dfn::TH.set((real_t)t_amb);            // the model at the measured temperature
    crumb(21);
    double dt = (now - t_prev) * 1e-3;
    float I = (float)(q / dt + I_SELF_A);
    t_prev = now;
    Ah_counted += I * dt / 3600.0;
    unsigned long t0 = millis();
    bool use = v_ok(V) && I == I;
    dfn::EkfOut o = ekf.step((real_t)I, (real_t)(use ? V : 3.7f), dfn_model::EKF_IDT, use);
    last_ms = millis() - t0;
    crumb(22);
    if (!(o.soc == o.soc) || !(o.V == o.V)) {                 // state lost: start again from the OCV
        crumb(26);
        out("# EKF state not finite: restarting from the OCV once the voltage is plausible");
        ready = false;
        return;
    }
    crumb(27);
    int flags = (use ? 1 : 0) | (V < 3.3f ? 1 << 3 : 0) | (V < 3.0f ? 1 << 4 : 0) | (1 << 6)
              | ((HAVE_DHT && t_amb == t_amb) ? 0 : 1 << 7) | (I < -0.01f ? 1 << 8 : 0);
    char line[260], *p = line;
    p = fmt(p, (now - t_start) * 1e-3, 1); *p++ = ',';
    p = fmt(p, V, 5); *p++ = ',';
    p = fmt(p, I, 5); *p++ = ',';
    p = fmt(p, last_i, 5); *p++ = ',';
    p = fmt(p, t_amb, 1); *p++ = ',';
    p = fmt(p, t_amb, 1); *p++ = ',';
    p = fmt(p, o.soc, 5); *p++ = ',';
    p = fmt(p, o.sd_soc, 5); *p++ = ',';
    p = fmt(p, NAN, 0); *p++ = ',';
    p = fmt(p, NAN, 0); *p++ = ',';
    p = fmt(p, o.V, 5); *p++ = ',';
    p = fmt(p, o.innov, 5); *p++ = ',';
    p = fmt(p, Ah_counted, 6); *p++ = ',';
    p = fmt(p, (double)flags, 0); *p++ = ',';
    p = fmt(p, o.Rf, 4); *p++ = ',';
    p = fmt(p, o.sd_Rf, 4); *p++ = ',';
    p = fmt(p, o.LLI, 5); *p++ = ',';
    if (o.res == o.res) { p = fmt(p, o.res * 1e6, 2); *p++ = 'e'; *p++ = '-'; *p++ = '6'; } else p = fmt(p, NAN, 0);
    *p++ = ',';
    p = fmt(p, o.nit, 0); *p++ = ',';
    p = fmt(p, (double)last_ms, 0); *p++ = ',';
    p = fmt(p, (double)nsamp, 0); *p++ = ',';
    p = fmt(p, dfn::TH.Tc, 2); *p = 0;
    crumb(23);
    out(line);
    crumb(24);
    last_o = o; have_o = true; last_show = millis();
    keep_state(o);
    crumb(28);
    show(I, V, o, last_ms);
    crumb(25);
}

// The nRF52840 core runs setup() and loop() on a 4 KB stack: the start from the OCV (Newton closure, OCV inversion,
// printf) overflowed it and the board froze as soon as a cell appeared. There the firmware runs in its own task with
// a 16 KB stack; the ESP32-S3's loop task has room enough.
#if defined(ARDUINO_ARCH_ESP32)
void setup() { app_setup(); }
void loop() { app_loop(); }
#else
static void app_task(void*) {
    app_setup();
    for (;;) { app_loop(); yield(); }
}
void setup() { xTaskCreate(app_task, "dfn", 4096, nullptr, TASK_PRIO_LOW, nullptr); }   // 4096 words = 16 KB
void loop() { delay(1000); }
#endif
