// Board layer of the online firmware: pins, critical sections and the 10 Hz sampler, for the two XIAO boards.
//   nrf52840: XIAO nRF52840, INA228 as soldered on the first board (SDA = D5, SCL = D4: the reverse of the drawing;
//             the drawing's SDA = D4 is found at start-up too); OLED SDA = D9, SCL = D8; DHT22 on D10.
//   esp32s3:  the Battage XIAO ESP32-S3 (BattageDiagnosticSystem/Embedded/firmware/include/config.h): INA228 SDA = D4,
//             SCL = D5; OLED on Wire1 SDA = D9, SCL = D8; DHT22 on D10.
#pragma once
#include <Arduino.h>
#include <Wire.h>

#if defined(ARDUINO_ARCH_ESP32)
#define BOARD_NAME "XIAO ESP32-S3"
static const int PIN_INA_SDA = D4, PIN_INA_SCL = D5;
static const int PIN_OLED_SDA = D9, PIN_OLED_SCL = D8;
static const int PIN_NTC = A1;                               // temperature (main.cpp finds which is there):
static const int PIN_DHT = D10;                              //   KY-013 NTC on D1 (A1, ADC1) or DHT22 on D10
static portMUX_TYPE board_mux = portMUX_INITIALIZER_UNLOCKED;
#define BOARD_LOCK() portENTER_CRITICAL(&board_mux)
#define BOARD_UNLOCK() portEXIT_CRITICAL(&board_mux)

// sampler: a FreeRTOS task on core 0 (the EKF runs in loop() on core 1)
static void (*board_cb)() = nullptr;
static uint32_t board_period_ms = 100;
static void board_task(void*) {
    TickType_t next = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&next, pdMS_TO_TICKS(board_period_ms));
        board_cb();
    }
}
static void board_start_sampler(uint32_t period_ms, void (*cb)()) {
    board_cb = cb;
    board_period_ms = period_ms;
    xTaskCreatePinnedToCore(board_task, "sampler", 4096, nullptr, 5, nullptr, 0);
}

#else   // nRF52840 (Adafruit-based Seeed core)
#include <Adafruit_TinyUSB.h>
#define BOARD_NAME "XIAO nRF52840"
static const int PIN_INA_SDA = D5, PIN_INA_SCL = D4;
static const int PIN_OLED_SDA = D9, PIN_OLED_SCL = D8;
static const int PIN_NTC = A1;                               // temperature (main.cpp finds which is there):
static const int PIN_DHT = D10;                              //   KY-013 NTC on D1 (A1) or DHT22 on D10
#define BOARD_LOCK() taskENTER_CRITICAL()
#define BOARD_UNLOCK() taskEXIT_CRITICAL()

// sampler: a FreeRTOS software timer (timer task above loop()'s priority, so the ~0.6 s EKF step does not stop it)
static SoftwareTimer board_timer;
static void (*board_cb)() = nullptr;
static void board_timer_cb(TimerHandle_t) { board_cb(); }
static void board_start_sampler(uint32_t period_ms, void (*cb)()) {
    board_cb = cb;
    board_timer.begin(period_ms, board_timer_cb);
    board_timer.start();
}

// Diagnostics of a freeze: a progress code in GPREGRET2, a watchdog that resets a stuck firmware, and the reset cause.
// POWER is the SoftDevice's once Bluetooth is on: sd_ calls then. The Adafruit bootloader runs first after a reset and
// clears GPREGRET2 and RESETREAS, so the report at boot is usually empty: build with -DDFN_TRACE to follow the codes
// live on the serial port instead (that is how the -Ofast fault of 2026-11-11 was found).
#include <nrf_soc.h>
static uint8_t board_prev_crumb = 0;
static uint32_t board_resetreas = 0;
static bool board_sd_on() { uint8_t en = 0; sd_softdevice_is_enabled(&en); return en; }
static void crumb(uint8_t s) {
#ifdef DFN_TRACE                                           // -DDFN_TRACE: each progress code on the serial port too
    if (s < 30 || s >= 40) { Serial.print("# @"); Serial.println(s); Serial.flush(); delay(20); }
#endif
    if (board_sd_on()) { sd_power_gpregret_clr(1, 0xFF); sd_power_gpregret_set(1, s); }
    else NRF_POWER->GPREGRET2 = s;
}
static void board_boot_info() {                            // before Bluetooth starts
    board_prev_crumb = (uint8_t)NRF_POWER->GPREGRET2;
    board_resetreas = NRF_POWER->RESETREAS;
    NRF_POWER->RESETREAS = 0xFFFFFFFFu;                     // write 1 to clear
    NRF_POWER->GPREGRET2 = 0;
}
static const char* board_reset_reason() {
    const uint32_t r = board_resetreas;
    if (r & 0x2) return "watchdog";
    if (r & 0x8) return "crash (lockup)";
    if (r & 0x4) return "software";
    if (r & 0x1) return "reset pin";
    if (r & 0x10000) return "wake from off";
    return "power-on or brown-out";
}
static void board_wdt_start(uint32_t seconds) {
    if (NRF_WDT->RUNSTATUS) return;                         // already running (cannot be changed until a reset)
    NRF_WDT->CONFIG = 1;                                    // keeps counting while the CPU sleeps
    NRF_WDT->CRV = seconds * 32768u;
    NRF_WDT->RREN = 1;
    NRF_WDT->TASKS_START = 1;
}
static void board_wdt_feed() { NRF_WDT->RR[0] = 0x6E524635u; }
#endif
#if defined(ARDUINO_ARCH_ESP32)
static void crumb(uint8_t) {}
static void board_wdt_feed() {}
#endif

// I2C buses: Wire = INA228, Wire1 = OLED
static void board_begin_i2c() {
#if defined(ARDUINO_ARCH_ESP32)
    Wire.begin(PIN_INA_SDA, PIN_INA_SCL, 400000);
    Wire1.begin(PIN_OLED_SDA, PIN_OLED_SCL, 400000);
#else
    Wire.setPins(PIN_INA_SDA, PIN_INA_SCL);
    Wire.begin();
    Wire.setClock(400000);
    Wire1.setPins(PIN_OLED_SDA, PIN_OLED_SCL);
    Wire1.begin();
    Wire1.setClock(400000);
#endif
}

// The INA228 lines the other way round (the first nRF52840 board has SDA on D5; schema_mini_oled has SDA on D4):
// main.cpp calls this when the INA228 does not answer in the board's default orientation, then tries again.
static bool ina_swapped = false;
static void board_swap_ina_pins() {
    ina_swapped = !ina_swapped;
    const int sda = ina_swapped ? PIN_INA_SCL : PIN_INA_SDA, scl = ina_swapped ? PIN_INA_SDA : PIN_INA_SCL;
#if defined(ARDUINO_ARCH_ESP32)
    Wire.end();
    Wire.begin(sda, scl, 400000);
#else
    Wire.end();
    Wire.setPins(sda, scl);
    Wire.begin();
    Wire.setClock(400000);
#endif
}
