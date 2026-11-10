// Board layer of the online firmware: pins, critical sections and the 10 Hz sampler, for the two XIAO boards.
//   nrf52840: XIAO nRF52840, INA228 as soldered on the first board (SDA = D5, SCL = D4: the reverse of the drawing).
//   esp32s3:  the Battage XIAO ESP32-S3 (BattageDiagnosticSystem/Embedded/firmware/include/config.h): INA228 SDA = D4,
//             SCL = D5; OLED on Wire1 SDA = D9, SCL = D8; DHT22 on D10.
#pragma once
#include <Arduino.h>
#include <Wire.h>

#if defined(ARDUINO_ARCH_ESP32)
#define BOARD_NAME "XIAO ESP32-S3"
static const int PIN_INA_SDA = D4, PIN_INA_SCL = D5;
static const int PIN_OLED_SDA = D9, PIN_OLED_SCL = D8;
#define HAVE_DHT 1
static const int PIN_DHT = D10;
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
#define HAVE_DHT 0
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
