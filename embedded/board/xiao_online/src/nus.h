// Nordic UART Service (NUS) over Bluetooth LE, the protocol of the Battage dashboard
// (BattageDiagnosticSystem/Embedded/dashboard/index.html): text lines, TX = notify, RX = write.
//   nrf52840: Bluefruit (in the Adafruit core); esp32s3: NimBLE-Arduino 1.4.3.
#pragma once
#include <Arduino.h>

namespace nus {
void begin(const char* name);
bool connected();
void send_line(const char* line);                    // '\n' appended, split by the link's MTU
bool read_line(char* buf, size_t len);               // one received command line, without '\n'
#if defined(ARDUINO_ARCH_ESP32)
// for the firmware update (ble_ota): payload per notification / write (ATT MTU - 3), a short connection interval
// (7.5-15 ms) while it runs, the interval in use [ms]
uint16_t payload();
void setFastLink(bool fast);
float connIntervalMs();
#endif
}
