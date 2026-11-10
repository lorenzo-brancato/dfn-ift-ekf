// ble_ota.h — firmware update over BLE (dashboard "Update firmware"), with rollback. ESP32-S3 only.
// Copied from BattageDiagnosticSystem/Embedded/firmware/src (same protocol, so the dashboard updates either firmware);
// ble_uart:: replaced by nus::.
//
// GATT service OTA_SERVICE, next to the Nordic UART Service:
//   CTRL (write + notify, text)   host -> "begin <size> <crc32 hex>" | "sync" | "abort"
//                                 board -> "ready <chunk> <window>" | "ack <offset>" | "done" | "err <text>"
//   DATA (write without response) [uint32 offset, little endian][chunk bytes]
// The host sends one window of bytes, then waits for "ack <offset>" and continues from
// that offset; chunks at another offset are dropped, so a lost chunk is just sent again.
// At the end the board checks the CRC32 and the image, saves its state and reboots.
//
// Rollback: an updated image boots "pending verify". It is confirmed after
// OTA_CONFIRM_S of normal running; if it resets or hangs before (loop watchdog, 5 s),
// the bootloader goes back to the previous firmware.
#pragma once
#include <Arduino.h>

class NimBLEServer;

namespace ble_ota {

constexpr const char *OTA_SERVICE = "B7A70001-5A1E-4C3B-9D1A-BA77A6E00001";
constexpr const char *OTA_CTRL    = "B7A70002-5A1E-4C3B-9D1A-BA77A6E00001";
constexpr const char *OTA_DATA    = "B7A70003-5A1E-4C3B-9D1A-BA77A6E00001";

using LogFn = void (*)(const char *line);
using HookFn = void (*)();

// Called by nus::begin before advertising starts.
void addService(NimBLEServer *server);
void onDisconnect();

// From setup(): log goes to the messages, before_reboot saves the NVS state.
// Starts the rollback check if this image was just installed.
void begin(LogFn log, HookFn before_reboot);
// From loop(): flash writes, replies, rollback confirmation, reboot.
void poll();
// An update is running: the CSV telemetry pauses on BLE to leave it the bandwidth.
bool active();
// First 8 hex digits of the running image's SHA-256 (the last 32 bytes of the .bin).
const char *firmwareId();

}  // namespace ble_ota
