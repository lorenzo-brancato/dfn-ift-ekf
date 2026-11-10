// ble_ota.cpp — see ble_ota.h. NimBLE-Arduino 1.4.x API, Arduino Update library.
#if defined(ARDUINO_ARCH_ESP32)
#include "ble_ota.h"

#include <NimBLEDevice.h>
#include <Update.h>
#include <esp_ota_ops.h>
#include <esp_partition.h>

#include "nus.h"

// Keep the new image "pending verify" after boot (the core would confirm it at once)
extern "C" bool verifyRollbackLater() { return true; }

namespace {

constexpr size_t RING_CAP = 16384;       // >= one window + the chunks in flight
constexpr uint32_t WINDOW = 8192;        // bytes sent between two acks
constexpr uint32_t IDLE_ABORT_MS = 30000;
constexpr uint32_t OTA_CONFIRM_S = 30;   // normal running time that confirms a new image

enum class State { IDLE, RECEIVING, REBOOT };

NimBLECharacteristic *ctrl = nullptr;
ble_ota::LogFn log_fn = nullptr;
ble_ota::HookFn before_reboot_fn = nullptr;

// Shared with the NimBLE task (callbacks) under mux
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
volatile State state = State::IDLE;
uint8_t ring[RING_CAP];
size_t ring_head = 0, ring_used = 0;    // head = oldest byte
uint32_t accepted = 0;                  // next expected offset
uint32_t dropped = 0;                   // chunks at the wrong offset (sent again by the host)
uint32_t last_rx_ms = 0;
char cmd_buf[64];
bool cmd_pending = false;
volatile bool disconnected = false;

// loop() side
uint32_t img_size = 0, img_crc = 0, crc = 0, written = 0, next_ack = 0;
uint32_t reboot_at_ms = 0;
bool pending_verify = false;
char fw_id[9] = "unknown";

uint32_t crc32Update(uint32_t c, const uint8_t *p, size_t n) {   // CRC-32 (zlib), bitwise
    c = ~c;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

void msg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void msg(const char *fmt, ...) {
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (log_fn) log_fn(buf);
}

void reply(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void reply(const char *fmt, ...) {
    if (ctrl == nullptr || !nus::connected()) return;
    char buf[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    ctrl->setValue(reinterpret_cast<const uint8_t *>(buf), strlen(buf));
    ctrl->notify();
}

void stopReceiving() {
    const bool was_receiving = state == State::RECEIVING;
    portENTER_CRITICAL(&mux);
    state = State::IDLE;
    ring_head = ring_used = 0;
    portEXIT_CRITICAL(&mux);
    if (was_receiving) nus::setFastLink(false);
}

void fail(const char *why) {
    if (Update.isRunning()) Update.abort();
    stopReceiving();
    reply("err %s", why);
    msg("# OTA failed: %s (the running firmware is unchanged)", why);
}

class CtrlCallbacks : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic *c) override {
        const std::string v = c->getValue();
        portENTER_CRITICAL(&mux);
        if (!cmd_pending) {
            const size_t n = std::min(v.size(), sizeof(cmd_buf) - 1);
            memcpy(cmd_buf, v.data(), n);
            cmd_buf[n] = '\0';
            cmd_pending = true;
        }
        portEXIT_CRITICAL(&mux);
    }
};

class DataCallbacks : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic *c) override {
        const std::string v = c->getValue();
        if (v.size() < 5) return;
        const uint8_t *d = reinterpret_cast<const uint8_t *>(v.data());
        const uint32_t off = uint32_t(d[0]) | uint32_t(d[1]) << 8 | uint32_t(d[2]) << 16 | uint32_t(d[3]) << 24;
        const size_t n = v.size() - 4;
        portENTER_CRITICAL(&mux);
        if (state == State::RECEIVING && off == accepted && ring_used + n <= RING_CAP &&
            accepted + n <= img_size) {
            size_t tail = (ring_head + ring_used) % RING_CAP;
            for (size_t k = 0; k < n; ++k) {
                ring[tail] = d[4 + k];
                tail = (tail + 1 == RING_CAP) ? 0 : tail + 1;
            }
            ring_used += n;
            accepted += n;
            last_rx_ms = millis();
        } else if (state == State::RECEIVING) {
            ++dropped;
        }
        portEXIT_CRITICAL(&mux);
    }
};

void handleCommand(char *line) {
    char *cmd = strtok(line, " ");
    if (cmd == nullptr) return;
    if (!strcmp(cmd, "begin")) {
        char *a = strtok(nullptr, " "), *b = strtok(nullptr, " ");
        const esp_partition_t *next = esp_ota_get_next_update_partition(nullptr);
        if (!a || !b) { reply("err usage: begin <size> <crc32>"); return; }
        if (Update.isRunning()) Update.abort();
        stopReceiving();
        img_size = strtoul(a, nullptr, 10);
        img_crc = strtoul(b, nullptr, 16);
        if (next == nullptr || img_size < 1024 || img_size > next->size) { fail("bad size"); return; }
        if (!Update.begin(img_size, U_FLASH)) { fail(Update.errorString()); return; }
        crc = written = 0;
        next_ack = WINDOW;
        portENTER_CRITICAL(&mux);
        accepted = 0;
        dropped = 0;
        last_rx_ms = millis();
        state = State::RECEIVING;
        portEXIT_CRITICAL(&mux);
        const uint16_t chunk = uint16_t(std::min<int>(nus::payload() - 4, 508));
        nus::setFastLink(true);
        msg("# OTA: receiving %lu bytes into %s (telemetry paused on BLE, interval %.1f ms, fast requested)",
            (unsigned long)img_size, next->label, nus::connIntervalMs());
        reply("ready %u %lu", unsigned(chunk), (unsigned long)WINDOW);
    } else if (!strcmp(cmd, "sync")) {
        if (state == State::RECEIVING) reply("ack %lu", (unsigned long)accepted);
        else reply("err idle");
    } else if (!strcmp(cmd, "abort")) {
        if (Update.isRunning()) Update.abort();
        stopReceiving();
        reply("err aborted");
        msg("# OTA aborted by the host");
    } else {
        reply("err unknown command");
    }
}

void drain() {
    static uint8_t buf[4096];
    while (true) {
        size_t n = 0;
        portENTER_CRITICAL(&mux);
        n = std::min(ring_used, sizeof(buf));
        for (size_t k = 0; k < n; ++k) buf[k] = ring[(ring_head + k) % RING_CAP];
        portEXIT_CRITICAL(&mux);
        if (n == 0) return;
        // Write outside the lock: erasing a 64 kB block takes a few hundred ms
        if (Update.write(buf, n) != n) { fail(Update.errorString()); return; }
        portENTER_CRITICAL(&mux);
        ring_head = (ring_head + n) % RING_CAP;
        ring_used -= n;
        portEXIT_CRITICAL(&mux);
        crc = crc32Update(crc, buf, n);
        written += n;
        if (written >= next_ack && written < img_size) {
            next_ack += WINDOW;
            reply("ack %lu", (unsigned long)accepted);
        }
        if (written >= img_size) break;
    }
    if (crc != img_crc) { fail("CRC mismatch"); return; }
    if (!Update.end()) { fail(Update.errorString()); return; }   // checks the image, sets the boot slot
    const float itvl = nus::connIntervalMs();
    stopReceiving();
    state = State::REBOOT;
    reboot_at_ms = millis() + 1500;
    msg("# OTA: image OK (%lu bytes, %lu chunks resent, interval %.1f ms), rebooting into it",
        (unsigned long)img_size, (unsigned long)dropped, itvl);
    reply("done");
}

}  // namespace

namespace ble_ota {

void addService(NimBLEServer *server) {
    NimBLEService *svc = server->createService(OTA_SERVICE);
    ctrl = svc->createCharacteristic(OTA_CTRL, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::NOTIFY);
    ctrl->setCallbacks(new CtrlCallbacks());
    NimBLECharacteristic *data = svc->createCharacteristic(OTA_DATA, NIMBLE_PROPERTY::WRITE_NR);
    data->setCallbacks(new DataCallbacks());
    svc->start();
}

void onDisconnect() { disconnected = true; }

void begin(LogFn log, HookFn before_reboot) {
    log_fn = log;
    before_reboot_fn = before_reboot;
    uint8_t sha[32];
    const esp_partition_t *run = esp_ota_get_running_partition();
    if (run && esp_partition_get_sha256(run, sha) == ESP_OK)
        snprintf(fw_id, sizeof(fw_id), "%02x%02x%02x%02x", sha[0], sha[1], sha[2], sha[3]);
    esp_ota_img_states_t st;
    if (run && esp_ota_get_state_partition(run, &st) == ESP_OK && st == ESP_OTA_IMG_PENDING_VERIFY) {
        pending_verify = true;
        enableLoopWDT();   // a hang before the confirmation resets, and the reset rolls back
        msg("# new firmware %s on %s: confirmed after %lu s of normal running, else rollback", fw_id,
              run->label, (unsigned long)OTA_CONFIRM_S);
    }
}

void poll() {
    if (pending_verify) {
        feedLoopWDT();
        if (millis() >= OTA_CONFIRM_S * 1000u) {
            pending_verify = false;
            esp_ota_mark_app_valid_cancel_rollback();
            disableLoopWDT();
            msg("# new firmware %s confirmed", fw_id);
        }
    }

    if (disconnected) {
        disconnected = false;
        if (state == State::RECEIVING) {
            if (Update.isRunning()) Update.abort();
            stopReceiving();
            msg("# OTA aborted: BLE disconnected (the running firmware is unchanged)");
        }
    }

    char line[sizeof(cmd_buf)];
    bool have = false;
    portENTER_CRITICAL(&mux);
    if (cmd_pending) {
        memcpy(line, cmd_buf, sizeof(line));
        cmd_pending = false;
        have = true;
    }
    portEXIT_CRITICAL(&mux);
    if (have) handleCommand(line);

    if (state == State::RECEIVING) {
        drain();
        if (state == State::RECEIVING && millis() - last_rx_ms > IDLE_ABORT_MS) fail("timeout");
    } else if (state == State::REBOOT && int32_t(millis() - reboot_at_ms) >= 0) {
        if (before_reboot_fn) before_reboot_fn();
        delay(100);
        ESP.restart();
    }
}

bool active() { return state != State::IDLE; }

const char *firmwareId() { return fw_id; }

}  // namespace ble_ota
#endif  // ARDUINO_ARCH_ESP32
