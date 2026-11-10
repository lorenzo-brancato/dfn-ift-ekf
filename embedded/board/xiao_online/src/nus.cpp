// see nus.h
#include "nus.h"
#include <string.h>

namespace {
// received text, written by the BLE stack, read from loop()
constexpr size_t RX_CAP = 256;
char rx_buf[RX_CAP];
size_t rx_len = 0;
}

#if defined(ARDUINO_ARCH_ESP32)
#include <NimBLEDevice.h>
#include <string>
#include "ble_ota.h"

namespace {
const char* NUS_SERVICE = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
const char* NUS_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";
const char* NUS_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";
NimBLEServer* server = nullptr;
NimBLECharacteristic* tx = nullptr;
volatile uint16_t mtu_payload = 20;
volatile uint16_t conn_handle = BLE_HS_CONN_HANDLE_NONE;
portMUX_TYPE rx_mux = portMUX_INITIALIZER_UNLOCKED;

class ServerCb : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* s, ble_gap_conn_desc* d) override {
        mtu_payload = 20;
        conn_handle = d->conn_handle;
        s->setDataLen(d->conn_handle, 251);               // full-size link-layer packets, if the client supports them
    }
    void onDisconnect(NimBLEServer*) override {           // advertising restarts by itself
        mtu_payload = 20;
        conn_handle = BLE_HS_CONN_HANDLE_NONE;
        ble_ota::onDisconnect();
    }
    void onMTUChange(uint16_t mtu, ble_gap_conn_desc*) override { mtu_payload = mtu > 3 ? uint16_t(mtu - 3) : 20; }
};
class RxCb : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* c) override {
        const std::string v = c->getValue();
        portENTER_CRITICAL(&rx_mux);
        if (rx_len + v.size() <= RX_CAP) { memcpy(rx_buf + rx_len, v.data(), v.size()); rx_len += v.size(); }
        portEXIT_CRITICAL(&rx_mux);
    }
};
}  // namespace

namespace nus {
void begin(const char* name) {
    NimBLEDevice::init(name);
    NimBLEDevice::setMTU(247);
    server = NimBLEDevice::createServer();
    server->setCallbacks(new ServerCb());
    NimBLEService* svc = server->createService(NUS_SERVICE);
    tx = svc->createCharacteristic(NUS_TX, NIMBLE_PROPERTY::NOTIFY);
    NimBLECharacteristic* rx = svc->createCharacteristic(NUS_RX, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
    rx->setCallbacks(new RxCb());
    svc->start();
    ble_ota::addService(server);                           // before advertising starts the server
    NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
    adv->addServiceUUID(NUS_SERVICE);
    adv->setScanResponse(true);
    adv->setMinInterval(uint16_t(500 / 0.625f));      // 0.5-0.8 s: the dashboard finds the board within ~1-2 s
    adv->setMaxInterval(uint16_t(800 / 0.625f));
    adv->start();
}
bool connected() { return server && server->getConnectedCount() > 0; }
void send_line(const char* line) {
    if (!connected() || !tx) return;
    std::string s(line);
    s += '\n';
    const size_t chunk = mtu_payload;
    for (size_t k = 0; k < s.size(); k += chunk) {
        tx->setValue(reinterpret_cast<const uint8_t*>(s.data() + k), std::min(chunk, s.size() - k));
        tx->notify();
    }
}
uint16_t payload() { return mtu_payload; }
void setFastLink(bool fast) {                              // units: interval 1.25 ms, supervision timeout 10 ms
    if (!connected() || conn_handle == BLE_HS_CONN_HANDLE_NONE) return;
    if (fast) server->updateConnParams(conn_handle, 6, 12, 0, 400);
    else server->updateConnParams(conn_handle, 24, 48, 0, 400);
}
float connIntervalMs() {
    ble_gap_conn_desc d;
    if (conn_handle == BLE_HS_CONN_HANDLE_NONE || ble_gap_conn_find(conn_handle, &d) != 0) return 0;
    return d.conn_itvl * 1.25f;
}
#define RX_LOCK() portENTER_CRITICAL(&rx_mux)
#define RX_UNLOCK() portEXIT_CRITICAL(&rx_mux)
}  // namespace nus

#else   // nRF52840: Bluefruit
#include <Adafruit_TinyUSB.h>
#include <bluefruit.h>

namespace {
BLEUart bleuart;
void on_rx(uint16_t) {
    while (bleuart.available()) {
        int c = bleuart.read();
        taskENTER_CRITICAL();
        if (rx_len < RX_CAP) rx_buf[rx_len++] = (char)c;
        taskEXIT_CRITICAL();
    }
}
}  // namespace

namespace nus {
void begin(const char* name) {
    Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);         // MTU 247: one notification per CSV line
    Bluefruit.begin();
    Bluefruit.setName(name);
    Bluefruit.setTxPower(0);
    bleuart.begin();
    bleuart.setRxCallback(on_rx);
    Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
    Bluefruit.Advertising.addService(bleuart);
    Bluefruit.ScanResponse.addName();
    Bluefruit.Advertising.restartOnDisconnect(true);
    Bluefruit.Advertising.setInterval(800, 1280);         // 0.5-0.8 s (units of 0.625 ms)
    Bluefruit.Advertising.start(0);
}
bool connected() { return Bluefruit.connected() && bleuart.notifyEnabled(); }
void send_line(const char* line) {
    if (!connected()) return;
    bleuart.write((const uint8_t*)line, strlen(line));
    bleuart.write((const uint8_t*)"\n", 1);
}
#define RX_LOCK() taskENTER_CRITICAL()
#define RX_UNLOCK() taskEXIT_CRITICAL()
}  // namespace nus
#endif

namespace nus {
bool read_line(char* buf, size_t len) {
    bool got = false;
    RX_LOCK();
    size_t nl = 0;
    while (nl < rx_len && rx_buf[nl] != '\n' && rx_buf[nl] != '\r') ++nl;
    if (nl < rx_len) {
        const size_t n = nl < len - 1 ? nl : len - 1;
        memcpy(buf, rx_buf, n);
        buf[n] = 0;
        size_t next = nl;
        while (next < rx_len && (rx_buf[next] == '\n' || rx_buf[next] == '\r')) ++next;
        memmove(rx_buf, rx_buf + next, rx_len - next);
        rx_len -= next;
        got = true;
    } else if (rx_len == RX_CAP) {
        rx_len = 0;                                        // overlong garbage
    }
    RX_UNLOCK();
    return got;
}
}  // namespace nus
