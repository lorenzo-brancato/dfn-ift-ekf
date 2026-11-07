// Seeed XIAO nRF52840: replay of a recorded profile over USB serial (protocol: include/dfn/replay.h).
// Host side: embedded/tools/replay.py --port COMx ... The model is chosen in platformio.ini (DFN_MODEL_HEADER).
#include <Arduino.h>
#include "dfn/replay.h"

static dfn::Replay rp;                    // the filter's buffers are static (.bss): no heap
static char line[256], out[256];
static size_t nline = 0;

static unsigned long now_us() { return micros(); }

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    rp.now_us = now_us;
    rp.handle("?", out);
    Serial.println(out);
}

void loop() {
    while (Serial.available()) {
        int c = Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            line[nline] = 0;
            if (nline) {
                rp.handle(line, out);
                Serial.println(out);
            }
            nline = 0;
        } else if (nline < sizeof(line) - 1) {
            line[nline++] = (char)c;
        }
    }
}
