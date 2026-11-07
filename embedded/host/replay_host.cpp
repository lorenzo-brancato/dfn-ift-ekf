// Host stand-in for the board: the replay protocol (dfn/replay.h) on stdin / stdout, for tools/replay.py --host-exe.
#include <chrono>
#include <cstdio>
#include <cstring>
#include "dfn/replay.h"

static dfn::Replay rp;

static unsigned long now_us() {
    using namespace std::chrono;
    return (unsigned long)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    rp.now_us = now_us;
    static char line[256], out[256];
    while (std::fgets(line, sizeof line, stdin)) {
        size_t n = std::strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n) continue;
        rp.handle(line, out);
        std::fputs(out, stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }
    return 0;
}
