// Line protocol of the replay (board sketch and host stand-in), ASCII, one command per line:
//
//   ?                        -> "# tag=<model> NZ= NX= NG= DT=<list> T=<degC> real=float|double cov=UD|Joseph bytes=<filter>"
//   R <mode> <dq>            reset: mode 'e' (EKF) or 'o' (open loop), start at q0 + dq [A.h below the top of the window]
//                            (q0: the model's CYC20 start) -> "# reset ..."
//   S <I> <V> <T> <dt>       one step: current [A, > 0 discharge], measured voltage [V] (EKF update; ignored in open
//                            loop; NaN: no update), temperature [degC] (checked against the model's, not used yet),
//                            step [s] (one of DT)
//                            -> "O <k> <V_pred> <soc> <sd_soc> <Rf> <sd_Rf> <res> <nit> <us>"
//                               V_pred = model voltage before the update, soc/sd_soc after it, us = step time
//   anything else            -> "! <message>"
// Numbers out are formatted here (no printf float support needed on the board).
#pragma once
#include <stdlib.h>
#include "dfn/ekf.h"

namespace dfn {

inline char* put_str(char* p, const char* s) { while (*s) *p++ = *s++; *p = 0; return p; }
inline char* put_int(char* p, long v) {
    char t[24]; int n = 0;
    bool neg = v < 0; unsigned long u = neg ? (unsigned long)(-v) : (unsigned long)v;
    do { t[n++] = (char)('0' + u % 10); u /= 10; } while (u);
    if (neg) *p++ = '-';
    while (n) *p++ = t[--n];
    *p = 0;
    return p;
}
// v in scientific notation with sig significant digits (sig <= 9)
inline char* put_sci(char* p, double v, int sig) {
    if (v != v) return put_str(p, "nan");
    if (v < 0) { *p++ = '-'; v = -v; }
    if (v > 1e300) return put_str(p, "inf");
    if (v == 0) { p = put_str(p, "0"); return p; }
    int e = 0;
    while (v >= 10) { v /= 10; ++e; }
    while (v < 1) { v *= 10; --e; }
    long m = 1;
    for (int i = 1; i < sig; ++i) m *= 10;
    long q = (long)(v * (double)m + 0.5);
    if (q >= 10 * m) { q /= 10; ++e; }
    char d[12]; int n = 0;
    for (int i = 0; i < sig; ++i) { d[sig - 1 - i] = (char)('0' + q % 10); q /= 10; }
    n = sig;
    *p++ = d[0];
    if (n > 1) { *p++ = '.'; for (int i = 1; i < n; ++i) *p++ = d[i]; }
    *p++ = 'e';
    return put_int(p, e);
}

class Replay {
public:
    EKF ekf;                                            // also runs the open loop (EKF::open_step)
    bool use_ekf = true, ready = false;
    long k = 0;
    unsigned long (*now_us)() = 0;

    static int idt_of(double dt) {
        for (int d = 0; d < NDT; ++d)
            if (r_abs(dt - (double)DT[d]) < 1e-6 * (1 + dt)) return d;
        return -1;
    }

    // handles one line (without the newline); writes the reply into out (>= 160 bytes)
    void handle(const char* line, char* out) {
        while (*line == ' ') ++line;
        char* p = out;
        if (line[0] == '?') {
            p = put_str(p, "# tag="); p = put_str(p, TAG);
            p = put_str(p, " NZ="); p = put_int(p, NZ); p = put_str(p, " NX="); p = put_int(p, NX);
            p = put_str(p, " NG="); p = put_int(p, NG); p = put_str(p, " DT=");
            for (int d = 0; d < NDT; ++d) { if (d) *p++ = ','; p = put_int(p, (long)DT[d]); }
            p = put_str(p, " T="); p = put_sci(p, (double)T - 273.15, 4);
            p = put_str(p, DFN_DOUBLE ? " real=double" : " real=float");
            p = put_str(p, DFN_CLOSURE_DOUBLE ? " closure=double" : " closure=float");
            p = put_str(p, DFN_COV_UD ? " cov=UD" : " cov=Joseph");
            p = put_str(p, " bytes="); p = put_int(p, (long)sizeof(EKF));
            return;
        }
        if (line[0] == 'R') {
            const char* s = line + 1;
            while (*s == ' ') ++s;
            use_ekf = *s != 'o';
            if (*s) ++s;
            double dq = strtod(s, 0);
            real_t z0[NZ];
            for (int i = 0; i < NZ; ++i) z0[i] = (real_t)((double)dfn_model::z0[i] + dq * (double)dz_dq[i]);
            ekf.init(z0, 0, EKF_IDT, 0);
            k = 0;
            ready = true;
            p = put_str(p, use_ekf ? "# reset ekf dq=" : "# reset open dq=");
            p = put_sci(p, dq, 4);
            return;
        }
        if (line[0] == 'S') {
            if (!ready) { put_str(p, "! send R first"); return; }
            char* e;
            const char* s = line + 1;
            double I = strtod(s, &e); s = e;
            double V = strtod(s, &e); s = e;
            double Tc = strtod(s, &e); s = e;
            double dt = strtod(s, &e);
            (void)Tc;
            int idt = idt_of(dt);
            if (idt < 0) { put_str(p, "! step length not in the model's DT"); return; }
            unsigned long t0 = now_us ? now_us() : 0;
            double Vp, soc, sd = 0, Rf = 1, sdRf = 0, res;
            int nit;
            if (k == 0) ekf.init(ekf.x, ekf.x[ITH], idt, (real_t)I);   // closure at the first current (EKF.init_g)
            EkfOut o = use_ekf ? ekf.step((real_t)I, (real_t)V, idt, V == V) : ekf.open_step((real_t)I, idt);
            Vp = o.V; soc = o.soc; sd = o.sd_soc; Rf = o.Rf; sdRf = o.sd_Rf; res = o.res; nit = o.nit;
            unsigned long us = now_us ? now_us() - t0 : 0;
            p = put_str(p, "O "); p = put_int(p, k++);
            *p++ = ' '; p = put_sci(p, Vp, 8);
            *p++ = ' '; p = put_sci(p, soc, 8);
            *p++ = ' '; p = put_sci(p, sd, 4);
            *p++ = ' '; p = put_sci(p, Rf, 6);
            *p++ = ' '; p = put_sci(p, sdRf, 4);
            *p++ = ' '; p = put_sci(p, res, 3);
            *p++ = ' '; p = put_int(p, nit);
            *p++ = ' '; p = put_int(p, (long)us);
            return;
        }
        put_str(p, "! unknown command");
    }
};

}  // namespace dfn
